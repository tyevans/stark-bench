"""Embed every query once, up front, in batches -- instead of one at a time.

## The defect this closes

`run_queries.run` is a bare `for query in queries` loop with an `await`
inside, and every agent's first act is a vector search whose query text is
embedded *inside* `ChunkRetriever.retrieve_chunks`. So a 280-query run made
280 separate HTTP round-trips to embed 280 short strings, each one waiting
for the last, with the GPU idle between them.

The compute involved is a couple of seconds. Measured on `qwen-rel-whole`,
`dense` took **78.7s for 280 queries** -- 0.28s each, almost none of it
embedding. It is latency and serialisation, and it is paid again by every
arm, every agent, every re-score.

## Why a wrapper rather than a change to the loop

Two alternatives were considered and are worse:

- **Concurrency in the runner.** `asyncio.gather` over the query set would
  overlap the round-trips, but the embedding peer reports `total_slots: 1`
  and the chat model runs `-np 1`, so the requests would serialise *at the
  server* and the client would merely look busy. That is the `1 x 128`
  mistake recorded in CLAUDE.md -- tuning against apparent activity rather
  than throughput -- in a new costume.
- **Batching inside the agent.** The `Agent` protocol is
  `retrieve(query, tools)`; no agent ever holds more than one query text.
  Changing that to hand agents the whole query set would make every agent
  responsible for its own batching and would put the query set inside the
  seam whose narrowness is the point.

This wrapper leaves both alone. The runner still iterates, agents still see
one query, and the round-trips are simply already done.

## What must not go wrong, and why each guard is here

**Only the query side is memoised.** `embed` passes straight through,
untouched. A cache spanning both sides would serve a document-side vector to
a query -- the exact hazard ADR 0043 exists for, and one that returns a
perfectly plausible cosine similarity rather than an error.

**Prewarming goes through `inner.embed_query`**, the same method the
per-query path calls. So whatever prefix the inner provider applies
(`query: ` for Nemotron, an instruction for BGE) is applied identically to
the prewarmed vector and the live one, *by construction* rather than by two
call sites agreeing.

This is emphatically **not** `PrefixedEmbeddingProvider` returning from the
dead. That class prepended strings the port could not express; this one
prepends nothing, knows no prefix, and would be equally correct against a
provider with no prefixes at all.

**A miss delegates rather than raises.** Unlike
`PrecomputedEmbeddingProvider`, where a miss means the control is silently
becoming a live run, a miss here is a *performance* fact, not a correctness
one: `deep` invents sub-queries that were never in the query set and must
still be able to embed them. The counters are what make a miss visible, and
`live_calls` on a `dense` run over a prewarmed set is expected to be exactly
**0** -- an assertion on the data, in the report, rather than trust that this
file was wired up.

**One vector per input text, in input order, duplicates included.** The
`EmbeddingProvider` port promises it and `EmbeddingProviderError` exists
because breaking it corrupts silently: a caller zipping results back onto
inputs attaches the wrong vector to the wrong thing and stores it happily.
Deduplication happens on the *request* to the server, never on the response
to the caller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redstring import EmbeddingProvider

logger = logging.getLogger(__name__)

#: Texts per request when prewarming. Queries are short -- a STaRK query is
#: a sentence, not a document -- so this is nowhere near the payload limits
#: that forced `--embed-batch 32` on the whole-document ingest arms
#: (B-PROXY-LIMITS-1, where a single chunk could reach 133,778 characters).
DEFAULT_PREWARM_BATCH = 128


class PrewarmedQueryEmbeddings:
    """Wraps an `EmbeddingProvider`, serving prewarmed query vectors from memory."""

    def __init__(
        self,
        inner: EmbeddingProvider,
        *,
        batch_size: int = DEFAULT_PREWARM_BATCH,
    ) -> None:
        self._inner = inner
        self._batch_size = batch_size
        self._vectors: dict[str, list[float]] = {}
        #: Query texts served from memory.
        self.hits = 0
        #: Query texts that had to be embedded live after prewarming. Zero is
        #: the expected value for every agent that only embeds the query set.
        self.misses = 0
        #: Round-trips made *after* prewarming. This is the number the defect
        #: was about; `hits` without it cannot distinguish a served vector
        #: from a served vector that also cost a request.
        self.live_calls = 0
        #: Round-trips made *during* prewarming.
        self.prewarm_requests = 0
        #: Distinct texts prewarmed. Lower than the query count when the set
        #: contains duplicates, which is a property of the data, not a bug.
        self.prewarm_texts = 0
        #: Set when `prewarm_or_log` caught a failure, and reported, so a
        #: cost column showing live calls carries its explanation.
        self.prewarm_failed = False

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """The corpus side, deliberately untouched. See the module docstring."""
        return await self._inner.embed(texts)

    async def prewarm_or_log(self, texts: Sequence[str]) -> None:
        """`prewarm`, but a failure is logged rather than fatal.

        Prewarming is an OPTIMISATION: it batches what a serial run would
        otherwise request one query at a time. A run that needs no
        embeddings must not be taken down by it, and a run that does will
        fail at its first `embed_query` a second later with the same error.

        Not hypothetical. `retrieve_chunks` embeds only for SEMANTIC and
        HYBRID -- a `lexical` arm is pure BM25 over Postgres and never
        touches the endpoint. Before this, such a run died in the prewarm
        for a capability it would never use, which is exactly the moment the
        shared inference host is most likely to be busy with something else.

        Nothing is retried or faked: `embed_query` still goes to the live
        provider and still raises.
        """
        try:
            await self.prewarm(texts)
        except Exception as error:
            self.prewarm_failed = True
            logger.warning(
                "query embedding prewarm failed (%s); continuing. A run "
                "needing embeddings will fail at its first query; a lexical "
                "run is unaffected.",
                error,
            )

    async def prewarm(self, texts: Sequence[str]) -> None:
        """Embed every distinct text once, in batches, and hold the vectors.

        Idempotent: texts already held are not re-embedded, so calling this
        twice costs one round-trip's worth of nothing.
        """
        # `dict.fromkeys` rather than `set`: order-stable, so a failure
        # mid-prewarm is reproducible rather than depending on hash seeding.
        pending = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if not pending:
            return
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            vectors = await self._inner.embed_query(batch)
            if len(vectors) != len(batch):
                # The port promises one per input. An adapter that batches,
                # retries a partial failure, or deduplicates internally can
                # break it, and zipping a short list onto the batch would
                # bind the wrong vector to the wrong query for the whole run.
                raise ValueError(
                    f"embed_query returned {len(vectors)} vectors for "
                    f"{len(batch)} texts; refusing to guess the alignment"
                )
            self.prewarm_requests += 1
            for text, vector in zip(batch, vectors, strict=True):
                self._vectors[text] = list(vector)
        self.prewarm_texts = len(self._vectors)
        logger.info(
            "prewarmed %s query vectors in %s requests",
            self.prewarm_texts,
            self.prewarm_requests,
        )

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Prewarmed vectors from memory; anything else delegated, in order."""
        if not texts:
            return []
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if missing:
            self.live_calls += 1
            vectors = await self._inner.embed_query(missing)
            if len(vectors) != len(missing):
                raise ValueError(
                    f"embed_query returned {len(vectors)} vectors for "
                    f"{len(missing)} texts; refusing to guess the alignment"
                )
            for text, vector in zip(missing, vectors, strict=True):
                self._vectors[text] = list(vector)
        # Built from `texts`, not from `missing` or from the map, so the
        # result is one vector per input in input order even when the input
        # repeats a text.
        out: list[list[float]] = []
        for text in texts:
            if text in missing:
                self.misses += 1
            else:
                self.hits += 1
            out.append(list(self._vectors[text]))
        return out

    def stats(self) -> dict[str, int]:
        """The counters, for the report. See the module docstring on `live_calls`."""
        return {
            "query_embed_prewarm_failed": int(self.prewarm_failed),
            "query_embed_prewarm_texts": self.prewarm_texts,
            "query_embed_prewarm_requests": self.prewarm_requests,
            "query_embed_hits": self.hits,
            "query_embed_misses": self.misses,
            "query_embed_live_calls": self.live_calls,
        }

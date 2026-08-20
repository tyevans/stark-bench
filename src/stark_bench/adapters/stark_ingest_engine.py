"""STaRK's SKB into redstring's stores.

The loader is a projection: it reads a knowledge base someone else built and
writes it through redstring's ports. It invents nothing and fetches nothing.
No extraction runs and no LLM is involved, so provenance records
`ExtractionMethod.MANUAL` -- the honest value.

Two ordering constraints come from the ports themselves:

- `upsert_relationships` raises `MissingEntityError` if an endpoint is
  missing, so entities are written before the relationships referencing them.
- Self-loops are rejected by validation. They are dropped here and *counted*,
  because a silent drop turns a recall ceiling into an apparent retrieval
  failure.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from redstring import (
    Entity,
    ExtractionMethod,
    Provenance,
    Relationship,
    RelationshipId,
    SourceId,
)
from redstring.domain.chunk import StoredChunk, chunk_id

from stark_bench.domain.stark_ids import STARK_ID_KEY, entity_id_for

logger = logging.getLogger(__name__)

#: Seconds between progress lines. Time-based rather than every-N-nodes
#: because node cost varies by two orders of magnitude across these corpora
#: -- a MAG author is one short chunk and a PRIME pathway is dozens -- so a
#: node counter goes quiet for minutes on the expensive stretches, which is
#: exactly when someone is watching.
PROGRESS_EVERY_SECONDS = 30.0

#: Substrings that identify "this input had too many tokens" in the provider
#: error. Matched on the message because `EmbeddingProviderError` flattens the
#: client exception into text -- the structured `n_prompt_tokens` the server
#: sends is not reachable from here.
#:
#: Both spellings are listed because llama.cpp emits the `type` field and
#: OpenAI-compatible servers vary in the prose; matching either is cheap and
#: matching neither means a real error propagates, which is the safe default.
_OVERSIZE_MARKERS = ("exceed_context_size_error", "larger than the max context size")

#: How many times a group may be re-chunked at half the size before giving up.
#: Four halvings take a 2400-character cap to 150, which is below any sane
#: floor -- if the text still does not fit by then the problem is not the cap.
MAX_RESPLIT_ATTEMPTS = 4

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from collections.abc import Set as AbstractSet

    from redstring import EmbeddingProvider, TenantId
    from redstring.domain.chunk import ChunkId

    from stark_bench.adapters.stark_artifacts import SkbEdge, SkbNode

BATCH = 500

#: Chunks are flushed independently of `BATCH`, which bounds the *entity*
#: count per flush. A skipped node (resume path) still counts toward
#: `BATCH` without adding a single chunk, while a non-skipped node with a
#: real chunker (`BoundaryPreferenceChunker`, not the 1:1 `WholeDocumentChunker`
#: control) can contribute many chunks per node -- so `len(batch) >= BATCH`
#: can go a long time between flushes while `chunk_batch` keeps growing.
#: `PostgresChunkStore.upsert_many` serialises the whole batch into one
#: `jsonb` parameter, and Postgres rejects a jsonb array once its total
#: element size passes 268,435,455 bytes (`ProgramLimitExceededError`).
#: Measured against a live Postgres with 768-dim embeddings and 2000-char
#: chunk text (`BoundaryPreferenceChunker`'s ceiling per chunk): 15,000
#: chunks succeeded, 16,000 failed. 1,000 is committed here for roughly a
#: 15x margin under that measured ceiling.
CHUNK_BATCH = 1000


@dataclass(frozen=True, slots=True)
class IngestReport:
    nodes: int
    edges: int
    self_loops_dropped: int
    chunks: int
    skipped: int = 0


def _entity(
    node: SkbNode, *, dataset: str, tenant_id: TenantId, observed_at: datetime
) -> Entity:
    return Entity(
        id=entity_id_for(dataset, node.node_id),
        tenant_id=tenant_id,
        name=node.name,
        normalized_name=node.name.casefold(),
        entity_type=node.node_type,
        external_ids={STARK_ID_KEY: node.node_id},
        provenance=Provenance(
            observed_at=observed_at,
            extraction_method=ExtractionMethod.MANUAL,
            confidence=1.0,
        ),
    )


async def ingest(
    nodes: Iterable[SkbNode],
    edges: Iterable[SkbEdge],
    *,
    dataset: str,
    tenant_id: TenantId,
    graph,
    chunks,
    chunker,
    embeddings: EmbeddingProvider | None = None,
    vector_for: Callable[[str], list[float]] | None = None,
    concurrency: int = 1,
    embed_batch: int = 64,
    existing_chunk_ids: AbstractSet[ChunkId] = frozenset(),
    resume: bool = True,
    total_nodes: int | None = None,
) -> IngestReport:
    """Load nodes (and their chunks) then edges, in that order.

    Chunk vectors come from exactly one of two sources:

    - `embeddings`: `EmbeddingProvider.embed(texts)`, live embedding.
    - `vector_for`: a per-node-id lookup (STaRK's precomputed vectors), used
      as-is with no embedding call at all. `WholeDocumentChunker` makes this a
      clean 1:1, one chunk per node. A miss is the caller's `vector_for` to
      raise on -- this function does not catch it.

    Exactly one must be given; both or neither is a caller error.

    ## Two knobs, and the second is the one that matters

    `embed_batch` is how many chunk texts go into **one** embedding request.
    `concurrency` is how many such requests are in flight at once.

    An earlier version issued one request *per node* and relied on
    `concurrency` alone, which on a corpus averaging 1.06 chunks per node
    means every request carried roughly one text. Batching is worth about
    **18%** over that, measured end to end on 3000 nodes at concurrency 8:
    1368 nodes/min at one text per request, 1612 at sixty-four.

    A standalone probe over the same range on a **single serial connection**
    gave 298 against 1850 texts/min, and that six-fold figure is recorded
    here only to be disbelieved. The ingest was never serial: eight requests
    were already in flight, which recovers most of the same ground by hiding
    round-trip latency. Comparing "no pipelining at all" to "batching plus
    pipelining" credits the whole gap to batching. Roughly 4.6x of it is
    concurrency and 1.18x is batching.

    The general form of that mistake is worth more than the number: **the
    baseline in a speedup claim has to be the thing you are actually
    replacing.**

    ## Which knob wins depends on the endpoint, so keep both

    **Client concurrency should be at least the server's `-np`.** That is
    the whole rule, and it took seven measurements and three wrong
    explanations to arrive at. Against llama.cpp with `-np 4`, over 3000
    nodes:

    | concurrency | batch | slots used | nodes/min |
    |---|---|---|---|
    | 1 | 128 | 1 of 4 | 1233 |
    | 2 | 1 | 2 of 4 | 1312 |
    | 8 | 1 | 4 of 4 | 1368 |
    | 2 | 16 | 2 of 4 | 1535 |
    | 2 | 128 | 2 of 4 | 1447 |
    | 8 | 64 | 4 of 4 | 1612 |
    | 4 | 128 | 4 of 4 | 1618 |

    A request occupies one slot, so `concurrency=1` leaves three quarters of
    the server idle no matter how large the batch -- and it was the slowest
    of the seven despite carrying 128 texts per request. Slots scale
    sublinearly (1 -> 2 is +17%, 2 -> 4 is +12%), so one slot already
    partially saturates the device and there is little beyond four.

    Batching raises the per-slot ceiling and is worth about 18% at fixed
    concurrency; concurrency decides how many slots are working at all.
    Neither substitutes for the other.

    ## Do not tune against GPU utilisation

    It was the most misleading signal in this investigation. `nvidia-smi`
    reported its highest number on `1 x 128` -- the slowest configuration
    measured -- because a kernel is resident whenever any slot is busy, and
    three idle slots look exactly like none. Utilisation answers "is the
    device doing something", not "is the device doing as much as it could".

    On a hosted API or a multi-replica server the balance shifts further
    toward concurrency still, since round-trip latency dominates and a
    provider may cap request size outright.

    `vector_for` does no I/O and ignores both.

    A batch is bounded by texts, not tokens, and that is safe here for a
    reason worth stating: llama.cpp splits an over-large *request* across
    decode batches by itself, and only a single **sequence** is bounded by
    `--ubatch-size`. The chunker's own size cap is what keeps a sequence
    inside that, so no batch size chosen here can produce a 413.

    ## Resuming an interrupted ingest

    `existing_chunk_ids` is the set of chunk ids this tenant's store already
    holds, loaded by the caller in one query up front -- this function never
    queries for it and never issues one lookup per node. A node is skipped
    (no chunking cost paid twice, no embedding call, no chunk write) only
    when **every** chunk it would produce already has an id in that set.
    Chunk ids are content-addressed over `(source_id, text)`
    (`redstring.domain.chunk.chunk_id`), so a node whose document text
    changed produces a different id, is not found in the set, and is
    re-embedded -- the skip is keyed on content, never on the node id alone.

    The node's entity is always written, skip or not: that upsert is a cheap,
    idempotent no-op and doing it unconditionally means a node's presence in
    the graph is never made to depend on this optimisation.

    `resume=False` ignores `existing_chunk_ids` entirely and re-embeds every
    node, for a deliberate full re-ingest.
    """
    if (embeddings is None) == (vector_for is None):
        raise ValueError("ingest needs exactly one of embeddings or vector_for")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if embed_batch < 1:
        raise ValueError("embed_batch must be at least 1")
    observed_at = datetime.now(UTC)
    known: set[str] = set()
    node_count = chunk_count = skipped_count = 0
    started_at = time.monotonic()
    last_report = started_at

    def _report(final: bool = False) -> None:
        elapsed = time.monotonic() - started_at
        rate = chunk_count / elapsed if elapsed > 0 else 0.0
        node_rate = node_count / elapsed if elapsed > 0 else 0.0
        eta = ""
        if total_nodes and node_rate > 0 and not final:
            remaining = (total_nodes - node_count) / node_rate
            eta = f" eta {remaining / 60:.0f}m"
        pct = f" ({node_count / total_nodes:.0%})" if total_nodes else ""
        logger.info(
            "ingest %s: %s/%s nodes%s, %s chunks (%s skipped), "
            "%.0f chunks/s, %.0f nodes/s, %.0fm elapsed%s",
            "done" if final else "progress",
            f"{node_count:,}",
            f"{total_nodes:,}" if total_nodes else "?",
            pct,
            f"{chunk_count:,}",
            f"{skipped_count:,}",
            rate,
            node_rate,
            elapsed / 60,
            eta,
        )

    batch: list[Entity] = []
    chunk_batch: list[StoredChunk] = []

    def _plan(
        node: SkbNode, max_chunk_size: int | None = None
    ) -> tuple[SkbNode, list, bool]:
        """Chunk a node and decide whether it needs vectors. No I/O.

        Split out from the embedding so that a whole group's texts can go
        into one request -- see the `embed_batch` note in the docstring.

        `max_chunk_size` overrides the chunker's own cap. It is only passed by
        the re-split path below, which re-chunks a whole group smaller after
        the provider rejected one of its texts for length.
        """
        result = chunker.chunk(node.document, max_chunk_size=max_chunk_size)
        source_id = SourceId(f"{dataset}:{node.node_id}")
        ids = [chunk_id(source_id, piece.text) for piece in result.chunks]
        wanted = not (resume and ids and all(i in existing_chunk_ids for i in ids))
        return node, result.chunks, wanted

    async def _embed_group(texts: list[str]) -> list[list[float]]:
        """One flat list of texts in, one flat list of vectors out, in order.

        Slices into `embed_batch`-sized requests and runs `concurrency` of
        them at a time. The result is reassembled by slice index rather than
        by completion order, because `asyncio.gather` preserves argument
        order but a future reader should not have to know that to trust the
        alignment -- and misaligning vectors with texts is a defect that
        produces a fully-populated store scoring like noise, with nothing
        raising anywhere.
        """
        if not texts:
            return []
        slices = [texts[i : i + embed_batch] for i in range(0, len(texts), embed_batch)]
        out: list[list[float]] = []
        for start in range(0, len(slices), concurrency):
            wave = slices[start : start + concurrency]
            for result in await asyncio.gather(*(embeddings.embed(s) for s in wave)):
                out.extend(result)
        if len(out) != len(texts):
            raise ValueError(
                f"embedding provider returned {len(out)} vectors for {len(texts)} "
                "texts; the port promises one per input, in order"
            )
        return out

    def _is_oversize(error: Exception) -> bool:
        text = str(error)
        return any(marker in text for marker in _OVERSIZE_MARKERS)

    async def _embed_planned(group, planned):
        """Embed a planned group, re-chunking smaller if the provider says no.

        The chunker caps **characters**; the provider caps **tokens**. That
        conversion is a per-corpus, per-tokenizer property nothing here can
        know, and estimating it went wrong three times on this benchmark --
        5000 characters, then 4000, then 2400, each from a defensible ratio
        and each rejected at ~2080 tokens. Every failure cost a full re-ingest.

        So the cap is no longer required to be right. When the provider
        rejects a text for length, the whole group is re-chunked at half the
        size and retried. Re-chunking the *group* rather than the offending
        text keeps chunk boundaries a pure function of the chunker and the
        document -- splitting one text in place would produce pieces whose
        `start_char`/`chunk_index` did not come from the chunker, and those
        offsets are what `chunk_id` is built from.

        This runs only on rejection, so a correct cap costs nothing.
        """
        size = None
        for attempt in range(MAX_RESPLIT_ATTEMPTS + 1):
            flat = [p.text for _, pieces, wanted in planned if wanted for p in pieces]
            try:
                return planned, await _embed_group(flat)
            except Exception as error:
                if not _is_oversize(error) or attempt == MAX_RESPLIT_ATTEMPTS:
                    raise
                longest = max(
                    (
                        len(p.text)
                        for _, pieces, wanted in planned
                        if wanted
                        for p in pieces
                    ),
                    default=0,
                )
                # Seeded from the longest piece actually produced, not from a
                # chunker attribute: neither chunker here exposes its cap, and
                # reading one would have raised AttributeError on the first
                # re-split -- the only path this code exists for.
                size = min(size or longest, longest) // 2
                if size < 1:
                    raise
                logger.warning(
                    "embed rejected a chunk as too long; re-chunking this group "
                    "at %s characters (attempt %s of %s): %s",
                    size,
                    attempt + 1,
                    MAX_RESPLIT_ATTEMPTS,
                    error,
                )
                planned = [_plan(node, max_chunk_size=size) for node in group]
        raise AssertionError("unreachable: the loop returns or raises")

    group_size = concurrency * embed_batch if embeddings is not None else 1
    node_iter = iter(nodes)
    while True:
        group = list(itertools.islice(node_iter, group_size))
        if not group:
            break

        planned = [_plan(node) for node in group]

        if vector_for is not None:
            processed = [
                (
                    node,
                    pieces,
                    [vector_for(node.node_id) for _ in pieces] if wanted else None,
                )
                for node, pieces, wanted in planned
            ]
        else:
            planned, vectors = await _embed_planned(group, planned)
            cursor = 0
            processed = []
            for node, pieces, wanted in planned:
                if not wanted:
                    processed.append((node, pieces, None))
                    continue
                processed.append((node, pieces, vectors[cursor : cursor + len(pieces)]))
                cursor += len(pieces)

        for node, pieces, vectors in processed:
            batch.append(
                _entity(
                    node, dataset=dataset, tenant_id=tenant_id, observed_at=observed_at
                )
            )
            known.add(node.node_id)
            node_count += 1

            if vectors is None:
                skipped_count += len(pieces)
                continue

            source_id = SourceId(f"{dataset}:{node.node_id}")
            for piece, vector in zip(pieces, vectors, strict=True):
                chunk_batch.append(
                    StoredChunk(
                        id=chunk_id(source_id, piece.text),
                        tenant_id=tenant_id,
                        source_id=source_id,
                        text=piece.text,
                        chunk_index=piece.chunk_index,
                        start_char=piece.start_char,
                        end_char=piece.end_char,
                        entity_ids=[entity_id_for(dataset, node.node_id)],
                        metadata={STARK_ID_KEY: node.node_id},
                        embedding=list(vector),
                    )
                )
                chunk_count += 1

            # Inside the per-node loop, not outside it. When the group was one
            # node this was the same place; batching made `group_size`
            # concurrency * embed_batch, and a group of 64 nodes at 500 chunks
            # each accumulated a single 5000-chunk upsert. `PostgresChunkStore`
            # serialises a batch into one jsonb parameter and Postgres rejects
            # that array past 268,435,455 bytes -- at 2048 dimensions, 5000
            # chunks is roughly 255 MB, so this was one corpus away from
            # `ProgramLimitExceededError` on a run that had already cost hours.
            #
            # Caught by test_chunk_batch_is_flushed_independently_of_entity_batch,
            # which existed for exactly this and is why the flush bound is
            # asserted on the observed call sizes rather than on the constant.
            if len(batch) >= BATCH or len(chunk_batch) >= CHUNK_BATCH:
                await graph.upsert_entities(batch)
                await chunks.upsert_many(chunk_batch)
                batch, chunk_batch = [], []

        # Outside the per-node loop but inside the group loop: one check per
        # group rather than per node, and it still fires on schedule because
        # a group is bounded by concurrency * embed_batch.
        if time.monotonic() - last_report >= PROGRESS_EVERY_SECONDS:
            _report()
            last_report = time.monotonic()

    if batch:
        await graph.upsert_entities(batch)
    if chunk_batch:
        await chunks.upsert_many(chunk_batch)

    _report(final=True)

    dropped = 0
    edge_count = 0
    rels: list[Relationship] = []
    for edge in edges:
        if edge.source == edge.target:
            dropped += 1
            continue
        if edge.source not in known or edge.target not in known:
            continue
        rels.append(
            Relationship(
                id=RelationshipId(uuid4()),
                tenant_id=tenant_id,
                source_entity_id=entity_id_for(dataset, edge.source),
                target_entity_id=entity_id_for(dataset, edge.target),
                relationship_type=edge.relation,
                confidence=1.0,
            )
        )
        edge_count += 1
        if len(rels) >= BATCH:
            await graph.upsert_relationships(rels)
            rels = []
    if rels:
        await graph.upsert_relationships(rels)

    return IngestReport(
        nodes=node_count,
        edges=edge_count,
        self_loops_dropped=dropped,
        chunks=chunk_count,
        skipped=skipped_count,
    )

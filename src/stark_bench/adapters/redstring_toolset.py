"""The instrumented, reader-only surface an agent sees.

Two things are deliberate. First, everything an agent touches is a *reader*:
no writer method is reachable, which is a type-level guarantee rather than a
matter of discipline. Second, every call is timed and counted, because cost is
a reported metric -- a deep agent buying four points of Hit@1 for forty times
the tokens is a different finding depending on which number you needed, and
Hit@1 alone cannot express it.

Traversal comes from `RelationshipStore`, not from `Retriever`: redstring's
`Retriever` holds `EntityReader` only and has no traversal at all.

The LLM seam is `extract(prompt, schema)`, a direct pass-through to
`LlmProvider.extract`: the port's only method is structured extraction
against a caller-supplied pydantic schema, and there is no free-text
`complete`/`.text` shape on it to fake. A typed result is also what an agent
wants -- parsing prose out of a completion is where that kind of thing breaks
first.

`ToolCall.tokens` is `int | None`, not defaulted to zero. `LlmProvider`
itself reports no usage, so today's `extract` always records `None` here --
but usage *is* measurable one layer below the port: `LangChainLlmProvider`
holds a LangChain `BaseChatModel`, and `ainvoke` on that populates
`AIMessage.usage_metadata` for OpenAI-compatible endpoints. A counting
wrapper around that chat model belongs with the CLI wiring, where a real
endpoint is configured -- not here, and not invented. Leaving `tokens` as
`None` rather than `0` keeps "the endpoint reported no usage" distinguishable
from "this call used no tokens", which a default of zero could not express.
"""

from __future__ import annotations

import math
import re
from time import perf_counter
from typing import TYPE_CHECKING

from redstring import ChunkRetriever, RetrievalMode

from stark_bench.domain.aggregation import aggregate
from stark_bench.domain import Passage, Ranked, ToolCall
from stark_bench.domain.stark_ids import STARK_ID_KEY, entity_id_for, node_id_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from redstring import EmbeddingProvider, LlmProvider, TenantId

MODES = {
    "semantic": RetrievalMode.SEMANTIC,
    "lexical": RetrievalMode.LEXICAL,
    "hybrid": RetrievalMode.HYBRID,
}


#: Texts per embedding request in `rank_texts`. The names are ~21
#: characters, so a large batch is cheap and the round trip dominates.
_RANK_EMBED_BATCH = 256

#: RRF's damping constant. 60 is the value from the original paper and is
#: not tuned here: tuning it against our own arms would fit the constant to
#: the measurement it is supposed to inform.
_RRF_K = 60

_RANK_WORD = re.compile(r"[a-z0-9]+")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    # A zero vector has no direction, so no similarity is defined. Zero is
    # the neutral answer; raising would take down a run over one degenerate
    # embedding.
    return dot / (na * nb) if na and nb else 0.0


def _bm25_scores(query: str, texts: Sequence[str]) -> list[float]:
    """BM25 of each text against the query, idf relative to `texts`."""
    docs = [_RANK_WORD.findall(t.lower()) for t in texts]
    terms = set(_RANK_WORD.findall(query.lower()))
    if not terms:
        return [0.0] * len(texts)
    total = len(docs)
    avg = sum(len(d) for d in docs) / total or 1.0
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc) & terms:
            df[term] = df.get(term, 0) + 1
    out: list[float] = []
    for doc in docs:
        score = 0.0
        length = len(doc) or 1
        for term in terms:
            freq = doc.count(term)
            if not freq:
                continue
            idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * length / avg)))
        out.append(score)
    return out


def _rrf(*channels: Sequence[float]) -> list[float]:
    """Reciprocal rank fusion of several score lists over the same items.

    Combines RANKS, not scores, so a channel cannot dominate by being
    differently scaled -- cosine sits on [-1, 1] while BM25 is unbounded and
    grows with idf. Ties take the same rank, so a channel that scores
    everything equally contributes a constant and drops out rather than
    injecting the order it happened to receive its inputs in.
    """
    size = len(channels[0])
    fused = [0.0] * size
    for scores in channels:
        order = sorted(range(size), key=lambda i: -scores[i])
        rank = 0
        previous: float | None = None
        for position, index in enumerate(order):
            if previous is None or scores[index] != previous:
                rank = position
                previous = scores[index]
            fused[index] += 1.0 / (_RRF_K + rank + 1)
    return fused


class RedstringToolset:
    """Satisfies `Toolset` over redstring's read ports."""

    def __init__(
        self,
        *,
        chunks,
        graph,
        embeddings: EmbeddingProvider,
        tenant_id: TenantId,
        dataset: str,
        llm: LlmProvider | None = None,
        aggregation: str = "max",
    ) -> None:
        self._chunks = chunks
        self._graph = graph
        self._tenant = tenant_id
        self._dataset = dataset
        self._llm = llm
        self._aggregation = aggregation
        self._embeddings = embeddings
        self._retriever = ChunkRetriever(embeddings=embeddings, chunks=chunks)
        # Entity names repeat heavily across queries -- 47,318 distinct
        # names serve ~3,300 per query over 280 queries -- so without a memo
        # this would re-embed the same strings hundreds of times. Process
        # local and unbounded: the whole distinct set is ~47k vectors, and
        # the run is minutes.
        self._text_vectors: dict[str, list[float]] = {}
        self.calls: list[ToolCall] = []

    def _record(
        self, tool: str, started: float, count: int, tokens: int | None = None
    ) -> None:
        self.calls.append(
            ToolCall(
                tool=tool,
                duration_s=perf_counter() - started,
                result_count=count,
                tokens=tokens,
            )
        )

    async def search_chunks(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Ranked]:
        """Retrieve chunks and fold them up to STaRK nodes.

        Overfetches chunks relative to `k`, because several chunks may belong
        to one node and folding shrinks the list.
        """
        started = perf_counter()
        result = await self._retriever.retrieve_chunks(
            text, self._tenant, k=k * 4, mode=MODES[mode]
        )
        scored = [
            (str(match.chunk.metadata[STARK_ID_KEY]), match.score)
            for match in result.matches
            if STARK_ID_KEY in match.chunk.metadata
        ]
        ranked = aggregate(scored, strategy=self._aggregation)[:k]
        self._record("search_chunks", started, len(ranked))
        return ranked

    async def search_passages(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Passage]:
        """Retrieve candidates and keep the text that retrieved each one.

        Folds to one passage per node like `search_chunks`, and keeps the
        text of that node's *best-scoring* chunk rather than concatenating
        its chunks. Concatenating would hand a reranker a different amount
        of evidence per candidate -- a hub node with forty chunks would get
        forty times the page space of a leaf -- and length is exactly the
        confound a reranker is supposed to be immune to.
        """
        started = perf_counter()
        result = await self._retriever.retrieve_chunks(
            text, self._tenant, k=k * 4, mode=MODES[mode]
        )
        best: dict[str, tuple[str, float]] = {}
        for match in result.matches:
            if STARK_ID_KEY not in match.chunk.metadata:
                continue
            node_id = str(match.chunk.metadata[STARK_ID_KEY])
            held = best.get(node_id)
            if held is None or match.score > held[1]:
                best[node_id] = (match.chunk.text, match.score)
        passages = [Passage(node_id=n, text=t, score=sc) for n, (t, sc) in best.items()]
        passages.sort(key=lambda p: (-p.score, p.node_id))
        passages = passages[:k]
        self._record("search_passages", started, len(passages))
        return passages

    async def get_node(self, node_id: str) -> dict[str, object] | None:
        started = perf_counter()
        entity = await self._graph.get_entity(
            entity_id_for(self._dataset, node_id), self._tenant
        )
        self._record("get_node", started, 0 if entity is None else 1)
        if entity is None:
            return None
        return {
            "node_id": node_id,
            "name": entity.name,
            "node_type": entity.entity_type,
        }

    async def neighbors(self, node_id: str, *, depth: int = 1) -> list[str]:
        started = perf_counter()
        found = await self._graph.neighbors(
            entity_id_for(self._dataset, node_id), self._tenant, depth=depth
        )
        ids = [node_id_of(entity) for entity in found]
        self._record("neighbors", started, len(ids))
        return ids

    async def get_relationships(self, node_id: str) -> list[tuple[str, str, str]]:
        """Edges as `(source_node_id, relation, target_node_id)`.

        `neighbors` returns entities with no edge type and no hop distance, so
        an agent that needs to know *how* two nodes connect calls this instead.
        """
        started = perf_counter()
        entity_id = entity_id_for(self._dataset, node_id)
        rels = await self._graph.get_relationships(entity_id, self._tenant)
        ids = {r.source_entity_id for r in rels} | {r.target_entity_id for r in rels}
        entities = await self._graph.get_entities(list(ids), self._tenant)
        lookup = {e.id: node_id_of(e) for e in entities}
        edges = [
            (
                lookup[r.source_entity_id],
                r.relationship_type,
                lookup[r.target_entity_id],
            )
            for r in rels
            if r.source_entity_id in lookup and r.target_entity_id in lookup
        ]
        self._record("get_relationships", started, len(edges))
        return edges

    async def rank_texts(
        self, query: str, texts: Sequence[str], *, mode: str = "hybrid"
    ) -> list[float]:
        """Score arbitrary strings against a query. See `ports.agent`.

        ## Why the adapter owns this rather than the agent

        An agent cannot embed: `Toolset` is its whole world and it may not
        import `harness`. Putting the mechanism here also keeps ADR 0043
        where it belongs -- the query goes through `embed_query` and the
        texts through `embed`, because the two sides of an asymmetric model
        take different prefixes and a caller that has to remember which is a
        caller that will eventually forget.

        ## Why reciprocal rank fusion rather than a weighted sum

        Cosine similarity lives on [-1, 1] and BM25 is unbounded and scales
        with idf, so summing them makes the weight depend on the corpus.
        RRF combines *ranks*, so neither channel can dominate by being
        differently scaled, and it needs no tuning constant that would
        itself have to be measured.

        ## Cost

        The dense half embeds every distinct text once, ever, into a process
        local memo. The lexical half is pure CPU. A `mode="lexical"` call
        touches no endpoint at all, which is what makes the
        lexical-versus-hybrid comparison cheap enough to be worth running.
        """
        if not texts:
            return []
        started = perf_counter()
        unique = list(dict.fromkeys(texts))

        lexical = _bm25_scores(query, unique)
        if mode == "lexical":
            by_text = dict(zip(unique, lexical, strict=True))
            self._record("rank_texts", started, len(unique))
            return [by_text[t] for t in texts]
        if mode not in ("hybrid", "dense"):
            raise ValueError(f"unknown rank_texts mode {mode!r}")

        missing = [t for t in unique if t not in self._text_vectors]
        for start in range(0, len(missing), _RANK_EMBED_BATCH):
            batch = missing[start : start + _RANK_EMBED_BATCH]
            vectors = await self._embeddings.embed(batch)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embed returned {len(vectors)} vectors for {len(batch)} "
                    "texts; refusing to guess the alignment"
                )
            for text, vector in zip(batch, vectors, strict=True):
                self._text_vectors[text] = list(vector)
        (query_vector,) = await self._embeddings.embed_query([query])
        dense = [_cosine(query_vector, self._text_vectors[t]) for t in unique]

        if mode == "dense":
            fused = dense
        else:
            fused = _rrf(dense, lexical)
        by_text = dict(zip(unique, fused, strict=True))
        self._record("rank_texts", started, len(unique))
        return [by_text[t] for t in texts]

    async def extract[S: BaseModel](self, prompt: str, schema: type[S]) -> S:
        if self._llm is None:
            raise RuntimeError("this toolset was built without an LLM provider")
        started = perf_counter()
        result = await self._llm.extract(prompt, schema)
        # `LlmProvider.extract` reports no usage; a counting wrapper one
        # layer below the port (a real chat model) is what would fill this
        # in. `None` here is honest -- not a synthesised 0.
        self._record("extract", started, 1, tokens=None)
        return result

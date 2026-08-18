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

Token counts are not recorded on any `ToolCall`: `LlmProvider.extract`
reports no usage data, so a `tokens` field could only ever be a fabricated
zero. LLM cost is counted in *calls* instead -- reporting distinguishes tool
calls from LLM calls by `tool == "extract"`.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from redstring import ChunkRetriever, RetrievalMode

from stark_bench.harness.aggregate import aggregate
from stark_bench.ports import Ranked, ToolCall
from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for, node_id_of

if TYPE_CHECKING:
    from pydantic import BaseModel

    from redstring import EmbeddingProvider, LlmProvider, TenantId

MODES = {
    "semantic": RetrievalMode.SEMANTIC,
    "lexical": RetrievalMode.LEXICAL,
    "hybrid": RetrievalMode.HYBRID,
}


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
        self._retriever = ChunkRetriever(embeddings=embeddings, chunks=chunks)
        self.calls: list[ToolCall] = []

    def _record(self, tool: str, started: float, count: int) -> None:
        self.calls.append(
            ToolCall(
                tool=tool,
                duration_s=perf_counter() - started,
                result_count=count,
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

    async def extract[S: BaseModel](self, prompt: str, schema: type[S]) -> S:
        if self._llm is None:
            raise RuntimeError("this toolset was built without an LLM provider")
        started = perf_counter()
        result = await self._llm.extract(prompt, schema)
        self._record("extract", started, 1)
        return result

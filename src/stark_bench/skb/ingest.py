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

from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from redstring import EmbeddingProvider, TenantId

    from stark_bench.skb.artifacts import SkbEdge, SkbNode

BATCH = 500


@dataclass(frozen=True, slots=True)
class IngestReport:
    nodes: int
    edges: int
    self_loops_dropped: int
    chunks: int


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
) -> IngestReport:
    """Load nodes (and their chunks) then edges, in that order.

    Chunk vectors come from exactly one of two sources:

    - `embeddings`: `EmbeddingProvider.embed(texts)`, live embedding.
    - `vector_for`: a per-node-id lookup (STaRK's precomputed vectors), used
      as-is with no embedding call at all. `WholeDocumentChunker` makes this a
      clean 1:1, one chunk per node. A miss is the caller's `vector_for` to
      raise on -- this function does not catch it.

    Exactly one must be given; both or neither is a caller error.

    `concurrency` bounds how many nodes are chunked-and-embedded at once. It
    only matters for the `embeddings` branch -- a live endpoint is shared
    infrastructure, and processing nodes one at a time against it wastes the
    round trip latency for no reason. `vector_for` does no I/O, so it always
    runs one node at a time regardless of this value; raising it there would
    only reorder writes into batches without buying anything.
    """
    if (embeddings is None) == (vector_for is None):
        raise ValueError("ingest needs exactly one of embeddings or vector_for")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    observed_at = datetime.now(UTC)
    known: set[str] = set()
    node_count = chunk_count = 0

    batch: list[Entity] = []
    chunk_batch: list[StoredChunk] = []

    async def _process(node: SkbNode) -> tuple[SkbNode, list, list]:
        result = chunker.chunk(node.document)
        if vector_for is not None:
            vectors = [vector_for(node.node_id) for _ in result.chunks]
        else:
            texts = [c.text for c in result.chunks]
            vectors = await embeddings.embed(texts) if texts else []
        return node, result.chunks, vectors

    group_size = concurrency if embeddings is not None else 1
    node_iter = iter(nodes)
    while True:
        group = list(itertools.islice(node_iter, group_size))
        if not group:
            break
        processed = await asyncio.gather(*(_process(node) for node in group))
        for node, pieces, vectors in processed:
            batch.append(
                _entity(
                    node, dataset=dataset, tenant_id=tenant_id, observed_at=observed_at
                )
            )
            known.add(node.node_id)
            node_count += 1

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

        if len(batch) >= BATCH:
            await graph.upsert_entities(batch)
            await chunks.upsert_many(chunk_batch)
            batch, chunk_batch = [], []

    if batch:
        await graph.upsert_entities(batch)
    if chunk_batch:
        await chunks.upsert_many(chunk_batch)

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
    )

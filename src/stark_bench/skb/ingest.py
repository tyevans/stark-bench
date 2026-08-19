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
    from collections.abc import Set as AbstractSet

    from redstring import EmbeddingProvider, TenantId
    from redstring.domain.chunk import ChunkId

    from stark_bench.skb.artifacts import SkbEdge, SkbNode

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

    Everything measured here is against **one llama.cpp process, one slot,
    one local GPU**, and that setting is unusually kind to batching and
    unusually unkind to concurrency: there is no per-request network latency
    worth hiding, and no second slot for a second request to occupy, so
    concurrent requests fragment work that could have been one large forward
    pass. On this endpoint `concurrency=1` with a large `embed_batch` was
    the best configuration observed.

    Do not read that as a general rule. Against a hosted API, a multi-slot
    or multi-replica server, or anything across a real network, round-trip
    latency dominates and concurrency is the knob that matters -- batching
    alone would leave most of the throughput unclaimed, and a provider that
    caps request size may not even permit a large batch. The two are
    independent and both are exposed for that reason.

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

    batch: list[Entity] = []
    chunk_batch: list[StoredChunk] = []

    def _plan(node: SkbNode) -> tuple[SkbNode, list, bool]:
        """Chunk a node and decide whether it needs vectors. No I/O.

        Split out from the embedding so that a whole group's texts can go
        into one request -- see the `embed_batch` note in the docstring.
        """
        result = chunker.chunk(node.document)
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

    group_size = concurrency * embed_batch if embeddings is not None else 1
    node_iter = iter(nodes)
    while True:
        group = list(itertools.islice(node_iter, group_size))
        if not group:
            break

        planned = [_plan(node) for node in group]

        if vector_for is not None:
            processed = [
                (node, pieces, [vector_for(node.node_id) for _ in pieces] if wanted else None)
                for node, pieces, wanted in planned
            ]
        else:
            flat = [p.text for _, pieces, wanted in planned if wanted for p in pieces]
            vectors = await _embed_group(flat)
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
        skipped=skipped_count,
    )

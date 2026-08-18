import pytest
from redstring import (
    FakeEmbeddingProvider,
    InMemoryChunkStore,
    InMemoryGraphStore,
    TenantId,
)
from uuid import uuid4

from stark_bench.skb.artifacts import SkbEdge, SkbNode
from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for
from stark_bench.skb.ingest import ingest
from stark_bench.skb.chunkers import WholeDocumentChunker


@pytest.fixture
def stores():
    return InMemoryGraphStore(), InMemoryChunkStore(dimension=8)


@pytest.mark.asyncio
async def test_it_writes_entities_carrying_their_stark_id(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [
        SkbNode("1", "drug", "aspirin", "a salicylate"),
        SkbNode("2", "gene", "PTGS2", "cyclooxygenase-2"),
    ]

    report = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.nodes == 2
    stored = await graph.get_entity(entity_id_for("prime", "1"), tenant)
    assert stored is not None
    assert stored.external_ids[STARK_ID_KEY] == "1"


@pytest.mark.asyncio
async def test_a_self_loop_is_dropped_and_counted(stores):
    """Redstring rejects self-loops. A silent drop would make a recall
    ceiling look like a retrieval failure, so the count is reported."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    edges = [SkbEdge("1", "1", "related_to")]

    report = await ingest(
        nodes,
        edges,
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.self_loops_dropped == 1
    assert report.edges == 0


@pytest.mark.asyncio
async def test_edges_referencing_unknown_nodes_do_not_abort_the_run(stores):
    """A bad edge followed by a good one.

    Stated this way on purpose: with only one bad edge at the end of the
    loop, `break` and `continue` are the same function, and a `break` would
    silently discard every later edge in a real corpus.
    """
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "x"), SkbNode("2", "gene", "PTGS2", "y")]
    edges = [SkbEdge("1", "999", "targets"), SkbEdge("1", "2", "targets")]

    report = await ingest(
        nodes,
        edges,
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.edges == 1


@pytest.mark.asyncio
async def test_ingest_is_idempotent(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    kwargs = dict(
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    first = await ingest(nodes, [], **kwargs)
    second = await ingest(nodes, [], **kwargs)

    assert first.nodes == second.nodes == 1
    assert len(await graph.find_entities(tenant)) == 1

import pytest
from pydantic import BaseModel
from redstring import (
    ChunkRetriever,
    FakeEmbeddingProvider,
    FakeLlmProvider,
    InMemoryChunkStore,
    InMemoryGraphStore,
    RetrievalMode,
    SlidingWindowChunker,
    TenantId,
)
from uuid import uuid4

from stark_bench.ports import Toolset
from stark_bench.skb.artifacts import SkbEdge, SkbNode
from stark_bench.skb.chunkers import WholeDocumentChunker
from stark_bench.skb.ids import STARK_ID_KEY
from stark_bench.skb.ingest import ingest
from stark_bench.tools.redstring_tools import RedstringToolset


@pytest.fixture
async def toolset():
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    nodes = [
        SkbNode("1", "drug", "aspirin", "aspirin inhibits cyclooxygenase"),
        SkbNode("2", "gene", "PTGS2", "PTGS2 encodes cyclooxygenase-2"),
    ]
    await ingest(
        nodes,
        [SkbEdge("1", "2", "targets")],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )
    return RedstringToolset(
        chunks=chunks,
        graph=graph,
        embeddings=FakeEmbeddingProvider(dimension=8),
        tenant_id=tenant,
        dataset="prime",
    )


@pytest.mark.asyncio
async def test_it_satisfies_the_toolset_protocol(toolset):
    assert isinstance(toolset, Toolset)


@pytest.mark.asyncio
async def test_search_returns_stark_node_ids_not_entity_ids(toolset):
    tools = toolset
    results = await tools.search_chunks("cyclooxygenase", k=5)
    assert results
    assert all(r.node_id in {"1", "2"} for r in results)


@pytest.mark.asyncio
async def test_every_call_is_recorded(toolset):
    tools = toolset
    await tools.search_chunks("cyclooxygenase", k=5)
    await tools.neighbors("1")
    assert [c.tool for c in tools.calls] == ["search_chunks", "neighbors"]
    assert all(c.duration_s >= 0 for c in tools.calls)


@pytest.mark.asyncio
async def test_neighbors_returns_stark_ids(toolset):
    tools = toolset
    assert await tools.neighbors("1") == ["2"]


@pytest.mark.asyncio
async def test_the_toolset_exposes_no_writer(toolset):
    """Reader-only is the point: an agent that cannot write cannot poison
    the KB mid-run."""
    tools = toolset
    for forbidden in ("upsert_entities", "upsert_many", "delete_by_tenant"):
        assert not hasattr(tools, forbidden)


class _Answer(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_extract_delegates_to_the_llm_provider_and_is_recorded():
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    llm = FakeLlmProvider(script=[{"text": "cyclooxygenase"}])
    tools = RedstringToolset(
        chunks=chunks,
        graph=graph,
        embeddings=FakeEmbeddingProvider(dimension=8),
        tenant_id=tenant,
        dataset="prime",
        llm=llm,
    )

    result = await tools.extract("what does aspirin inhibit?", _Answer)

    assert result == _Answer(text="cyclooxygenase")
    assert [c.tool for c in tools.calls] == ["extract"]
    # A missing measurement and a measurement of nothing are different facts:
    # `LlmProvider` reports no usage, so this must be `None`, never `0`.
    assert tools.calls[0].tokens is None


@pytest.mark.asyncio
async def test_extract_without_an_llm_raises(toolset):
    tools = toolset
    with pytest.raises(RuntimeError):
        await tools.extract("what does aspirin inhibit?", _Answer)


@pytest.mark.asyncio
async def test_search_folds_multiple_chunks_of_one_node_using_max():
    """The shared `toolset` fixture chunks every document whole, so every
    node has exactly one chunk -- `aggregate()`'s own docstring calls that
    the degenerate case, where `max`, `mean` and `sum` all agree. A folding
    bug is invisible under that fixture. This test forces a real multi-chunk
    node and pins its scores so the three strategies genuinely disagree.
    """
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    embeddings = FakeEmbeddingProvider(dimension=8)
    document = (
        "Aspirin inhibits cyclooxygenase enzymes in the arachidonic acid "
        "pathway, blocking prostaglandin synthesis in inflamed tissue. "
        "Separately, aspirin also produces antiplatelet effects through "
        "irreversible acetylation of platelet cyclooxygenase-1, reducing "
        "thromboxane production and clotting risk in the bloodstream."
    )
    chunker = SlidingWindowChunker(
        default_chunk_size=120, default_overlap=0, min_chunk_size=20
    )
    report = await ingest(
        [SkbNode("1", "drug", "aspirin", document)],
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=chunker,
        embeddings=embeddings,
    )
    # Guard the fixture itself: if this ever regresses to one chunk, the
    # rest of the test would pass for the wrong reason (the degenerate case).
    assert (
        report.chunks > 1
    ), "fixture document must actually split into multiple chunks"

    query_text = "cyclooxygenase inhibition"

    # Independent oracle: read the raw per-chunk scores straight from
    # retrieval, then reduce them ourselves. The toolset's own `aggregate()`
    # is not used to compute the expectation.
    raw_retriever = ChunkRetriever(embeddings=embeddings, chunks=chunks)
    raw = await raw_retriever.retrieve_chunks(
        query_text, tenant, k=20, mode=RetrievalMode.SEMANTIC
    )
    node_scores = [
        m.score for m in raw.matches if m.chunk.metadata.get(STARK_ID_KEY) == "1"
    ]
    assert len(node_scores) > 1
    expected_max = max(node_scores)
    expected_mean = sum(node_scores) / len(node_scores)
    # If the two happened to coincide, max vs. mean couldn't be told apart
    # below -- this is what makes the test able to fail at all.
    assert expected_max != pytest.approx(expected_mean)

    tools = RedstringToolset(
        chunks=chunks,
        graph=graph,
        embeddings=embeddings,
        tenant_id=tenant,
        dataset="prime",
        aggregation="max",
    )
    results = await tools.search_chunks(query_text, k=5, mode="semantic")
    matches = [r for r in results if r.node_id == "1"]

    # Folding: one node, one entry -- not one per chunk.
    assert len(matches) == 1
    # And it is the max over that node's chunks, not the mean or sum.
    assert matches[0].score == pytest.approx(expected_max)
    assert matches[0].score != pytest.approx(expected_mean)

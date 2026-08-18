import pytest
from pydantic import BaseModel
from redstring import (
    FakeEmbeddingProvider,
    FakeLlmProvider,
    InMemoryChunkStore,
    InMemoryGraphStore,
    TenantId,
)
from uuid import uuid4

from stark_bench.ports import Toolset
from stark_bench.skb.artifacts import SkbEdge, SkbNode
from stark_bench.skb.chunkers import WholeDocumentChunker
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

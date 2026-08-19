"""The whole pipeline on twelve nodes: ingest, retrieve, aggregate, score."""

from pathlib import Path
from uuid import uuid4

import pytest
from redstring import (
    FakeEmbeddingProvider,
    InMemoryChunkStore,
    InMemoryGraphStore,
    TenantId,
)

from stark_bench.agents.hybrid import HybridAgent
from stark_bench.application.run_queries import run
from stark_bench.harness.scoring import score_predictions
from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries
from stark_bench.skb.chunkers import WholeDocumentChunker
from stark_bench.skb.ingest import ingest
from stark_bench.tools.redstring_tools import RedstringToolset

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_skb"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_whole_pipeline_produces_a_number():
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    embeddings = FakeEmbeddingProvider(dimension=8)

    report = await ingest(
        read_nodes(FIXTURE / "nodes.jsonl"),
        read_edges(FIXTURE / "edges.jsonl"),
        dataset="fixture",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=embeddings,
    )
    assert report.nodes == 12
    assert report.self_loops_dropped == 1

    pairs = list(read_queries(FIXTURE / "queries.jsonl"))
    queries = [q for q, _ in pairs]
    answers = {q.query_id: a for q, a in pairs}

    tools = RedstringToolset(
        chunks=chunks,
        graph=graph,
        embeddings=embeddings,
        tenant_id=tenant,
        dataset="fixture",
    )
    predictions = await run(HybridAgent(k=20), queries, tools)
    metrics = score_predictions(predictions, answers, candidate_ids=list(range(1, 13)))

    assert set(metrics) >= {"mrr", "hit@1", "hit@5", "recall@20"}
    assert 0.0 <= metrics["mrr"] <= 1.0

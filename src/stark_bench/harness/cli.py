"""Bring one config to life: ingest, or run and score.

Two subcommands, `--ingest` and `--run`, over one `RunConfig`. Ingest and
retrieval are separate invocations deliberately -- a run this large is worth
being able to retry independently of the other.

`--ingest-edges` defaults OFF. `dense` and `hybrid` retrieve through
`ChunkRetriever`, which holds a `ChunkStore` and never touches the graph;
edges are only needed by traversal agents, which this CLI does not yet run.
Whichever way it was run is recorded in the report, so a later reader is not
left guessing whether a number came from a graph-less store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid5

from redstring import TenantId
from redstring.chunks.adapters.postgres import PostgresChunkStore
from redstring.graph.adapters.neo4j import Neo4jGraphStore

from stark_bench.agents.dense import DenseAgent
from stark_bench.agents.hybrid import HybridAgent
from stark_bench.harness.config import RunConfig, load_config
from stark_bench.harness.providers import (
    PrecomputedEmbeddingProvider,
    node_vector_lookup,
)
from stark_bench.harness.report import summarise_cost, write_report
from stark_bench.harness.runner import run
from stark_bench.harness.scoring import score_predictions
from stark_bench.skb.artifacts import (
    read_doc_embeddings,
    read_edges,
    read_nodes,
    read_queries,
    read_query_embeddings,
)
from stark_bench.skb.chunkers import WholeDocumentChunker
from stark_bench.skb.ids import NAMESPACE_STARK
from stark_bench.skb.ingest import ingest
from stark_bench.tools.redstring_tools import RedstringToolset

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "results"

POSTGRES_DSN = "postgresql://stark:stark@localhost:55432/stark"
NEO4J_URI = "bolt://localhost:57687"
NEO4J_AUTH = ("neo4j", "starkbench")

CHUNKERS = {"whole-document": WholeDocumentChunker}
AGENTS = {"dense": DenseAgent, "hybrid": HybridAgent}


def _tenant_for(config: RunConfig) -> TenantId:
    """One deterministic tenant per config name -- stable across re-runs."""
    return TenantId(uuid5(NAMESPACE_STARK, f"tenant:{config.name}"))


def _data_dir(config: RunConfig) -> Path:
    return DATA_ROOT / config.dataset


async def _do_ingest(config: RunConfig, *, ingest_edges: bool) -> dict[str, object]:
    if config.embeddings != "precomputed-ada002":
        raise NotImplementedError(
            f"ingest only wires precomputed-ada002 today, got {config.embeddings!r}"
        )
    if config.chunker not in CHUNKERS:
        raise NotImplementedError(f"unknown chunker {config.chunker!r}")

    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    started = time.monotonic()
    logger.info("loading precomputed doc embeddings from %s", data_dir / "doc_emb.npz")
    doc_embeddings = read_doc_embeddings(data_dir / "doc_emb.npz")
    vector_for = node_vector_lookup(doc_embeddings)

    chunks = await PostgresChunkStore.connect(POSTGRES_DSN, dimension=config.dimension)
    graph = Neo4jGraphStore.connect(NEO4J_URI, auth=NEO4J_AUTH)
    await chunks.ensure_schema()
    await graph.ensure_schema()
    try:
        nodes = read_nodes(data_dir / "nodes.jsonl")
        edges = read_edges(data_dir / "edges.jsonl") if ingest_edges else iter(())

        report = await ingest(
            nodes,
            edges,
            dataset=config.dataset,
            tenant_id=tenant_id,
            graph=graph,
            chunks=chunks,
            chunker=CHUNKERS[config.chunker](),
            vector_for=vector_for,
        )
    finally:
        await chunks.close()
        await graph.close()

    elapsed = time.monotonic() - started
    return {
        "nodes": report.nodes,
        "chunks": report.chunks,
        "edges": report.edges,
        "self_loops_dropped": report.self_loops_dropped,
        "edges_ingested": ingest_edges,
        "wall_time_s": elapsed,
    }


async def _do_run(config: RunConfig) -> None:
    if config.embeddings != "precomputed-ada002":
        raise NotImplementedError(
            f"run only wires precomputed-ada002 today, got {config.embeddings!r}"
        )
    if config.agent not in AGENTS:
        raise NotImplementedError(f"unknown agent {config.agent!r}")

    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    pairs = list(read_queries(data_dir / f"queries.{config.split}.jsonl"))
    queries = [q for q, _ in pairs]
    answers = {q.query_id: a for q, a in pairs}

    logger.info(
        "loading precomputed query embeddings from %s", data_dir / "query_emb.npz"
    )
    query_vectors_by_id = read_query_embeddings(data_dir / "query_emb.npz")
    vectors_by_text = {
        q.text: query_vectors_by_id[str(q.query_id)]
        for q in queries
        if str(q.query_id) in query_vectors_by_id
    }
    embeddings = PrecomputedEmbeddingProvider(
        vectors_by_text, dimension=config.dimension
    )

    chunks = await PostgresChunkStore.connect(POSTGRES_DSN, dimension=config.dimension)
    graph = Neo4jGraphStore.connect(NEO4J_URI, auth=NEO4J_AUTH)
    await chunks.ensure_schema()
    await graph.ensure_schema()
    try:
        tools = RedstringToolset(
            chunks=chunks,
            graph=graph,
            embeddings=embeddings,
            tenant_id=tenant_id,
            dataset=config.dataset,
            aggregation=config.aggregation,
        )
        agent = AGENTS[config.agent](k=config.k)

        predictions = await run(agent, queries, tools, k=config.k)

        candidates_path = data_dir / "candidates.json"
        candidate_ids = [int(c) for c in json.loads(candidates_path.read_text())]
        metrics = score_predictions(predictions, answers, candidate_ids=candidate_ids)
        cost = summarise_cost(tools.calls, queries=len(queries))
    finally:
        await chunks.close()
        await graph.close()

    write_report(
        RESULTS_ROOT / f"{config.name}.json",
        config=config,
        metrics=metrics,
        cost=cost,
        ingest={},
        queries=len(queries),
    )
    print(metrics)  # noqa: T201


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--ingest-edges",
        action="store_true",
        default=False,
        help="Also load edges into the graph store. Off by default: dense "
        "and hybrid retrieve through ChunkRetriever, which never touches "
        "the graph, so this costs ~16k transactions for no benefit to them.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.ingest:
        report = asyncio.run(_do_ingest(config, ingest_edges=args.ingest_edges))
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        (RESULTS_ROOT / f"{config.name}.ingest.json").write_text(
            json.dumps(report, indent=2)
        )
        print(report)  # noqa: T201

    if args.run:
        asyncio.run(_do_run(config))


if __name__ == "__main__":
    main()

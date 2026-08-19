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
import itertools
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid5

import asyncpg
from redstring import TenantId
from redstring.chunks.adapters.postgres import PostgresChunkStore
from redstring.extraction.chunkers.boundary_preference_chunker import (
    BoundaryPreferenceChunker,
)
from redstring.graph.adapters.neo4j import Neo4jGraphStore
from redstring.llm.adapters.langchain_embedding import LangChainEmbeddingProvider

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

if TYPE_CHECKING:
    from redstring import EmbeddingProvider

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "results"

POSTGRES_DSN = "postgresql://stark:stark@localhost:55432/stark"
NEO4J_URI = "bolt://localhost:57687"
NEO4J_AUTH = ("neo4j", "starkbench")

#: The endpoint backing `nomic-embed-text`. Shared infrastructure -- see
#: `--embed-concurrency` on the CLI.
NOMIC_BASE_URL = "http://192.168.1.14:8080/v1/"

CHUNKERS = {
    "whole-document": WholeDocumentChunker,
    "boundary-preference": BoundaryPreferenceChunker,
}
AGENTS = {"dense": DenseAgent, "hybrid": HybridAgent}
LIVE_EMBEDDINGS = {"nomic-embed-text"}


def _tenant_for(config: RunConfig) -> TenantId:
    """One deterministic tenant per config name -- stable across re-runs."""
    return TenantId(uuid5(NAMESPACE_STARK, f"tenant:{config.name}"))


def _data_dir(config: RunConfig) -> Path:
    return DATA_ROOT / config.dataset


def _table_for(config: RunConfig) -> str:
    """One chunk-store table per embedding model, never shared.

    ADR 0002 records that a different embedding model means a new store.
    Dimension differs here anyway (1536 vs 768), but width alone would not
    protect against a second same-width model landing in the same table
    later -- the dimension guard cannot see a model-identity mismatch, so
    the table name is keyed on the embeddings identifier by construction.
    """
    slug = config.embeddings.replace("-", "_")
    return f"kg_chunks_{slug}"


def _live_embeddings_for(config: RunConfig) -> EmbeddingProvider:
    if config.embeddings == "nomic-embed-text":
        return LangChainEmbeddingProvider.openai_compatible(
            base_url=NOMIC_BASE_URL,
            model="nomic-embed-text",
            dimension=config.dimension,
        )
    raise NotImplementedError(f"no live embedding provider for {config.embeddings!r}")


async def _load_existing_chunk_ids(table: str, tenant_id: TenantId) -> set[str]:
    """This tenant's chunk ids, in one query -- the resume skip's input.

    A per-node lookup would be ~129k round trips against `PostgresChunkStore`,
    which has no bulk-id method on the `ChunkStore` port. Querying the table
    directly, once, up front, is what keeps the skip check in-memory
    thereafter. `table` is the same value `_table_for` derives -- a slug of
    `config.embeddings`, never caller input -- so it is safe to interpolate.
    """
    connection = await asyncpg.connect(POSTGRES_DSN)
    try:
        rows = await connection.fetch(
            f"SELECT id FROM {table} WHERE tenant_id = $1",  # nosec B608
            tenant_id,
        )
    finally:
        await connection.close()
    return {str(row["id"]) for row in rows}


async def _do_ingest(
    config: RunConfig,
    *,
    ingest_edges: bool,
    embed_concurrency: int = 4,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, object]:
    if config.embeddings != "precomputed-ada002" and config.embeddings not in (
        LIVE_EMBEDDINGS
    ):
        raise NotImplementedError(f"no ingest wiring for {config.embeddings!r}")
    if config.chunker not in CHUNKERS:
        raise NotImplementedError(f"unknown chunker {config.chunker!r}")

    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    started = time.monotonic()

    vector_for = None
    embeddings = None
    if config.embeddings == "precomputed-ada002":
        logger.info(
            "loading precomputed doc embeddings from %s", data_dir / "doc_emb.npz"
        )
        doc_embeddings = read_doc_embeddings(data_dir / "doc_emb.npz")
        vector_for = node_vector_lookup(doc_embeddings)
    else:
        logger.info(
            "embedding live against %r (concurrency=%d)",
            config.embeddings,
            embed_concurrency,
        )
        embeddings = _live_embeddings_for(config)

    table = _table_for(config)
    chunks = await PostgresChunkStore.connect(
        POSTGRES_DSN, table=table, dimension=config.dimension
    )
    graph = Neo4jGraphStore.connect(NEO4J_URI, auth=NEO4J_AUTH)
    await chunks.ensure_schema()
    await graph.ensure_schema()
    try:
        existing_chunk_ids: set[str] = set()
        existing_ids_load_s = 0.0
        if resume:
            load_started = time.monotonic()
            existing_chunk_ids = await _load_existing_chunk_ids(table, tenant_id)
            existing_ids_load_s = time.monotonic() - load_started
            logger.info(
                "loaded %d existing chunk ids for tenant in %.2fs (resume)",
                len(existing_chunk_ids),
                existing_ids_load_s,
            )

        nodes = read_nodes(data_dir / "nodes.jsonl")
        if limit is not None:
            nodes = itertools.islice(nodes, limit)
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
            embeddings=embeddings,
            concurrency=embed_concurrency if embeddings is not None else 1,
            existing_chunk_ids=existing_chunk_ids,
            resume=resume,
        )
    finally:
        await chunks.close()
        await graph.close()

    elapsed = time.monotonic() - started
    return {
        "nodes": report.nodes,
        "chunks": report.chunks,
        "skipped": report.skipped,
        "edges": report.edges,
        "self_loops_dropped": report.self_loops_dropped,
        "edges_ingested": ingest_edges,
        "resume": resume,
        "existing_ids_load_s": existing_ids_load_s,
        "wall_time_s": elapsed,
    }


async def _do_run(config: RunConfig) -> None:
    if config.embeddings != "precomputed-ada002" and config.embeddings not in (
        LIVE_EMBEDDINGS
    ):
        raise NotImplementedError(f"no run wiring for {config.embeddings!r}")
    if config.agent not in AGENTS:
        raise NotImplementedError(f"unknown agent {config.agent!r}")

    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    pairs = list(read_queries(data_dir / f"queries.{config.split}.jsonl"))
    queries = [q for q, _ in pairs]
    answers = {q.query_id: a for q, a in pairs}

    if config.embeddings == "precomputed-ada002":
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
    else:
        logger.info("embedding queries live against %r", config.embeddings)
        embeddings = _live_embeddings_for(config)

    chunks = await PostgresChunkStore.connect(
        POSTGRES_DSN, table=_table_for(config), dimension=config.dimension
    )
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
    parser.add_argument(
        "--embed-concurrency",
        type=int,
        default=4,
        help="Nodes chunked-and-embedded at once, for a config with live "
        "embeddings. The inference endpoint is shared -- do not raise this "
        "without confirming spare capacity with whoever else uses it. "
        "Ignored for precomputed-embeddings configs, which do no live I/O.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N nodes. For calibrating throughput "
        "against a live endpoint before committing to a full run -- never "
        "use it for a config whose numbers get reported.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ingest is resumable by default: a node whose chunk already "
        "exists in the store (same source and text -- an unchanged node) is "
        "skipped rather than re-embedded, so an interrupted ingest restarts "
        "cheaply instead of re-embedding from zero. Pass --no-resume to "
        "force a full re-ingest of every node regardless of what the store "
        "already holds -- the escape hatch, because skipping work is "
        "exactly the kind of optimisation that can hide a bug.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.ingest:
        report = asyncio.run(
            _do_ingest(
                config,
                ingest_edges=args.ingest_edges,
                embed_concurrency=args.embed_concurrency,
                limit=args.limit,
                resume=not args.no_resume,
            )
        )
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        (RESULTS_ROOT / f"{config.name}.ingest.json").write_text(
            json.dumps(report, indent=2)
        )
        print(report)  # noqa: T201

    if args.run:
        asyncio.run(_do_run(config))


if __name__ == "__main__":
    main()

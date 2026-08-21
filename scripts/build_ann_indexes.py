"""Build one partial HNSW index per tenant on the chunk table.

**These indexes do not work, and this script is kept as the record of why.**
Built on 2026-08-21 for three tenants, 5.7GB, ~9 minutes; all three ended
with `idx_scan = 0`. redstring's `_semantic_candidates_sql` orders by
`1 - (embedding <=> $2) / 2 DESC, id ASC`, and an HNSW index can only serve
`ORDER BY embedding <=> $2` ascending -- a rescaled similarity sorted
descending with a tie-break on `id` is not a shape it can answer, so every
query takes a parallel sequential scan regardless. Worse than useless:
`qwen-rel-sliding1k` dense went 165.8s -> 519.1s, because 5.7GB of unread
index evicts the table from the page cache. See BACKLOG
B-ANN-INDEX-UNREACHABLE-1 before running this.

**Verify with `idx_scan`, not `EXPLAIN`.** `EXPLAIN` on the simplified
`ORDER BY embedding <=> $1 LIMIT 20` shows an index scan and is what made
this look correct for an hour. That is a query the codebase never issues.
`pg_stat_user_indexes.idx_scan` after a real run is the check with teeth.

**Why partial, and why this is not simply "redstring forgot an index".**
redstring's pgvector adapter has no ANN index deliberately, and says so at
length: an ANN index over a multi-tenant table either lets the planner take
the `k` globally nearest rows and drop other tenants' afterwards -- returning
a handful of rows, or none, for a tenant with genuine neighbours -- or gets
skipped entirely in favour of the tenant filter, costing write throughput to
be never read. Both outcomes look correct. See redstring BACKLOG B10k, which
names three ways out; this script is its option (3), a partial index per large
tenant, chosen because it does not scale past a few tenants and this benchmark
has exactly six.

`WHERE tenant_id = '...'` is what makes it safe: the planner may only use the
index for queries carrying that predicate, so post-filtering cannot lose rows
and no tenant can see another's vectors.

**This changes retrieval from exact to approximate**, which is the whole
reason it needs validating rather than just building. An HNSW scan can miss a
true neighbour, and a miss lands in MRR looking exactly like a weaker
architecture. `--validate` re-scores an arm whose exact number is already
known and compares; do not trust an indexed number until that has passed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from uuid import uuid5

import asyncpg

from stark_bench.domain.stark_ids import NAMESPACE_STARK

TABLE = "kg_chunks_qwen3_embedding_0_6b_1e22db42"

#: `<=>` is cosine distance, so the opclass must be the cosine one -- an
#: l2 opclass index simply never gets used by a cosine query, silently.
OPCLASS = "vector_cosine_ops"

DSN = os.environ.get(
    "STARK_BENCH_PG_DSN", "postgresql://stark:stark@127.0.0.1:55432/stark"
)


def tenant_for(name: str) -> str:
    return str(uuid5(NAMESPACE_STARK, f"tenant:{name}"))


async def build(
    names: list[str],
    *,
    m: int,
    ef_construction: int,
    maintenance_work_mem: str,
) -> None:
    conn = await asyncpg.connect(DSN)
    try:
        # HNSW builds are memory-bound, and the cliff is sharp rather than
        # gradual: once the graph does not fit, pgvector finishes the build
        # on disk and the rate collapses. 549,697 rows x 1024 dims x 4 bytes
        # is 2.25GB of vectors before graph links, so 2GB was under the
        # line. Measured on that corpus: at 2GB the "loading tuples" phase
        # advanced 61% -> 63.6% in three minutes (~0.9%/min, ~40 minutes
        # remaining); at 12GB, 39.9% -> 49.3% in four (~2.4%/min). Better
        # by 2.6x, not the order of magnitude the cliff metaphor suggests
        # -- 12GB clears the vectors but the graph links still spill.
        await conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
        await conn.execute("SET max_parallel_maintenance_workers = 4")
        for name in names:
            tenant = tenant_for(name)
            index = f"{TABLE}_hnsw_{name.replace('-', '_')}"
            rows = await conn.fetchval(
                f"select count(*) from {TABLE} where tenant_id = $1", tenant
            )
            if not rows:
                print(f"  skip   {name}: tenant holds no chunks")
                continue
            print(f"  build  {name}: {rows:,} rows -> {index}", flush=True)
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {index} ON {TABLE} "
                f"USING hnsw (embedding {OPCLASS}) "
                f"WITH (m = {m}, ef_construction = {ef_construction}) "
                f"WHERE tenant_id = '{tenant}'"
            )
            size = await conn.fetchval(
                "select pg_size_pretty(pg_relation_size($1::regclass))", index
            )
            print(f"  done   {name}: index is {size}", flush=True)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="config names to index")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--maintenance-work-mem", default="12GB")
    args = parser.parse_args()
    asyncio.run(
        build(
            args.names,
            m=args.m,
            ef_construction=args.ef_construction,
            maintenance_work_mem=args.maintenance_work_mem,
        )
    )


if __name__ == "__main__":
    main()

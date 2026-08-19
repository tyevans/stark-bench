#!/usr/bin/env python
"""How far along is an arm's ingest, scoped correctly.

Exists because the obvious query is wrong. Three arms share one chunk
table -- they share a model, dimension and prefixes, so ADR 0002 and 0043
are satisfied -- and are separated only by `tenant_id`. So

    select count(*) from kg_chunks_nemotron_3_embed_1b_d38d8f8b;

sums three arms. On 2026-08-19 that made an arm at 133,919 chunks read as
141,673 against an expected ~136,700: apparently finished and overshooting
when it was neither, which sent an hour of reasoning down the wrong path.

The fix for a trap like that is not a note telling people to remember the
`where` clause. It is a command that cannot omit it. Both the table name
and the tenant id are derived here the same way the CLI derives them, so
neither can be typed wrong.

    uv run python scripts/progress.py native-wholedoc
    uv run python scripts/progress.py            # every configured arm
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stark_bench.adapters.config_file import load_config  # noqa: E402
from stark_bench.harness.cli import POSTGRES_DSN, _table_for, _tenant_for  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
EXPECTED_NODES = 129375


def query_for(name: str) -> tuple[str, str, str]:
    """Return (table, tenant, SQL) for one arm, without touching a database.

    Split out from the query so the scoping can be tested: the defect this
    script exists to prevent is a missing `where tenant_id`, and that is
    visible in the SQL without a live Postgres.
    """
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    table = _table_for(config)
    tenant = str(_tenant_for(config))
    # The identifier is computed by the CLI, never user text, so there is
    # nothing here to interpolate unsafely. The tenant is a bound parameter.
    sql = f"select count(*) from {table} where tenant_id = $1"  # noqa: S608
    return table, tenant, sql


async def counts(name: str) -> tuple[str, int, int]:
    """Return (table, chunks for this arm's tenant, expected nodes)."""
    table, tenant, sql = query_for(name)
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        chunks = await conn.fetchval(sql, tenant)
    finally:
        await conn.close()
    return table, chunks, EXPECTED_NODES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="config names; default all")
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    for name in names:
        try:
            table, chunks, nodes = asyncio.run(counts(name))
        except Exception as error:  # noqa: BLE001 -- a missing table is an answer
            print(f"{name:20s} -- {type(error).__name__}: {error}")
            continue
        # Granularity, not progress: an arm is not "106% done" at 1.06
        # chunks per node. Chunkers here emit at least one chunk per node,
        # so this is >= 1.0 for a finished arm and below it for a partial.
        ratio = chunks / nodes if nodes else 0.0
        state = "complete" if ratio >= 1.0 else f"{ratio:.1%} of one-chunk-per-node"
        print(f"{name:20s} {chunks:>8,} chunks  {ratio:.3f}/node  {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

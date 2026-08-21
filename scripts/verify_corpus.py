"""Compare what each ingest report CLAIMS against what the store HOLDS.

The second half of B-RESUME-COMPLETE-1. `resume_is_safe` establishes that
the chunker has not changed and that the run finished; neither says the rows
are actually there.

The report records `chunks` (written) and `skipped` (already present), so
`chunks + skipped` is what the arm believes its corpus contains. Comparing
that to `count(*)` for the tenant is the check with teeth, and it caught a
189-chunk shortfall on `qwen-rel-sliding1k` the first time it ran.

**Scoped to the tenant, always.** Configs sharing an embedding model share
one table and are separated only by `tenant_id`; a bare `count(*)` sums
several arms. CLAUDE.md records that making one arm at 133,919 read as
141,673 against a target of ~136,700 -- finished and overshooting when it
was neither.

Exit 0 if every arm with a report agrees, 1 otherwise. Reports for corpora
that no longer exist are listed as MISSING rather than failing: the stores
have been dropped deliberately before, and a result file outliving its
corpus is expected rather than wrong.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import asyncpg  # noqa: E402

from stark_bench.adapters.config_file import load_config  # noqa: E402
from stark_bench.composition.cli import (  # noqa: E402
    POSTGRES_DSN,
    _table_for,
    _tenant_for,
    ingest_report_path,
)

#: Rows may legitimately fall short of the reported count when two chunks
#: share an id -- ids are content-addressed, so an upsert merges them. A
#: handful is a curiosity; a large gap is a lost write. Expressed as a
#: fraction because the arms differ by two orders of magnitude in size.
TOLERANCE = 0.001


async def main() -> int:
    pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=2)
    failures = 0
    try:
        async with pool.acquire() as conn:
            tables = {
                r["table_name"]
                for r in await conn.fetch(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_name LIKE 'kg_chunks%'"
                )
            }
            for path in sorted(Path("config").glob("*.yaml")):
                config = load_config(path)
                report_path = ingest_report_path(config)
                if not report_path.exists():
                    continue
                report = json.loads(report_path.read_text(encoding="utf-8"))
                claimed = report.get("chunks", 0) + report.get("skipped", 0)
                if not claimed:
                    continue
                table = _table_for(config)
                if table not in tables:
                    print(f"  MISSING  {config.name:<22} table {table} does not exist")
                    continue
                actual = await conn.fetchval(
                    f"SELECT count(*) FROM {table} WHERE tenant_id = $1",  # noqa: S608
                    str(_tenant_for(config)),
                )
                gap = claimed - actual
                share = abs(gap) / claimed
                ok = share <= TOLERANCE
                failures += 0 if ok else 1
                flag = "ok      " if ok else "MISMATCH"
                print(
                    f"  {flag} {config.name:<22} claimed {claimed:>9,} "
                    f"actual {actual:>9,} gap {gap:>+7,} ({share:.3%})"
                )
    finally:
        await pool.close()
    if failures:
        print(
            f"\n{failures} arm(s) disagree with their report by more than {TOLERANCE:.1%}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

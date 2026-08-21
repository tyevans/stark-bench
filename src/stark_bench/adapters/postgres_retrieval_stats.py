"""Whether an ANN index served a run, read back from Postgres.

Here rather than in the composition root because `asyncpg` is confined to
this directory -- `tests/test_dependencies_stay_confined.py` is the gate,
and it caught this on the first run of the code below.

Reaching past redstring's `ChunkStore` again, knowingly, for the same
reason `postgres_chunk_index.py` does: the port has no notion of an index,
and it should not grow one to satisfy a benchmark's reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from stark_bench.domain.stark_ids import TenantId


async def retrieval_stats(dsn: str, table: str, tenant: TenantId) -> dict[str, object]:
    """Whether an ANN index served this run, and how wide its search was.

    **The number in a report is not interpretable without this.** An HNSW
    index makes retrieval approximate: measured on `qwen-rel-whole`, the
    default `hnsw.ef_search = 40` costs 0.0117 of recall@20 against an exact
    scan, and 800 matches it to every digit. Nothing else in the report can
    see the difference -- `config_verbatim` is the config FILE, and the index
    lives in Postgres where no config mentions it -- so two runs of the same
    arm on different sides of this produce files that differ only in the
    metric, which reads as an architecture result and is not one.

    `idx_scan` is recorded rather than inferred, and it is the field to read
    first. An index can exist, be the right opclass, and still go unused --
    that is exactly what happened here before redstring PR #71: three
    indexes, 5.7GB, `idx_scan = 0` on all of them, and retrieval 3.1x slower
    than with no index at all. `EXPLAIN` on a simplified query shape says
    otherwise and is how the mistake survived an hour.

    Raises rather than returning `{}` if the query fails. `--run` already
    refuses up front against a missing corpus, so there is no legitimate
    path here with no table, and an empty block would be indistinguishable
    from a broken query -- see the comment on the `try`.
    """
    conn = await asyncpg.connect(dsn)
    try:
        # Not wrapped in a broad `except`. The first draft caught
        # `UndefinedTableError` around the whole block and returned `{}`,
        # which is also what a *typo in this query* produces -- and did:
        # `s.idx_scan` against an alias named `i` raised exactly that, and
        # the run reported a clean empty provenance block instead of
        # failing. A check that cannot fail loudly is not a check.
        rows = await conn.fetch(
            "SELECT i.indexrelname, i.idx_scan, pg_get_indexdef(i.indexrelid) AS indexdef "
            "  FROM pg_stat_user_indexes i "
            "  JOIN pg_class c ON c.oid = i.indexrelid "
            "  JOIN pg_am am ON am.oid = c.relam "
            " WHERE i.relname = $1 AND am.amname IN ('hnsw', 'ivfflat')",
            table,
        )
        ef_search = await conn.fetchval("SHOW hnsw.ef_search")
    finally:
        await conn.close()

    # Scoped to this arm: every config sharing an embedding model shares one
    # table, so an index belonging to another tenant is not this run's.
    mine = [row for row in rows if str(tenant) in row["indexdef"]]
    return {
        "ann_index": mine[0]["indexrelname"] if mine else None,
        "ann_index_definition": mine[0]["indexdef"] if mine else None,
        # Cumulative since the last stats reset, not this run's count, so
        # read it as "has this index ever been used" rather than as a total.
        "ann_index_scans_cumulative": mine[0]["idx_scan"] if mine else None,
        "hnsw_ef_search": ef_search,
        "retrieval_is_exact": not mine,
    }

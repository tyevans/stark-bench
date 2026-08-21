"""`ChunkIdIndex` over the chunk table redstring's Postgres store writes.

Goes around `ChunkStore` deliberately: the port has no bulk-id method, and
the alternative is ~129,000 round trips. Reaching past a port is a thing to
do knowingly and in one place, which is what this module is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from uuid import UUID


class PostgresChunkIdIndex:
    """One query for a tenant's chunk ids."""

    def __init__(self, dsn: str, table: str) -> None:
        """Hold the DSN and table; connect per call.

        `table` must be a name this application derived -- in practice
        `CorpusIdentity.table_name()`, which is a slug of a model id and can
        contain only `[a-z0-9_]`. It is interpolated into SQL because an
        identifier cannot be a bind parameter, so that provenance is the
        thing keeping this safe and it is asserted rather than assumed.
        """
        if not table.replace("_", "").isalnum() or not table.islower():
            raise ValueError(
                f"table must be a bare lowercase identifier, not {table!r} -- "
                "it is interpolated into SQL and cannot be caller input"
            )
        self._dsn = dsn
        self._table = table

    async def count_for_tenant(self, tenant_id: UUID) -> int:
        """How many chunks this tenant holds.

        `ids_for_tenant` answers the same question but materialises every
        id -- 549,886 strings for one arm -- and a preflight only needs to
        know whether the number is zero.
        """
        connection = await asyncpg.connect(self._dsn)
        try:
            return int(
                await connection.fetchval(
                    f"SELECT count(*) FROM {self._table} WHERE tenant_id = $1",  # nosec B608
                    tenant_id,
                )
            )
        finally:
            await connection.close()

    async def ids_for_tenant(self, tenant_id: UUID) -> set[str]:
        connection = await asyncpg.connect(self._dsn)
        try:
            rows = await connection.fetch(
                f"SELECT id FROM {self._table} WHERE tenant_id = $1",  # nosec B608
                tenant_id,
            )
        finally:
            await connection.close()
        return {str(row["id"]) for row in rows}


class InMemoryChunkIdIndex:
    """The same port, for tests and for a first ingest.

    Not a stub: `--no-resume` genuinely has no prior ids, and a use case
    that must be handed a database to run a clean ingest would be one that
    cannot be tested without one.
    """

    def __init__(self, ids: dict[UUID, set[str]] | None = None) -> None:
        self._ids = ids or {}

    async def count_for_tenant(self, tenant_id: UUID) -> int:
        return len(self._ids.get(tenant_id, set()))

    async def ids_for_tenant(self, tenant_id: UUID) -> set[str]:
        return set(self._ids.get(tenant_id, set()))

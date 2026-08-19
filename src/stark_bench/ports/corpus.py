"""Capabilities the ingest needs that redstring's own ports do not offer.

## Why `ChunkIdIndex` exists

Resuming an ingest needs the set of chunk ids a tenant already holds, in
**one** query. redstring's `ChunkStore` has no bulk-id method, and asking
per node would be ~129,000 round trips -- so the ingest was reaching past
the port and issuing raw SQL from the CLI.

That is a missing capability, not a shortcut, and naming it as a port is
what lets the use case stay ignorant of Postgres while an adapter answers
the question however it can. It also makes the in-memory implementation
trivial, which is what makes the use case testable without a database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID


@runtime_checkable
class ChunkIdIndex(Protocol):
    """Which chunks a tenant already has, cheaply and in bulk."""

    async def ids_for_tenant(self, tenant_id: UUID) -> set[str]:
        """Every chunk id stored for this tenant.

        One call, not one per node. An implementation that loops is
        satisfying the signature and defeating the point.
        """
        ...

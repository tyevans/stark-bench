"""The table name is interpolated into SQL, so its shape is a guard.

An identifier cannot be a bind parameter, so the only thing keeping this
safe is that the name was derived by this application from a model id.
That provenance is asserted at construction rather than assumed, because
"it is always called with a safe value" is exactly the property that stops
being true later.
"""

from __future__ import annotations

import pytest

from stark_bench.adapters.postgres_chunk_index import (
    InMemoryChunkIdIndex,
    PostgresChunkIdIndex,
)
from stark_bench.domain import CorpusIdentity
from stark_bench.ports import ChunkIdIndex


@pytest.mark.parametrize(
    "table",
    [
        "kg_chunks; DROP TABLE users",
        "kg chunks",
        "KgChunks",
        "kg-chunks",
        'kg"chunks',
        "kg_chunks_Nemotron",
    ],
)
def test_a_table_name_that_is_not_a_bare_identifier_is_refused(table):
    with pytest.raises(ValueError, match="bare lowercase identifier"):
        PostgresChunkIdIndex("postgresql://x/y", table)


def test_the_names_this_application_actually_derives_are_accepted():
    """The guard must not reject the real inputs -- it would be caught late.

    Built from `CorpusIdentity` rather than written out, because the guard
    and the generator have to agree and a hand-typed example could drift
    from what `table_name()` produces.
    """
    for identity in (
        CorpusIdentity("Nemotron-3-Embed-1B", 2048, "passage: ", "query: "),
        CorpusIdentity("precomputed-ada002", 1536),
    ):
        PostgresChunkIdIndex("postgresql://x/y", identity.table_name())


def test_both_implementations_satisfy_the_port():
    assert isinstance(
        PostgresChunkIdIndex("postgresql://x/y", "kg_chunks"), ChunkIdIndex
    )
    assert isinstance(InMemoryChunkIdIndex(), ChunkIdIndex)


@pytest.mark.asyncio
async def test_the_in_memory_index_returns_a_copy():
    """A caller mutating the result must not corrupt the next read.

    redstring's own compliance suite requires this of every store read
    method, for a reason learned four times there: returning the live
    object is correct on every read and wrong only afterwards, so no
    assertion about the returned value can see it.
    """
    from uuid import uuid4

    tenant = uuid4()
    index = InMemoryChunkIdIndex({tenant: {"a", "b"}})
    first = await index.ids_for_tenant(tenant)
    first.add("c")
    assert await index.ids_for_tenant(tenant) == {"a", "b"}


@pytest.mark.asyncio
async def test_an_unknown_tenant_has_no_ids_rather_than_raising():
    """A first ingest is the normal case, not an error."""
    from uuid import uuid4

    assert await InMemoryChunkIdIndex().ids_for_tenant(uuid4()) == set()

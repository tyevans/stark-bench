"""The cache has to outlive the process, or a sweep's second arm pays again.

Marked `integration` because it needs the Postgres from `docker compose`;
`addopts` deselects that marker, so run it with `-m integration`.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from stark_bench.adapters.postgres_embedding_cache import PostgresEmbeddingCache
from stark_bench.ports.embedding_cache import cache_key

DSN = "postgresql://stark:stark@localhost:55432/stark"
TABLE = "kg_embedding_cache_test"


@pytest_asyncio.fixture
async def cache():
    store = await PostgresEmbeddingCache.connect(DSN, table=TABLE)
    await store.ensure_schema()
    await store.execute(f"TRUNCATE {TABLE}")
    yield store
    await store.close()


@pytest.mark.integration
async def test_a_vector_survives_a_new_connection(cache):
    key = cache_key(model="m", document_prefix="", text="hello")
    await cache.put_many({key: [0.5, -0.25, 1.0]})

    reopened = await PostgresEmbeddingCache.connect(DSN, table=TABLE)
    try:
        got = await reopened.get_many([key])
    finally:
        await reopened.close()
    assert [round(x, 4) for x in got[key]] == [0.5, -0.25, 1.0]


@pytest.mark.integration
async def test_misses_are_absent_rather_than_none(cache):
    present = cache_key(model="m", document_prefix="", text="here")
    absent = cache_key(model="m", document_prefix="", text="gone")
    await cache.put_many({present: [1.0]})
    assert set(await cache.get_many([present, absent])) == {present}


@pytest.mark.integration
async def test_putting_the_same_key_twice_does_not_raise(cache):
    """Two arms racing on the same text is normal, not an error."""
    key = cache_key(model="m", document_prefix="", text="hello")
    await cache.put_many({key: [1.0, 2.0]})
    await cache.put_many({key: [1.0, 2.0]})
    assert await cache.count() == 1


@pytest.mark.integration
async def test_dimensions_may_differ_between_rows(cache):
    """One cache serves every arm, and arms differ in dimension."""
    small = cache_key(model="nomic-embed-text", document_prefix="", text="x")
    large = cache_key(model="qwen3-embedding-0.6b", document_prefix="", text="x")
    await cache.put_many({small: [0.0] * 768, large: [0.0] * 1024})
    got = await cache.get_many([small, large])
    assert len(got[small]) == 768
    assert len(got[large]) == 1024


@pytest.mark.integration
async def test_an_empty_query_does_not_hit_the_database(cache):
    assert await cache.get_many([]) == {}
    await cache.put_many({})

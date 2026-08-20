"""An `EmbeddingCache` in Postgres, so a sweep pays the endpoint once.

Deliberately NOT the chunk table. That table is redstring's, its rows are
per-tenant, and its `chunk_id` is built from `start_char` and `chunk_index` --
so the same text under two chunkers is two rows with two ids, which is exactly
the duplication this exists to remove. A separate content-addressed table has
one row per distinct `(model, document_prefix, text)` and no notion of tenant
at all.

Stored as `REAL[]` rather than pgvector. Lookup here is by exact key and there
is no similarity search over this table, so it needs a primary key and no
vector index -- and a `vector` column would pin the table to one dimension
when the entire point is that one cache serves arms at 768, 1024 and 2048.
"""

from __future__ import annotations

import asyncpg


class PostgresEmbeddingCache:
    def __init__(self, pool: asyncpg.Pool, table: str) -> None:
        self._pool = pool
        self._table = table

    @classmethod
    async def connect(
        cls, dsn: str, *, table: str = "kg_embedding_cache"
    ) -> PostgresEmbeddingCache:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
        return cls(pool, table)

    async def ensure_schema(self) -> None:
        await self._pool.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "  key BYTEA PRIMARY KEY,"
            "  vector REAL[] NOT NULL"
            ")"
        )

    async def get_many(self, keys: list[bytes]) -> dict[bytes, list[float]]:
        if not keys:
            return {}
        rows = await self._pool.fetch(
            f"SELECT key, vector FROM {self._table} WHERE key = ANY($1::bytea[])",
            keys,
        )
        return {row["key"]: list(row["vector"]) for row in rows}

    async def put_many(self, items: dict[bytes, list[float]]) -> None:
        if not items:
            return
        # `DO NOTHING` rather than `DO UPDATE`: the key determines the vector,
        # so a conflict means two arms computed the same thing and either
        # answer is correct. Two arms racing over one corpus is normal.
        await self._pool.executemany(
            f"INSERT INTO {self._table} (key, vector) VALUES ($1, $2) "
            "ON CONFLICT (key) DO NOTHING",
            [(key, [float(x) for x in vector]) for key, vector in items.items()],
        )

    async def count(self) -> int:
        return await self._pool.fetchval(f"SELECT count(*) FROM {self._table}")

    async def execute(self, sql: str) -> None:
        await self._pool.execute(sql)

    async def close(self) -> None:
        await self._pool.close()

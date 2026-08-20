"""An `EmbeddingCache` in a dict, for tests and single-process runs."""

from __future__ import annotations


class InMemoryEmbeddingCache:
    """Not thread-safe and not bounded -- it holds one process's corpus.

    The counters are for tests. A cache that is never consulted and a cache
    that always misses are indistinguishable from the outside, and the second
    is a bug while the first is a wiring defect -- this project has shipped
    the wiring defect twice.
    """

    def __init__(self) -> None:
        self._store: dict[bytes, list[float]] = {}
        self.gets = 0
        self.puts = 0

    async def get_many(self, keys: list[bytes]) -> dict[bytes, list[float]]:
        self.gets += 1
        return {key: self._store[key] for key in keys if key in self._store}

    async def put_many(self, items: dict[bytes, list[float]]) -> None:
        self.puts += 1
        self._store.update(items)

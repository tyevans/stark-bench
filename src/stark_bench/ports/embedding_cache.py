"""Content-addressed lookup for vectors this benchmark has already computed.

The key is the whole point. `_table_for` already folds the model and both
prefixes into the chunk table name, because a corpus embedded with a prefix
and the same corpus embedded without it are not comparable vectors (ADR 0002,
ADR 0043). A cache keyed on the text alone would undo that in a way nothing
downstream could detect: the store fills, every count is right, and cosine
similarity between two models' vectors returns a plausible number.

The query prefix is deliberately NOT in the key. This caches the corpus side
only -- `embed`, not `embed_query` -- and query vectors are computed 280 at a
time, which is not worth caching and would be a second place to get the key
wrong.
"""

from __future__ import annotations

import hashlib
from typing import Protocol


def cache_key(*, model: str, document_prefix: str, text: str) -> bytes:
    """A 32-byte key over the three things that change the resulting vector.

    Fields are joined with a NUL, which cannot occur in a model id or in a
    STaRK document, so ("ab", "c") and ("a", "bc") cannot collide into one
    key. Length-prefixing would also work; a separator is cheaper to read.
    """
    digest = hashlib.sha256()
    for field in (model, document_prefix, text):
        digest.update(field.encode("utf-8"))
        digest.update(b"\x00")
    return digest.digest()


class EmbeddingCache(Protocol):
    """Batch in, batch out. Misses are absent from the result, not `None`.

    Batched because the alternative is one round trip per chunk, and a group
    here is `concurrency * embed_batch` chunks -- the cache must not cost more
    than the embedding it saves.
    """

    async def get_many(self, keys: list[bytes]) -> dict[bytes, list[float]]: ...

    async def put_many(self, items: dict[bytes, list[float]]) -> None: ...

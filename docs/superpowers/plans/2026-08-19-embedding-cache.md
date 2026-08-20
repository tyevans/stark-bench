# Embedding Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop re-embedding chunk text this benchmark has already embedded, so every arm after the first costs a fraction of an endpoint pass.

**Architecture:** A content-addressed vector cache in Postgres, keyed on `(model, document_prefix, sha256(text))`, consulted inside `_embed_group` in `stark_bench/adapters/stark_ingest_engine.py`. Hits are served from the table; only misses reach the provider, and the surviving misses are still batched exactly as they are today. The cache is an optional constructor argument — `None` reproduces current behaviour byte for byte.

**Tech Stack:** Python 3.13, `asyncpg`, Postgres 16 (`docker compose`, port 55432), `pytest`.

**Spec:** This document. The design rationale is in Background below; there is no separate spec file.

## Background — why this is worth building

Arms differ only by chunker, and most documents are too short for a chunker to touch:

| corpus | documents unaffected by chunking |
|---|---|
| `prime` | 86% under 1,000 chars |
| `prime-rel` | 80.5% under 2,400 chars |

Every config gets its own `tenant_id` in a shared chunk table (ADR 0002), so the *same chunk text* is embedded and stored once per arm. A three-chunker sweep pays roughly 3× for embedding while ~80% of the resulting vectors are byte-identical duplicates. Measured today, one `prime-rel` arm is ~2 hours; the second and third are almost entirely re-work.

This does not replace the existing resume path. Resume skips chunks **already stored in this tenant**. This cache skips **embedding text seen in any tenant, ever** — including across runs, and including text whose `chunk_id` differs because `start_char` or `chunk_index` differ under another chunker.

## Global Constraints

- **The cache key MUST include the model and the document prefix.** ADR 0002 and ADR 0043: a corpus embedded with a prefix and the same corpus embedded without it are not comparable vectors, and two models' vectors must never mix. A key of `sha256(text)` alone silently mixes them, and cosine similarity between mixed vectors returns a perfectly plausible number. This is the single highest-risk defect in the feature.
- **`stark_bench.agents` may not import `harness`, `skb`, or `sidecar`** — enforced by `lint-imports` on commit. This work touches none of those, but do not add imports that cross it.
- **Never edit dependency tables by hand** — use `uv add` / `uv remove`.
- **Stage explicit paths, never `git add -A`** — the working tree routinely carries the `pyproject.toml`/`uv.lock` redstring path switch.
- Quality gates run on `git commit` via pre-commit. Do not run them separately first; commit and re-`git add` if hooks reformat.
- Run tests with `uv run pytest -p no:randomly`. The `integration` marker is deselected by `addopts`; select it explicitly when you mean it.
- Vectors are `list[float]`. Dimension comes from the config, not from the cache.

---

### Task 1: The cache port and an in-memory implementation

**Files:**
- Create: `src/stark_bench/ports/embedding_cache.py`
- Create: `src/stark_bench/adapters/memory_embedding_cache.py`
- Test: `tests/adapters/test_embedding_cache_key.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `cache_key(*, model: str, document_prefix: str, text: str) -> bytes` in `stark_bench.ports.embedding_cache`
  - `EmbeddingCache` protocol with `async get_many(keys: list[bytes]) -> dict[bytes, list[float]]` and `async put_many(items: dict[bytes, list[float]]) -> None`
  - `InMemoryEmbeddingCache` in `stark_bench.adapters.memory_embedding_cache`, implementing that protocol, with a `.gets` and `.puts` call counter for tests

- [ ] **Step 1: Write the failing test**

```python
"""The key must separate what ADR 0002 and ADR 0043 say are different vectors.

A cache keyed on the text alone would serve a nomic vector to a qwen arm, or
an unprefixed vector to a prefixed one. Both produce a fully-populated store
whose cosine similarities are perfectly plausible and quietly wrong -- the
exact failure shape CLAUDE.md records six times over.
"""

from __future__ import annotations

import pytest

from stark_bench.ports.embedding_cache import cache_key
from stark_bench.adapters.memory_embedding_cache import InMemoryEmbeddingCache


def test_same_inputs_give_the_same_key():
    a = cache_key(model="m", document_prefix="p: ", text="hello")
    b = cache_key(model="m", document_prefix="p: ", text="hello")
    assert a == b


def test_a_different_model_is_a_different_key():
    a = cache_key(model="nomic-embed-text", document_prefix="", text="hello")
    b = cache_key(model="qwen3-embedding-0.6b", document_prefix="", text="hello")
    assert a != b


def test_a_different_prefix_is_a_different_key():
    a = cache_key(model="m", document_prefix="passage: ", text="hello")
    b = cache_key(model="m", document_prefix="search_document: ", text="hello")
    assert a != b


def test_an_absent_prefix_differs_from_an_empty_one_only_if_the_text_differs():
    """Empty prefix is a real value -- qwen uses it -- not a missing one."""
    a = cache_key(model="m", document_prefix="", text="hello")
    b = cache_key(model="m", document_prefix="", text="hello")
    assert a == b


def test_fields_cannot_be_smeared_into_each_other():
    """Concatenation without a separator makes ("ab","c") and ("a","bc") collide.

    A cache that collides two configs serves one arm's vectors to another.
    """
    a = cache_key(model="ab", document_prefix="c", text="x")
    b = cache_key(model="a", document_prefix="bc", text="x")
    assert a != b


@pytest.mark.asyncio
async def test_in_memory_cache_round_trips_and_reports_misses():
    cache = InMemoryEmbeddingCache()
    k = cache_key(model="m", document_prefix="", text="hello")
    assert await cache.get_many([k]) == {}
    await cache.put_many({k: [1.0, 2.0]})
    assert await cache.get_many([k]) == {k: [1.0, 2.0]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:randomly tests/adapters/test_embedding_cache_key.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stark_bench.ports.embedding_cache'`

- [ ] **Step 3: Write minimal implementation**

`src/stark_bench/ports/embedding_cache.py`:

```python
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
    """Batch in, batch out. Misses are absent from the result, not None.

    Batched because the alternative is one round trip per chunk, and a group
    here is `concurrency * embed_batch` chunks -- the cache must not cost more
    than the embedding it saves.
    """

    async def get_many(self, keys: list[bytes]) -> dict[bytes, list[float]]: ...

    async def put_many(self, items: dict[bytes, list[float]]) -> None: ...
```

`src/stark_bench/adapters/memory_embedding_cache.py`:

```python
"""An `EmbeddingCache` in a dict, for tests and for `--cache-memory` runs."""

from __future__ import annotations


class InMemoryEmbeddingCache:
    """Not thread-safe and not bounded -- it holds one process's corpus.

    Counters are for tests: a cache that is never consulted and a cache that
    always misses are indistinguishable from the outside, and the second is a
    bug while the first is a wiring defect.
    """

    def __init__(self) -> None:
        self._store: dict[bytes, list[float]] = {}
        self.gets = 0
        self.puts = 0

    async def get_many(self, keys: list[bytes]) -> dict[bytes, list[float]]:
        self.gets += 1
        return {k: self._store[k] for k in keys if k in self._store}

    async def put_many(self, items: dict[bytes, list[float]]) -> None:
        self.puts += 1
        self._store.update(items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -p no:randomly tests/adapters/test_embedding_cache_key.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/stark_bench/ports/embedding_cache.py \
        src/stark_bench/adapters/memory_embedding_cache.py \
        tests/adapters/test_embedding_cache_key.py
git commit -m "Add an embedding cache port keyed on model, prefix and text"
```

---

### Task 2: Consult the cache inside the ingest engine

**Files:**
- Modify: `src/stark_bench/adapters/stark_ingest_engine.py` — the `ingest` signature (~line 118) and `_embed_group` (~line 294)
- Test: `tests/adapters/test_ingest_uses_the_cache.py`

**Interfaces:**
- Consumes: `cache_key`, `EmbeddingCache`, `InMemoryEmbeddingCache` from Task 1.
- Produces:
  - `ingest(..., embedding_cache: EmbeddingCache | None = None, cache_model: str = "", cache_document_prefix: str = "")` — three new keyword-only arguments, all defaulting to the current behaviour.
  - `IngestReport` gains `cache_hits: int` and `cache_misses: int`.

**Why the wrap goes at `_embed_group` and not at the provider:** `_embed_group` is the single place a flat list of texts becomes a flat list of vectors, in order. Wrapping the provider instead would cache after batching, so a batch containing one miss would still send all 32 texts. Wrapping here lets the misses re-batch.

- [ ] **Step 1: Write the failing test**

```python
"""A second arm over the same text must not call the provider again.

The provider here returns a vector encoding its own input (the pattern from
`test_ingest_batching.py`), so this can assert the CACHED vectors are the
right ones and not merely that some vector arrived. A cache that returns a
plausible wrong vector is the defect this whole feature could introduce.
"""

from __future__ import annotations

import pytest
from redstring import InMemoryChunkStore, InMemoryGraphStore, TenantId
from uuid import uuid4

from stark_bench.adapters.chunkers import WholeDocumentChunker
from stark_bench.adapters.memory_embedding_cache import InMemoryEmbeddingCache
from stark_bench.adapters.stark_artifacts import SkbNode
from stark_bench.adapters.stark_ingest_engine import ingest

DIM = 8


def _fingerprint(text: str) -> list[float]:
    body = [float(ord(c)) for c in text[: DIM - 1]]
    return [float(len(text)), *body, *([0.0] * (DIM - 1 - len(body)))][:DIM]


class CountingProvider:
    model = "fingerprint"
    dimension = DIM

    def __init__(self) -> None:
        self.texts_embedded: list[str] = []

    async def embed(self, texts):
        self.texts_embedded.extend(texts)
        return [_fingerprint(t) for t in texts]

    async def embed_query(self, texts):
        return await self.embed(texts)


def _nodes():
    return [
        SkbNode(node_id="1", node_type="gene/protein", name="A", document="alpha doc"),
        SkbNode(node_id="2", node_type="gene/protein", name="B", document="beta doc"),
    ]


async def _run(provider, cache, tenant):
    chunks = InMemoryChunkStore()
    return await ingest(
        _nodes(),
        iter(()),
        dataset="prime",
        tenant_id=tenant,
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
        embedding_cache=cache,
        cache_model="fingerprint",
        cache_document_prefix="",
    ), chunks


@pytest.mark.asyncio
async def test_a_second_tenant_embeds_nothing_and_stores_the_right_vectors():
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()

    first, _ = await _run(provider, cache, TenantId(uuid4()))
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]
    assert first.cache_hits == 0
    assert first.cache_misses == 2

    provider.texts_embedded.clear()
    second, chunks = await _run(provider, cache, TenantId(uuid4()))

    assert provider.texts_embedded == [], "the second arm re-embedded text"
    assert second.cache_hits == 2
    assert second.cache_misses == 0

    # The vectors must be the ones belonging to these texts, not merely present.
    stored = {c.text: list(c.embedding) for c in chunks.all_chunks()}
    assert stored == {
        "alpha doc": _fingerprint("alpha doc"),
        "beta doc": _fingerprint("beta doc"),
    }


@pytest.mark.asyncio
async def test_a_different_prefix_does_not_hit_the_cache():
    """The failure this feature could introduce, asserted directly."""
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()
    await _run(provider, cache, TenantId(uuid4()))
    provider.texts_embedded.clear()

    chunks = InMemoryChunkStore()
    report = await ingest(
        _nodes(),
        iter(()),
        dataset="prime",
        tenant_id=TenantId(uuid4()),
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
        embedding_cache=cache,
        cache_model="fingerprint",
        cache_document_prefix="passage: ",   # <-- the only change
    )
    assert report.cache_hits == 0
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]


@pytest.mark.asyncio
async def test_no_cache_is_unchanged_behaviour():
    provider = CountingProvider()
    chunks = InMemoryChunkStore()
    report = await ingest(
        _nodes(),
        iter(()),
        dataset="prime",
        tenant_id=TenantId(uuid4()),
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
    )
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]
    assert report.cache_hits == 0 and report.cache_misses == 0


@pytest.mark.asyncio
async def test_a_partial_hit_embeds_only_the_missing_text():
    """The interesting case: order must survive a mix of hits and misses."""
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()
    await _run(provider, cache, TenantId(uuid4()))
    provider.texts_embedded.clear()

    extra = [*_nodes(), SkbNode(node_id="3", node_type="gene/protein", name="C", document="gamma doc")]
    chunks = InMemoryChunkStore()
    report = await ingest(
        extra,
        iter(()),
        dataset="prime",
        tenant_id=TenantId(uuid4()),
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
        embedding_cache=cache,
        cache_model="fingerprint",
        cache_document_prefix="",
    )
    assert provider.texts_embedded == ["gamma doc"]
    assert report.cache_hits == 2 and report.cache_misses == 1
    stored = {c.text: list(c.embedding) for c in chunks.all_chunks()}
    assert stored["gamma doc"] == _fingerprint("gamma doc")
    assert stored["alpha doc"] == _fingerprint("alpha doc")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:randomly tests/adapters/test_ingest_uses_the_cache.py -v`
Expected: FAIL — `TypeError: ingest() got an unexpected keyword argument 'embedding_cache'`

- [ ] **Step 3: Write minimal implementation**

In `src/stark_bench/adapters/stark_ingest_engine.py`, add to the `IngestReport` dataclass (near the existing `edges: int` at ~line 94):

```python
    #: Chunk texts served from the cache rather than the endpoint, and those
    #: that had to be embedded. Reported because a sweep's second arm should
    #: be almost entirely hits, and a run that is not is telling you the key
    #: is wrong -- which is otherwise invisible.
    cache_hits: int = 0
    cache_misses: int = 0
```

Add the three keyword-only parameters to `ingest` after `existing_chunk_ids`:

```python
    embedding_cache: EmbeddingCache | None = None,
    cache_model: str = "",
    cache_document_prefix: str = "",
```

Add near the other counters, before `_plan` is defined:

```python
    cache_hits = 0
    cache_misses = 0
```

Replace the body of `_embed_group` with:

```python
    async def _embed_group(texts: list[str]) -> list[list[float]]:
        """One flat list of texts in, one flat list of vectors out, in order.

        Slices into `embed_batch`-sized requests and runs `concurrency` of
        them at a time. The result is reassembled by slice index rather than
        by completion order, because `asyncio.gather` preserves argument
        order but a future reader should not have to know that to trust the
        alignment -- and misaligning vectors with texts is a defect that
        produces a fully-populated store scoring like noise, with nothing
        raising anywhere.

        The cache is consulted HERE rather than around the provider so that
        the surviving misses re-batch. Wrapping the provider would leave a
        32-text request carrying 31 texts it already had.
        """
        nonlocal cache_hits, cache_misses
        if not texts:
            return []

        if embedding_cache is None:
            hits: dict[bytes, list[float]] = {}
            keys = [b""] * len(texts)
        else:
            keys = [
                cache_key(
                    model=cache_model,
                    document_prefix=cache_document_prefix,
                    text=t,
                )
                for t in texts
            ]
            hits = await embedding_cache.get_many(list(dict.fromkeys(keys)))
            cache_hits += sum(1 for k in keys if k in hits)

        missing_positions = [i for i, k in enumerate(keys) if k not in hits]
        cache_misses += len(missing_positions) if embedding_cache is not None else 0
        missing = [texts[i] for i in missing_positions]

        fetched: list[list[float]] = []
        if missing:
            slices = [
                missing[i : i + embed_batch]
                for i in range(0, len(missing), embed_batch)
            ]
            for start in range(0, len(slices), concurrency):
                wave = slices[start : start + concurrency]
                for result in await asyncio.gather(
                    *(embeddings.embed(s) for s in wave)
                ):
                    fetched.extend(result)
            if len(fetched) != len(missing):
                raise ValueError(
                    f"embedding provider returned {len(fetched)} vectors for "
                    f"{len(missing)} texts; the port promises one per input, "
                    "in order"
                )

        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        for position, vector in zip(missing_positions, fetched, strict=True):
            out[position] = vector
        for i, key in enumerate(keys):
            if out[i] is None:
                out[i] = hits[key]

        if embedding_cache is not None and missing:
            await embedding_cache.put_many(
                {
                    keys[position]: vector
                    for position, vector in zip(
                        missing_positions, fetched, strict=True
                    )
                }
            )

        if any(v is None for v in out):
            raise ValueError("a text was neither cached nor embedded")
        return out
```

Add the import at the top of the file, beside the other `stark_bench` imports:

```python
from stark_bench.ports.embedding_cache import EmbeddingCache, cache_key
```

And pass the counters into the returned report:

```python
    return IngestReport(
        nodes=node_count,
        edges=edge_count,
        self_loops_dropped=dropped,
        chunks=chunk_count,
        skipped=skipped_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -p no:randomly tests/adapters/ -v`
Expected: PASS — the 5 new tests plus every existing `test_ingest_batching.py` test, which must still pass unchanged since `embedding_cache` defaults to `None`.

- [ ] **Step 5: Prove the test can fail**

This repo's standing rule: break the implementation on purpose and watch the suite go red. Temporarily drop `cache_document_prefix` from the key construction in `_embed_group` (pass `document_prefix=""` unconditionally), then run:

Run: `uv run pytest -p no:randomly tests/adapters/test_ingest_uses_the_cache.py -v`
Expected: `test_a_different_prefix_does_not_hit_the_cache` FAILS. Restore the line and confirm green before committing. If it passes with the defect in place, the test is worthless — fix the test, not the implementation.

- [ ] **Step 6: Commit**

```bash
git add src/stark_bench/adapters/stark_ingest_engine.py \
        tests/adapters/test_ingest_uses_the_cache.py
git commit -m "Serve repeated chunk text from the cache instead of the endpoint"
```

---

### Task 3: A Postgres-backed cache that survives the process

**Files:**
- Create: `src/stark_bench/adapters/postgres_embedding_cache.py`
- Test: `tests/adapters/test_postgres_embedding_cache.py`

**Interfaces:**
- Consumes: `cache_key`, `EmbeddingCache` from Task 1.
- Produces: `PostgresEmbeddingCache.connect(dsn: str, *, table: str = "kg_embedding_cache") -> PostgresEmbeddingCache`, with `ensure_schema()`, `get_many`, `put_many`, `close()`, and `count()`.

**Why a plain `REAL[]` and not pgvector:** this table is looked up by exact key only — there is no similarity search over it — so it needs a primary key and no vector index. A `vector` column would additionally pin the table to one dimension, and the whole point is that one cache serves every arm.

- [ ] **Step 1: Write the failing test**

```python
"""The cache has to outlive the process, or a sweep's second arm pays again.

Marked `integration` because it needs the Postgres from `docker compose`;
`addopts` deselects that marker, so run it with `-m integration`.
"""

from __future__ import annotations

import pytest

from stark_bench.adapters.postgres_embedding_cache import PostgresEmbeddingCache
from stark_bench.ports.embedding_cache import cache_key

DSN = "postgresql://stark:stark@localhost:55432/stark"
TABLE = "kg_embedding_cache_test"


@pytest.fixture
async def cache():
    c = await PostgresEmbeddingCache.connect(DSN, table=TABLE)
    await c.ensure_schema()
    await c.execute(f"TRUNCATE {TABLE}")
    yield c
    await c.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_vector_survives_a_new_connection(cache):
    k = cache_key(model="m", document_prefix="", text="hello")
    await cache.put_many({k: [0.5, -0.25, 1.0]})
    await cache.close()

    reopened = await PostgresEmbeddingCache.connect(DSN, table=TABLE)
    try:
        assert await reopened.get_many([k]) == {k: [0.5, -0.25, 1.0]}
    finally:
        await reopened.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_misses_are_absent_rather_than_none(cache):
    present = cache_key(model="m", document_prefix="", text="here")
    absent = cache_key(model="m", document_prefix="", text="gone")
    await cache.put_many({present: [1.0]})
    got = await cache.get_many([present, absent])
    assert set(got) == {present}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_putting_the_same_key_twice_does_not_raise(cache):
    """Two arms racing on the same text is normal, not an error."""
    k = cache_key(model="m", document_prefix="", text="hello")
    await cache.put_many({k: [1.0, 2.0]})
    await cache.put_many({k: [1.0, 2.0]})
    assert await cache.count() == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dimensions_may_differ_between_rows(cache):
    """One cache serves every arm, and arms differ in dimension."""
    a = cache_key(model="nomic-embed-text", document_prefix="", text="x")
    b = cache_key(model="qwen3-embedding-0.6b", document_prefix="", text="x")
    await cache.put_many({a: [0.0] * 768, b: [0.0] * 1024})
    got = await cache.get_many([a, b])
    assert len(got[a]) == 768 and len(got[b]) == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:randomly -m integration tests/adapters/test_postgres_embedding_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stark_bench.adapters.postgres_embedding_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
"""An `EmbeddingCache` in Postgres, so a sweep pays the endpoint once.

Deliberately NOT in the chunk table. That table is redstring's, its rows are
per-tenant, and its `chunk_id` is built from `start_char` and `chunk_index` --
so the same text under two chunkers is two rows with two ids, which is the
duplication this exists to remove. A separate content-addressed table has one
row per distinct (model, prefix, text) and no notion of tenant at all.

Stored as `REAL[]` rather than pgvector: lookup is by exact key, there is no
similarity search over this table, and a `vector` column would pin it to one
dimension when the entire point is that one cache serves arms at 768, 1024
and 2048 dimensions.
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
        # ON CONFLICT DO NOTHING rather than DO UPDATE: the key determines the
        # vector, so a conflict means two arms computed the same thing and
        # either answer is correct. Racing arms are normal.
        await self._pool.executemany(
            f"INSERT INTO {self._table} (key, vector) VALUES ($1, $2) "
            "ON CONFLICT (key) DO NOTHING",
            [(k, [float(x) for x in v]) for k, v in items.items()],
        )

    async def count(self) -> int:
        return await self._pool.fetchval(f"SELECT count(*) FROM {self._table}")

    async def execute(self, sql: str) -> None:
        await self._pool.execute(sql)

    async def close(self) -> None:
        await self._pool.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d && uv run pytest -p no:randomly -m integration tests/adapters/test_postgres_embedding_cache.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/stark_bench/adapters/postgres_embedding_cache.py \
        tests/adapters/test_postgres_embedding_cache.py
git commit -m "Persist the embedding cache in Postgres, content-addressed"
```

---

### Task 4: Wire the cache into the CLI, and prove the call site uses it

**Files:**
- Modify: `src/stark_bench/composition/cli.py` — `_do_ingest` (~line 451-510) and the argument parser
- Test: `tests/composition/test_cli_passes_the_cache.py`

**Interfaces:**
- Consumes: `PostgresEmbeddingCache` from Task 3; the `ingest(...)` keywords from Task 2.
- Produces: a `--no-cache` CLI flag; `_do_ingest` constructs and passes the cache.

**This task exists because of a specific failure mode.** CLAUDE.md records it twice in one session: `_live_embeddings_for` was reverted to drop both `*_prefix=` arguments and all 39 tests passed; `write_report` was reverted to `ingest={}` and all 45 passed. Both times the helper was exhaustively tested and nothing asserted the call site used it. An AST check on the call site is the documented remedy — see `tests/composition/test_ingest_stats_reach_the_report.py`.

- [ ] **Step 1: Write the failing test**

```python
"""A perfect cache nobody passes is a cache that does nothing.

This repo has shipped that exact defect twice in one session -- an
exhaustively tested helper and a call site that stopped calling it, with a
fully green suite both times. Running `_do_ingest` for real needs Postgres,
Neo4j and a live endpoint, so the call site is checked by reading the source.
"""

from __future__ import annotations

import ast
import inspect

from stark_bench.composition import cli


def _ingest_call() -> ast.Call:
    tree = ast.parse(inspect.getsource(cli._do_ingest))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "ingest_corpus":
                return node
    raise AssertionError("_do_ingest no longer calls ingest_corpus")


def test_the_cache_reaches_the_ingest():
    passed = {kw.arg for kw in _ingest_call().keywords}
    assert "embedding_cache" in passed, "_do_ingest built a cache and did not pass it"


def test_the_key_fields_reach_the_ingest():
    """A cache without the model and prefix mixes two arms' vectors."""
    passed = {kw.arg for kw in _ingest_call().keywords}
    assert "cache_model" in passed
    assert "cache_document_prefix" in passed


def test_no_cache_flag_exists():
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "no_cache")
    assert action.const is True or action.nargs == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:randomly tests/composition/test_cli_passes_the_cache.py -v`
Expected: FAIL — `AssertionError: _do_ingest built a cache and did not pass it`

- [ ] **Step 3: Write minimal implementation**

In `cli.py`, add to the parser beside `--no-resume`:

```python
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Re-embed every chunk instead of reusing vectors this benchmark "
            "has already computed for the same (model, document_prefix, text). "
            "The cache is on by default and is what makes a chunking sweep "
            "cost roughly one endpoint pass instead of three -- ~80%% of this "
            "corpus is short enough that every chunker produces identical "
            "text. Pass this to force a cold run, which is the escape hatch: "
            "reusing work is exactly the kind of optimisation that can hide a "
            "bug."
        ),
    )
```

In `_do_ingest`, after the chunk store is opened and before `ingest_corpus` is called:

```python
    embedding_cache = None
    if embeddings is not None and not no_cache:
        embedding_cache = await PostgresEmbeddingCache.connect(POSTGRES_DSN)
        await embedding_cache.ensure_schema()
```

Pass it through, and close it in the existing `finally`:

```python
        outcome = await ingest_corpus(
            ...,
            embedding_cache=embedding_cache,
            cache_model=config.embeddings,
            cache_document_prefix=config.document_prefix,
            ...
        )
```

```python
    finally:
        await chunks.close()
        await graph.close()
        if embedding_cache is not None:
            await embedding_cache.close()
```

Thread `no_cache` from `main` into `_do_ingest` as a keyword argument, matching how `no_resume` is already threaded.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -p no:randomly -v`
Expected: PASS — the whole suite, including the 3 new tests.

- [ ] **Step 5: Prove the test can fail**

Delete the `embedding_cache=embedding_cache` line from the `ingest_corpus` call, run the suite, and confirm `test_the_cache_reaches_the_ingest` fails while everything else stays green. That green-everything-else is the point: it is what the two historical defects looked like. Restore the line.

- [ ] **Step 6: Commit**

```bash
git add src/stark_bench/composition/cli.py \
        tests/composition/test_cli_passes_the_cache.py
git commit -m "Pass the embedding cache from the CLI, and test the call site"
```

---

### Task 5: Report the saving, and document it

**Files:**
- Modify: `src/stark_bench/composition/cli.py` — `_ingest_stats` / `write_report` wiring
- Modify: `CLAUDE.md` — the "Running it" section
- Modify: `BACKLOG.md` — delete nothing; this plan closes no existing entry
- Test: `tests/composition/test_cache_stats_reach_the_report.py`

**Interfaces:**
- Consumes: `IngestReport.cache_hits` / `.cache_misses` from Task 2.
- Produces: `cache_hits` and `cache_misses` keys in the `ingest` block of every results file.

- [ ] **Step 1: Write the failing test**

```python
"""A cost number that does not reach the report cannot be read.

Same shape as `test_ingest_stats_reach_the_report.py`, and for the same
reason: `write_report(ingest={})` once emptied the cost column of every
report ever written, and nothing raised.
"""

from __future__ import annotations

import json

import pytest

from stark_bench.composition.cli import _ingest_stats, ingest_report_path
from stark_bench.domain.run_config import RunConfig


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        name="test-cache-stats",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="qwen3-embedding-0.6b",
        dimension=1024,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )


def test_cache_counters_survive_the_process_boundary(config, monkeypatch, tmp_path):
    monkeypatch.setattr("stark_bench.composition.cli.RESULTS_ROOT", tmp_path)
    path = ingest_report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"nodes": 10, "chunks": 12, "cache_hits": 9, "cache_misses": 3}
        )
    )
    stats = _ingest_stats(config)
    assert stats["cache_hits"] == 9
    assert stats["cache_misses"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:randomly tests/composition/test_cache_stats_reach_the_report.py -v`
Expected: FAIL if `_ingest_stats` filters keys; PASS immediately if it passes the dict through. **If it passes without any change, that is the correct outcome** — delete nothing, keep the test as the regression guard, and note it in the commit.

- [ ] **Step 3: Write minimal implementation**

Only if Step 2 failed: widen whatever key list `_ingest_stats` uses to include `cache_hits` and `cache_misses`.

Then add to `CLAUDE.md` under "Running it":

```markdown
Chunk vectors are cached across arms, content-addressed on `(model,
document_prefix, sha256(text))` in `kg_embedding_cache`. ~80% of these
corpora are short enough that every chunker emits identical text, so a
three-chunker sweep costs roughly **one** endpoint pass rather than three,
and re-running an arm after a config change costs only what actually changed.

The key carries the model and the prefix because a corpus embedded with a
prefix and the same corpus embedded without it are not comparable vectors
(ADR 0002, ADR 0043) -- a cache keyed on text alone would serve one arm's
vectors to another and nothing downstream could tell.

`--no-cache` forces a cold run. Every report carries `cache_hits` and
`cache_misses`, and a sweep's second arm that is not almost entirely hits is
telling you the key is wrong.
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -p no:randomly -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stark_bench/composition/cli.py CLAUDE.md \
        tests/composition/test_cache_stats_reach_the_report.py
git commit -m "Report cache hits and misses, and document the key"
```

---

### Task 6: Measure it on the mini set before trusting it

**Files:**
- No source changes. This task is a verification gate.

- [ ] **Step 1: Cold run, cache cleared**

```bash
docker compose up -d
uv run python - <<'PY'
import asyncio
from stark_bench.adapters.postgres_embedding_cache import PostgresEmbeddingCache
async def m():
    c = await PostgresEmbeddingCache.connect("postgresql://stark:stark@localhost:55432/stark")
    await c.ensure_schema()
    await c.execute("TRUNCATE kg_embedding_cache")
    await c.close()
asyncio.run(m())
PY
time uv run python -m stark_bench.composition.cli --config config/qwen-mini-wholedoc.yaml \
    --ingest --no-resume --embed-concurrency 2 --embed-batch 32
```

Record wall time and the reported `cache_hits` / `cache_misses`. Expected: hits 0, misses equal to chunk count.

- [ ] **Step 2: Warm run on a different chunker over the same corpus**

```bash
time uv run python -m stark_bench.composition.cli --config config/qwen-mini-sliding1k.yaml \
    --ingest --no-resume --embed-concurrency 2 --embed-batch 32
```

Expected: a large majority of hits, and wall time down sharply. `prime-mini` is a random sample of PRIME, so ~86% of its documents are under 1,000 characters and produce identical text under both chunkers.

- [ ] **Step 3: Prove the vectors are actually right, not merely fast**

A fast wrong answer is the failure this feature can produce. Compare a warm-cache arm against the pre-cache scores already in `results/`:

```bash
uv run python -m stark_bench.composition.cli --config config/qwen-mini-wholedoc.yaml --run --agent dense
```

Expected: **mrr exactly 0.3590479185937771**, the figure recorded before the cache existed. Not "close" — identical. Embedding is deterministic at temperature zero and the cache either returns the same vector or a different one.

If it differs at all, stop and do not proceed: the cache is serving wrong vectors, which is precisely the silent defect this repo has hit six times.

- [ ] **Step 4: Record the numbers**

Add the measured cold and warm wall times, and the hit rate, to the `CLAUDE.md` paragraph from Task 5, replacing any estimate with the measurement. An unmeasured speedup claim in this repo is a claim that will be wrong — CLAUDE.md's own "6x from batching" story is the precedent.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "Record the measured cache hit rate and speedup"
```

---

## Self-Review

**Spec coverage.** The Background asks for one thing — stop re-embedding identical chunk text across arms — and Tasks 1-4 deliver it end to end (key, engine, persistence, wiring), Task 5 makes the saving legible, Task 6 verifies it. The `--no-cache` escape hatch, the cross-dimension requirement, and the model/prefix key constraint from Global Constraints each have a named test.

**Placeholder scan.** Every code step carries real code. Task 5 Step 3 is conditional on Step 2's outcome, which is stated explicitly rather than left as "adjust as needed". Task 6 has no source changes by design and says so.

**Type consistency.** `cache_key(*, model, document_prefix, text) -> bytes` is defined in Task 1 and called with those exact keywords in Tasks 2, 3 and 6. `get_many(list[bytes]) -> dict[bytes, list[float]]` and `put_many(dict[bytes, list[float]]) -> None` are consistent across the protocol, the in-memory adapter, the Postgres adapter and the engine. `IngestReport.cache_hits` / `.cache_misses` are introduced in Task 2 and read in Task 5.

**One known gap, deliberately left.** Nothing evicts from `kg_embedding_cache`. At ~1.2M distinct chunks across these corpora and 1024 floats each, the table is on the order of 5 GB — acceptable on this machine and simpler than a policy nobody will tune. If it becomes a problem, `TRUNCATE` is the answer and `--no-cache` is the bypass.

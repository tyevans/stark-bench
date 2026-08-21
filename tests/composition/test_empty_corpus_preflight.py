"""A `--run` against an empty store must refuse, not score.

B-EPHEMERAL-STORES-1's detection half. The volumes are fixed; what was open
is that a store which is UP and EMPTY fails nowhere.

Two real incidents. A per-model chunk-table rename orphaned `vss-control`'s
corpus: all 280 queries retrieved nothing, `runner.run` logged zero failures
because retrieval SUCCEEDED and returned empty, and the only symptom was a
`ValueError: min() arg is an empty sequence` from inside a 3.11 subprocess.
Separately a `docker compose down`-shaped event took 589,790 chunks; that
one at least failed loudly, with a `ConnectionRefusedError` that reads as
"the container is down" rather than "the corpus is gone".

The empty-but-up case is the dangerous one: the arms would score
low-but-plausible numbers and be read as a bad retriever.
"""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import stark_bench.composition.cli as cli_mod
from stark_bench.adapters.postgres_chunk_index import InMemoryChunkIdIndex

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")


async def test_the_index_counts_without_materialising_every_id() -> None:
    """549,886 ids is a lot of strings to build in order to compare to zero."""
    tenant = uuid4()
    index = InMemoryChunkIdIndex({tenant: {"a", "b", "c"}})
    assert await index.count_for_tenant(tenant) == 3


async def test_an_unknown_tenant_counts_zero() -> None:
    assert await InMemoryChunkIdIndex({}).count_for_tenant(uuid4()) == 0


async def test_an_empty_corpus_raises(monkeypatch) -> None:
    tenant = uuid4()

    class Empty:
        def __init__(self, *a, **k) -> None: ...
        async def count_for_tenant(self, t: UUID) -> int:
            return 0

    monkeypatch.setattr(cli_mod, "PostgresChunkIdIndex", Empty)
    config = _config()
    with pytest.raises(cli_mod.EmptyCorpusError) as excinfo:
        await cli_mod._require_a_corpus(config, tenant)
    message = str(excinfo.value)
    assert config.name in message
    assert str(tenant) in message, "the tenant must be named; arms share a table"
    assert "--ingest" in message, "say what to do about it"


async def test_a_populated_corpus_passes(monkeypatch) -> None:
    class Full:
        def __init__(self, *a, **k) -> None: ...
        async def count_for_tenant(self, t: UUID) -> int:
            return 129_375

    monkeypatch.setattr(cli_mod, "PostgresChunkIdIndex", Full)
    await cli_mod._require_a_corpus(_config(), uuid4())


async def test_one_single_chunk_is_enough_to_proceed(monkeypatch) -> None:
    """The preflight asserts non-empty, not complete. `verify_corpus.py` is
    the tool for completeness, and conflating them would make a partial
    resume impossible."""

    class Barely:
        def __init__(self, *a, **k) -> None: ...
        async def count_for_tenant(self, t: UUID) -> int:
            return 1

    monkeypatch.setattr(cli_mod, "PostgresChunkIdIndex", Barely)
    await cli_mod._require_a_corpus(_config(), uuid4())


def test_the_run_calls_the_preflight() -> None:
    """Correct and unreachable is this repo's signature defect."""
    body = _SRC.split("async def _do_run")[1]
    assert "_require_a_corpus(" in body


def test_the_preflight_runs_before_any_query() -> None:
    """Failing after an hour of retrieval is the outcome it exists to
    prevent."""
    body = _SRC.split("async def _do_run")[1].split("\nasync def ")[0]
    assert body.index("_require_a_corpus(") < body.index("await run(")


def test_the_preflight_is_scoped_to_the_tenant() -> None:
    """Configs sharing a model share a table. A bare count would pass on
    another arm's rows -- the exact confusion CLAUDE.md records making one
    arm at 133,919 read as 141,673."""
    body = _SRC.split("async def _require_a_corpus")[1].split("\nasync def ")[0]
    assert "count_for_tenant" in body
    tree = ast.parse(_SRC)
    assert any(
        isinstance(n, ast.ClassDef) and n.name == "EmptyCorpusError"
        for n in ast.walk(tree)
    )


def _config():
    from stark_bench.domain.run_config import RunConfig

    return RunConfig(
        name="qwen-rel-whole",
        dataset="prime-rel",
        split="test-0.1",
        chunker="whole-document",
        embeddings="qwen3-embedding-0.6b",
        dimension=1024,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )

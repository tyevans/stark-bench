"""Both repositories' commits must land in every report this run writes.

## The gap this closes

redstring PR #72 changed what `SlidingWindowChunker` emits. Four configs
name `sliding-1000-500` and two of their tenants are live, the larger
holding 549,697 rows written by the pre-fix chunker. **The chunker's name
did not change**, so `config_verbatim` and the ingest report say exactly
what they said before -- and this project's best retrieval-only numbers
describe a corpus the current library would not rebuild.

That is the third time an identifier stayed constant while the thing it
named moved: a model id reused for a different model (ADR 0002), a chat
model whose id read `64k` at every `-np`, and now a chunker name across a
behaviour fix. A commit hash cannot drift from what it identifies.

## Why ingest matters more than run

An ingest decides what text a tenant *holds*. A run against that tenant
can be re-done; the corpus cannot be un-built. So both are asserted, and
the ingest path is not treated as the lesser case.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stark_bench.composition.cli as cli_module
from stark_bench.adapters.source_provenance import source_provenance

_KEYS = (
    "stark_bench_commit",
    "stark_bench_branch",
    "stark_bench_src_dirty",
    "redstring_commit",
    "redstring_branch",
    "redstring_src_dirty",
)


def _cli_source() -> str:
    return Path(cli_module.__file__).read_text(encoding="utf-8")


def _function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(_cli_source())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == name
    )


def _calls_provenance(node: ast.AST) -> bool:
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "source_provenance"
        for inner in ast.walk(node)
    )


def test_the_run_path_records_provenance() -> None:
    assert _calls_provenance(_function("_do_run")), (
        "_do_run never calls source_provenance, so no scored arm records "
        "which code produced it"
    )


def test_the_ingest_path_records_provenance() -> None:
    """The corpus is the thing that cannot be un-built."""
    assert _calls_provenance(_function("_do_ingest")), (
        "_do_ingest never calls source_provenance; an ingest decides what a "
        "tenant holds and a chunker fix changes that without renaming the "
        "chunker"
    )


def test_provenance_reports_both_repositories() -> None:
    """One commit is half an answer: the benchmark and the library both move."""
    recorded = source_provenance()
    assert set(recorded) == set(
        _KEYS
    ), f"source_provenance returned {sorted(recorded)}, expected {sorted(_KEYS)}"


def test_this_checkout_resolves_to_a_real_commit() -> None:
    """A helper that silently returns Nones everywhere records nothing.

    Asserts against the repository the tests are running from, which is by
    construction a git checkout -- so `None` here means the helper is
    broken, not that the environment is unusual.
    """
    recorded = source_provenance()
    commit = recorded["stark_bench_commit"]
    assert isinstance(commit, str) and len(commit) == 40, (
        f"stark_bench_commit is {commit!r}; the tests run from a git "
        "checkout, so this must resolve"
    )


def test_dirtiness_is_scoped_to_library_source() -> None:
    """Scoped to `src/`, or the flag fires constantly and gets ignored.

    `CLAUDE.md` records that dirty session notes and `bench/*.yaml` are
    normal in the redstring checkout and that only `src/` means the arm is
    measuring an uncommitted state.
    """
    source = Path(
        __import__("stark_bench.adapters.source_provenance", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert '"status", "--porcelain", "--", "src"' in source, (
        "the dirty check is not scoped to src/; an unscoped check flags the "
        "routine dirt CLAUDE.md says to ignore"
    )


def test_a_path_that_is_not_a_repository_records_nothing() -> None:
    """Pins the non-zero-exit path: git runs, fails, and we record `None`.

    Note this does NOT reach the `except` clause -- `git -C /nowhere` exits
    non-zero rather than raising, which a deliberate break of that clause
    revealed. The exception path is covered by the test below.
    """
    from stark_bench.adapters.source_provenance import _repo_state

    state = _repo_state(Path("/nonexistent-repository-path"))
    assert state == {"commit": None, "branch": None, "src_dirty": None}


def test_provenance_never_raises_when_git_itself_is_unavailable(monkeypatch) -> None:
    """A run must not die because the git binary is missing.

    The realistic case is a container without git, or a source tarball.
    Provenance is metadata about a run; losing it is a gap in a report,
    while raising here would lose the run itself.
    """
    import subprocess as sp

    from stark_bench.adapters import source_provenance as module

    def _explode(*args: object, **kwargs: object) -> None:
        raise OSError("git: not found")

    monkeypatch.setattr(sp, "run", _explode)
    recorded = module.source_provenance()
    assert set(recorded) == set(_KEYS)
    assert all(
        value is None for value in recorded.values()
    ), f"expected every field unrecorded, got {recorded}"


def test_provenance_is_captured_before_the_run_not_after() -> None:
    """The checkout can move while an arm is in flight, and did.

    On 2026-08-21 redstring `main` gained two commits mid-run. The process
    had already imported the old code, so its behaviour was the old commit
    -- while a probe at report time would have recorded the new one, which
    is the silent-basis error the field exists to prevent.

    Asserts by position: the `source_provenance()` call must appear before
    the `await run(` that executes the query set.
    """
    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    captured = source.index("provenance = source_provenance()")
    executed = source.index("predictions = await run(")
    assert captured < executed, (
        "source_provenance() is called after the query set runs, so a "
        "checkout that moves mid-run records the wrong commit"
    )

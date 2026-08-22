"""Which source produced a number: both repositories, by commit.

## Why this exists

`config_verbatim` records the configuration and `retrieval_is_exact`,
`hnsw_ef_search` and `chat_n_ctx` record the serving basis. None of them
records the **code**, and on 2026-08-21 that became the gap that mattered.

redstring PR #72 changed what `SlidingWindowChunker` emits: a break point
landing at or before the previous chunk's end now falls back to the hard
boundary, and the loop stops once a chunk reaches the end of the text
rather than emitting one final wholly-contained window. Four configs name
`sliding-1000-500` and two of their tenants are live, the larger holding
549,697 rows written by the pre-fix chunker.

The chunker's *name* did not change. `sliding-1000-500` is what an ingest
report stored before the fix and what it stores after -- the identifier
stayed constant while the behaviour moved underneath it, which is the same
shape as a chat model whose id reads `64k` at every `-np`, and as a model
id reused for a different model (ADR 0002). This project has now been bitten
by that pattern three times.

A commit hash cannot drift from what it identifies, which is the whole
argument for recording it rather than a version string.

## Why both repositories

`stark_bench` decides what is measured and redstring decides how. A change
to either moves a number, and this repo is a *consumer* of a path
dependency pointing at a working checkout on whatever branch it happens to
be on -- today's arms were all measured on `perf/indexable-semantic-order`
while `CLAUDE.md` said `main`.

## Why `dirty` is recorded rather than refused

`CLAUDE.md`'s standing instruction is to check the redstring checkout is
clean of *library* source before quoting a number, and that session notes
and `bench/*.yaml` being dirty is normal. Encoding that split here would
put a second policy next to the one in the docs; recording the fact lets a
reader apply the policy they actually hold. `dirty` is scoped to `src/`
for that reason -- a dirty `src/` is the case where the commit hash is a
lie, and nothing else is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str | None:
    """One git command, or `None` for anything that goes wrong.

    Never raises. Provenance is metadata about a run; a run must not fail
    because git was unavailable, the checkout was a tarball, or the binary
    was missing. `None` reads as "not recorded", which is true.
    """
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def _repo_state(repo: Path) -> dict[str, object]:
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "src_dirty": None}
    # Scoped to `src/` deliberately: a dirty working tree elsewhere is
    # routine here (session notes, bench configs, this repo's own
    # pyproject/uv.lock path switch), and flagging it would train everyone
    # to ignore the flag.
    status = _git(repo, "status", "--porcelain", "--", "src")
    return {
        "commit": commit,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "src_dirty": bool(status) if status is not None else None,
    }


def _redstring_root() -> Path | None:
    """The redstring checkout backing this venv, found through the import.

    Located from the installed module rather than from a configured path,
    because the configured path is in `pyproject.toml` and can disagree
    with what is actually importable -- `editable = true` exists in this
    project precisely because a plain path copy went stale once and cost a
    calibration run.
    """
    try:
        import redstring
    except ImportError:
        return None
    package = Path(redstring.__file__).resolve().parent
    for parent in package.parents:
        if (parent / ".git").exists():
            return parent
    return None


def source_provenance() -> dict[str, object]:
    """Commit, branch and `src/` cleanliness for both repositories.

    Every value may be `None`, and a caller must not treat `None` as
    `False`: "not recorded" and "clean" are different facts, and conflating
    them is how a report claims knowledge it does not have.
    """
    bench_root = Path(__file__).resolve().parents[3]
    redstring_root = _redstring_root()
    bench = _repo_state(bench_root)
    library = (
        _repo_state(redstring_root)
        if redstring_root is not None
        else {"commit": None, "branch": None, "src_dirty": None}
    )
    return {
        "stark_bench_commit": bench["commit"],
        "stark_bench_branch": bench["branch"],
        "stark_bench_src_dirty": bench["src_dirty"],
        "redstring_commit": library["commit"],
        "redstring_branch": library["branch"],
        "redstring_src_dirty": library["src_dirty"],
    }

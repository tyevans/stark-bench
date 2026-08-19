"""The gate for the gates: quality checks run on commit, so the hook must exist.

Without `.git/hooks/pre-commit`, every `git commit` succeeds unconditionally
and CI is the first thing to disagree, on a branch already pushed. An absent
hook is indistinguishable from a passing one -- nothing is printed either way.

Two things here are load-bearing and were each learned the hard way:

**It matches `hook-impl`, from pre-commit's generated body.** Matching the
string "pre-commit" would also match git's own `pre-commit.sample`, so
copying the sample into place would pass.

**It asks git where the hooks live rather than assuming `.git/hooks`.** In a
worktree `.git` is a *file* pointing into the parent's `.git/worktrees/`, so
the assumed path does not exist and the test failed while the hook was
installed and working. A false negative here is the mirror of the failure
the test exists to catch, and trains people to ignore it -- which is worse
than not having it. `core.hooksPath` is honoured for the same reason:
pre-commit respects it, so a repo that sets it would otherwise fail here
with the hook correctly installed somewhere else.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _hook_path() -> Path:
    """Where git itself says this repository's hooks live.

    Handles the plain checkout, the worktree, and `core.hooksPath`, because
    all three are legitimate and only the first works by assumption.
    """
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if configured:
        base = Path(configured)
        return base / "pre-commit" if base.is_absolute() else ROOT / base / "pre-commit"

    resolved = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks/pre-commit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    path = Path(resolved)
    return path if path.is_absolute() else ROOT / path


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="CI runs the tools as separate jobs"
)
def test_the_pre_commit_hook_is_installed():
    hook = _hook_path()
    assert hook.exists(), f"run: uv run pre-commit install (looked in {hook})"
    assert "hook-impl" in hook.read_text()


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="needs a real git dir")
def test_the_hook_path_resolves_to_a_real_git_directory():
    """The resolution itself must work, not merely return a string.

    `_hook_path` returning a plausible-but-wrong path is the defect that was
    just fixed, and the assertion above cannot tell "hook missing" from
    "looked in the wrong place" -- both are `exists() is False`.
    """
    hook = _hook_path()
    assert hook.parent.exists(), f"hooks directory does not exist: {hook.parent}"
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    common_path = Path(common) if Path(common).is_absolute() else ROOT / common
    assert (
        hook.parent.resolve() == (common_path / "hooks").resolve()
    ), "hooks resolved outside the repository's common git dir"

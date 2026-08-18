import os
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".git" / "hooks" / "pre-commit"


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="CI runs the tools as separate jobs"
)
def test_the_pre_commit_hook_is_installed():
    """Match on `hook-impl`, from pre-commit's generated body.

    Matching on the string "pre-commit" would also match git's own
    `pre-commit.sample`, so copying the sample into place would pass.
    """
    assert HOOK.exists(), "run: uv run pre-commit install"
    assert "hook-impl" in HOOK.read_text()

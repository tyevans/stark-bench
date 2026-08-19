"""Every module named in the import contracts must be a real module.

`import-linter` accepts a `forbidden_modules` entry naming a package that
does not exist, and says nothing. So does an `exhaustive_ignores` entry.
Both then sit there forever, and the day someone re-creates a package with
that name they get a rule they never agreed to -- or, more often, the list
simply drifts into fiction and stops being reviewable.

This is the same failure this project has already hit with ruff and mypy
per-file ignores: a per-file exemption that matches no file passes
silently, so a shrinking-exemption ratchet stops shrinking and nobody is
told. The lesson recorded then was that **every exemption list needs a test
that its entries still match something**. This is that test for the import
contracts.

Caught one on its first run: `stark_bench.tools` was still in the `agents`
forbidden list one commit after the package was moved into `adapters`.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _config() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["tool"]["importlinter"]


def _module_exists(dotted: str) -> bool:
    """True if `dotted` is importable *as a name*, without importing it.

    `find_spec` imports parent packages but not the module itself, which is
    what we want: this must not execute an adapter's module body, and an
    adapter that fails to import for an unrelated reason must not be
    reported as missing.
    """
    try:
        spec = importlib.util.find_spec(dotted)
    except ModuleNotFoundError:
        return False
    if spec is None:
        return False
    # A *namespace* package -- a directory with no `__init__.py` -- has a
    # spec with `origin is None`, and `find_spec` happily returns one. That
    # is not a module this project defines, and it is exactly how the first
    # version of this test was inert: deleting `tools/` left a
    # `__pycache__` directory behind, Python read the surviving directory as
    # a namespace package, and the check reported the deleted package as
    # present. Every package here has an `__init__.py`.
    return spec.origin is not None


def _forbidden_entries() -> list[tuple[str, str]]:
    return [
        (contract.get("name", "<unnamed>"), module)
        for contract in _config().get("contracts", [])
        for key in ("source_modules", "forbidden_modules")
        for module in contract.get(key, [])
    ]


@pytest.mark.parametrize(
    ("contract", "module"),
    _forbidden_entries(),
    ids=lambda value: value.replace(".", "_").replace(" ", "-"),
)
def test_every_module_named_in_a_contract_exists(contract: str, module: str) -> None:
    assert _module_exists(module), (
        f"contract {contract!r} names {module!r}, which does not exist. "
        f"A rule about a module that is gone is not enforcing anything; "
        f"delete the entry, or restore the module."
    )


def test_every_exhaustive_ignore_exists() -> None:
    """An ignore for a package that is gone is a finished slice nobody closed.

    This list is the definition of done for the hexagonal refactor, so an
    entry that no longer names anything makes the remaining work look
    larger than it is.
    """
    missing = [
        name
        for contract in _config().get("contracts", [])
        for name in contract.get("exhaustive_ignores", [])
        if not _module_exists(f"stark_bench.{name}")
    ]
    assert not missing, (
        f"exhaustive_ignores names packages that no longer exist: {missing}. "
        f"Remove them -- the list is what says how much refactor is left."
    )

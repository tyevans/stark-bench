"""A third-party client belongs in one directory, and that is checked.

import-linter sees first-party imports only, so it cannot notice `asyncpg`
appearing in the composition root or `neo4j` in an agent. This is the gate
for that, and it is deliberately a *table*: a new client adds a row in the
same commit, because a rule that holds only because nobody has broken it is
indistinguishable from no rule.

Guarded in both directions. A row naming a directory that has stopped
importing its library fails too -- otherwise the table rots into a list of
things that used to be true, and the check passes forever while protecting
nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "stark_bench"

#: library prefix -> the one directory (relative to `stark_bench`) allowed
#: to import it.
CONFINED = {
    # Reaching past redstring's `ChunkStore` for a bulk-id query, knowingly
    # and in one place. See `adapters/postgres_chunk_index.py`.
    "asyncpg": "adapters",
    # `stark_qa` pulls `ogb` -> `rdkit`, which crashes on NumPy 2.x, so it
    # can only ever run in the 3.11 subprocess. An import anywhere else
    # breaks `import stark_bench` for everyone.
    "stark_qa": "sidecar",
    # Comes in with `stark_qa`; same reasoning, and heavier.
    "torch": "sidecar",
}

#: Libraries that are NOT confined and why, so nobody adds a row that then
#: has to be deleted. `neo4j` was listed here once and the reverse check
#: caught it on its first run: nothing under `src/` imports it, because
#: `Neo4jGraphStore` comes from redstring. A row for a library we do not
#: import is a rule that cannot fail.
UNCONFINED_ON_PURPOSE = {
    "numpy": "used by both skb/ and sidecar/ -- an array is not a client",
    "redstring": "the library under test; every layer that stores may use it",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _modules_importing(library: str) -> set[str]:
    """Top-level package under `stark_bench` for each module importing it."""
    owners = set()
    for path in SRC.rglob("*.py"):
        if library in _imports(path):
            owners.add(path.relative_to(SRC).parts[0])
    return owners


@pytest.mark.parametrize(("library", "directory"), sorted(CONFINED.items()))
def test_a_library_is_imported_only_from_its_directory(library, directory):
    leaked = _modules_importing(library) - {directory}
    assert not leaked, (
        f"{library} is imported from {sorted(leaked)}, but is confined to "
        f"{directory}/. Move the import, or add a row if the confinement changed."
    )


@pytest.mark.parametrize(("library", "directory"), sorted(CONFINED.items()))
def test_the_row_still_describes_something_real(library, directory):
    """A row for a library nobody imports any more is dead weight that passes.

    This is the direction that rots silently: the forward check keeps
    passing while the table drifts into a description of the past.
    """
    owners = _modules_importing(library)
    assert owners, (
        f"nothing under src/ imports {library} any more -- delete its row "
        "rather than leaving a rule that cannot fail"
    )
    assert (
        directory in owners
    ), f"{library} is confined to {directory}/, but nothing there imports it"

"""Redstring's production adapters are not in `redstring.__all__`.

We import them by dotted path against a pinned version. This test turns a
version bump that moves them into a loud failure here rather than a confusing
one during a two-hour ingest.
"""

import importlib

import pytest

PATHS = [
    ("redstring.vector.adapters.pgvector", "PgVectorStore"),
    ("redstring.graph.adapters.neo4j", "Neo4jGraphStore"),
    ("redstring.chunks.adapters.postgres", "PostgresChunkStore"),
]


@pytest.mark.parametrize(("module", "name"), PATHS)
def test_adapter_path_still_resolves(module, name):
    mod = importlib.import_module(module)
    assert hasattr(
        mod, name
    ), f"{module}.{name} moved; the redstring pin needs revisiting"

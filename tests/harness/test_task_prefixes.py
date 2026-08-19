"""The task prefix must reach the server, and must not share a vector space.

Both halves are silent when wrong. A corpus embedded without its prefix
produces well-formed vectors that cluster sensibly and score plausibly -- the
only symptom is a retrieval number below what the model can do, which reads as
"this model is mediocre" and is exactly the conclusion this project nearly
drew. Two corpora embedded with *different* prefixes into one table is worse:
cosine between them is meaningless and nothing raises.
"""

from __future__ import annotations

import pytest

from stark_bench.harness.cli import _table_for
from stark_bench.harness.config import RunConfig


def _config(**overrides: object) -> RunConfig:
    """A config built field-by-field, never through a helper that fills all of them.

    `RunConfig`'s own defaults are what `document_prefix` and `query_prefix`
    are being checked against in the unprefixed cases, so a factory passing
    every field would make those cases vacuous.
    """
    base = dict(
        name="t",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="nomic-embed-text",
        dimension=768,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )
    base.update(overrides)
    return RunConfig(**base)  # type: ignore[arg-type]


def test_prefixes_default_to_empty_on_the_type_itself():
    """Constructed directly, not through a factory -- ada-002 relies on these."""
    config = RunConfig(
        name="t",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="precomputed-ada002",
        dimension=1536,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )
    assert config.document_prefix == ""
    assert config.query_prefix == ""


def test_an_unprefixed_table_name_is_unchanged():
    """The ada-002 rows already in Postgres were written against this name.

    Appending a digest unconditionally would orphan every one of them, which
    is the failure that already cost this project a full run.
    """
    assert (
        _table_for(_config(embeddings="precomputed-ada002"))
        == "kg_chunks_precomputed_ada002"
    )


def test_a_prefix_moves_the_corpus_to_a_different_table():
    bare = _table_for(_config())
    prefixed = _table_for(
        _config(document_prefix="search_document: ", query_prefix="search_query: ")
    )
    assert bare != prefixed


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The concatenation trap: without a separator, ("ab", "") and
        # ("a", "b") hash identically and two incomparable corpora share a
        # table. This pair is the whole reason the identity string is
        # NUL-joined, and it is the only input that can see the difference.
        (("ab", ""), ("a", "b")),
        # Which side the prefix sits on is a real distinction: document-only
        # and query-only are different models' conventions (nomic vs BGE).
        (("search_document: ", ""), ("", "search_document: ")),
        # And the prefix text itself has to matter, not merely its presence.
        (("search_document: ", "search_query: "), ("passage: ", "query: ")),
    ],
)
def test_distinct_prefix_pairs_never_collide(left, right):
    assert _table_for(
        _config(document_prefix=left[0], query_prefix=left[1])
    ) != _table_for(_config(document_prefix=right[0], query_prefix=right[1]))


def test_the_same_prefix_pair_is_stable_across_calls():
    """A digest that moved between runs would orphan the previous ingest."""
    args = {"document_prefix": "search_document: ", "query_prefix": "search_query: "}
    assert _table_for(_config(**args)) == _table_for(_config(**args))


def test_the_native_configs_actually_state_the_nomic_prefixes():
    """The wiring is worthless if no config seats it -- this is the gate on that.

    Reads the shipped YAML rather than a fixture: the defect being prevented
    is a config file that forgot, and a fixture cannot forget.
    """
    from pathlib import Path

    from stark_bench.harness.config import load_config

    root = Path(__file__).resolve().parents[2] / "config"
    native = [
        p
        for p in sorted(root.glob("*.yaml"))
        if load_config(p).embeddings == "nomic-embed-text"
    ]
    assert native, "no nomic config found -- this test would pass vacuously"
    for path in native:
        config = load_config(path)
        assert config.document_prefix == "search_document: ", path.name
        assert config.query_prefix == "search_query: ", path.name


def test_the_live_provider_is_built_with_the_configured_prefixes(monkeypatch):
    """The wiring test, and it exists because its absence was demonstrated.

    Deleting both `*_prefix=` arguments from `_live_embeddings_for` -- which
    is precisely the original defect, restored by hand -- left the other 39
    harness tests green. `_table_for` reacting to a prefix proves only that
    the config field is read somewhere, not that a single byte of it reaches
    the embedding server.

    Captures the call rather than asserting on a private attribute of the
    provider: what this module is responsible for is *passing* the prefixes.
    Whether the provider then applies them to the right side is redstring's
    own compliance suite, which runs against a live endpoint.
    """
    from stark_bench.harness import cli

    seen: dict[str, object] = {}

    def _capture(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli.LangChainEmbeddingProvider, "openai_compatible", _capture)
    cli._live_embeddings_for(
        _config(document_prefix="search_document: ", query_prefix="search_query: ")
    )

    # Asserted separately and with *different* values, because a single
    # `prefix` threaded to both sides -- or the two arguments swapped -- is a
    # real way to get this wrong, and equal prefixes could not tell either
    # from correct.
    assert seen["document_prefix"] == "search_document: "
    assert seen["query_prefix"] == "search_query: "

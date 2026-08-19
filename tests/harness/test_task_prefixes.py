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
from stark_bench.domain.run_config import RunConfig


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
        embeddings="Nemotron-3-Embed-1B",
        dimension=2048,
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
    prefixed = _table_for(_config(document_prefix="passage: ", query_prefix="query: "))
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
        # and query-only are different models' conventions (Nemotron puts a
        # prefix on both sides; BGE puts one only on the query).
        (("passage: ", ""), ("", "passage: ")),
        # And the prefix text itself has to matter, not merely its presence.
        (("passage: ", "query: "), ("search_document: ", "search_query: ")),
    ],
)
def test_distinct_prefix_pairs_never_collide(left, right):
    assert _table_for(
        _config(document_prefix=left[0], query_prefix=left[1])
    ) != _table_for(_config(document_prefix=right[0], query_prefix=right[1]))


def test_the_same_prefix_pair_is_stable_across_calls():
    """A digest that moved between runs would orphan the previous ingest."""
    args = {"document_prefix": "passage: ", "query_prefix": "query: "}
    assert _table_for(_config(**args)) == _table_for(_config(**args))


def test_the_native_configs_actually_state_the_models_prefixes():
    """The wiring is worthless if no config seats it -- this is the gate on that.

    Reads the shipped YAML rather than a fixture: the defect being prevented
    is a config file that forgot, and a fixture cannot forget.
    """
    from pathlib import Path

    from stark_bench.adapters.config_file import load_config

    root = Path(__file__).resolve().parents[2] / "config"
    native = [
        p
        for p in sorted(root.glob("*.yaml"))
        if load_config(p).embeddings == "Nemotron-3-Embed-1B"
    ]
    assert native, "no live-embedding config found -- this test would pass vacuously"
    for path in native:
        config = load_config(path)
        assert config.document_prefix == "passage: ", path.name
        assert config.query_prefix == "query: ", path.name


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
        _config(document_prefix="passage: ", query_prefix="query: ")
    )

    # Asserted separately and with *different* values, because a single
    # `prefix` threaded to both sides -- or the two arguments swapped -- is a
    # real way to get this wrong, and equal prefixes could not tell either
    # from correct.
    assert seen["document_prefix"] == "passage: "
    assert seen["query_prefix"] == "query: "


def test_every_shipped_config_yields_a_legal_postgres_identifier():
    """A model id is a vendor's string; a table name is not.

    `Nemotron-3-Embed-1B` carries capitals and redstring's chunk store
    rejects a table that is not a bare lowercase identifier -- which is how
    this was found, at the start of an ingest rather than in review. The
    check runs over the shipped configs rather than invented names because
    the defect is a real model id arriving, and no invented name would have
    had capitals in it.
    """
    import re
    from pathlib import Path

    from stark_bench.adapters.config_file import load_config

    root = Path(__file__).resolve().parents[2] / "config"
    paths = sorted(root.glob("*.yaml"))
    assert paths, "no configs found -- this test would pass vacuously"
    for path in paths:
        table = _table_for(load_config(path))
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", table), f"{path.name}: {table!r}"

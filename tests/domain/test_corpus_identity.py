"""`CorpusIdentity` decides which vectors may be compared, so it is pinned.

The table names below are LITERAL on purpose. Deriving an expectation from
the implementation would make every test pass for any digest function,
including one that changes between releases -- and a changed digest orphans
every row a previous ingest wrote, which has already cost this project one
full corpus.
"""

from __future__ import annotations

import re

import pytest

from stark_bench.domain import CorpusIdentity

NEMOTRON = CorpusIdentity("Nemotron-3-Embed-1B", 2048, "passage: ", "query: ")
ADA = CorpusIdentity("precomputed-ada002", 1536)


def test_the_live_table_names_are_exactly_what_is_on_disk():
    """These two strings name tables that currently hold rows.

    If this test fails, the change under it orphans a corpus. That is not a
    hypothetical: the digest changed once during this refactor, mid-ingest,
    and only printing the name caught it.
    """
    assert NEMOTRON.table_name() == "kg_chunks_nemotron_3_embed_1b_d38d8f8b"
    assert ADA.table_name() == "kg_chunks_precomputed_ada002"


def test_an_unprefixed_identity_gets_no_digest_suffix():
    """ada-002's rows predate the digest and must keep their table."""
    assert ADA.table_name() == "kg_chunks_precomputed_ada002"
    assert not ADA.is_prefixed


def test_dimension_changes_identity_but_not_the_table_name():
    """Documented asymmetry, so a reader does not "fix" it.

    Dimension is in equality because two widths cannot share a store. It is
    out of the digest because putting it in renames live tables, and it
    cannot disagree in practice -- width is a property of the model.
    """
    other = CorpusIdentity("Nemotron-3-Embed-1B", 1024, "passage: ", "query: ")
    assert other != NEMOTRON
    assert not other.comparable_with(NEMOTRON)
    assert other.table_name() == NEMOTRON.table_name()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The concatenation trap: with no separator, ("ab", "") and
        # ("a", "b") hash identically and two incomparable corpora share a
        # table. This pair is the only input that can see it.
        (("ab", ""), ("a", "b")),
        # Which side the prefix sits on is a real distinction between model
        # families -- Nemotron prefixes both, BGE only the query.
        (("passage: ", ""), ("", "passage: ")),
        # And the text has to matter, not merely its presence.
        (("passage: ", "query: "), ("search_document: ", "search_query: ")),
    ],
)
def test_distinct_prefix_pairs_never_share_a_table(left, right):
    a = CorpusIdentity("m", 8, *left)
    b = CorpusIdentity("m", 8, *right)
    assert a.table_name() != b.table_name()
    assert not a.comparable_with(b)


def test_every_table_name_is_a_legal_postgres_identifier():
    """A model id is a vendor's string; a table name is not.

    `Nemotron-3-Embed-1B` has capitals, and redstring's chunk store rejects
    anything but a bare lowercase identifier -- found at the start of an
    ingest rather than in review.
    """
    for identity in (
        NEMOTRON,
        ADA,
        CorpusIdentity("Weird/Model.v2 (b)", 8, "p: ", "q: "),
    ):
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", identity.table_name()), identity


def test_the_digest_is_stable_across_calls():
    """A digest that moved between runs would orphan the previous ingest."""
    assert NEMOTRON.digest() == NEMOTRON.digest()
    assert (
        CorpusIdentity("m", 8, "a", "b").digest()
        == CorpusIdentity("m", 8, "a", "b").digest()
    )


def test_an_identity_is_hashable_and_frozen():
    """It keys dicts and must not drift after a store is opened against it."""
    assert len({NEMOTRON, ADA, NEMOTRON}) == 2
    with pytest.raises(Exception):
        NEMOTRON.model = "other"  # type: ignore[misc]


@pytest.mark.parametrize(("model", "dimension"), [("", 8), ("m", 0), ("m", -1)])
def test_a_meaningless_identity_is_refused_at_construction(model, dimension):
    with pytest.raises(ValueError):
        CorpusIdentity(model, dimension)

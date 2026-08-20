"""At `per_type=1`, *which* neighbour name survives is the whole signal.

`first_relations` keeps the first name in document order, which is
arbitrary. This picks by BM25 against the query. Measured on real data, the
two disagree on 48.2% of candidates -- so this is not a refinement, it is a
different arm.

The selector it replaces was `name.lower() in query.lower()`: all-or-nothing,
unable to rank among several matches, and unable to match `HLA-DRB1` against
the name `HLA-DRB1 allele`.
"""

from __future__ import annotations

import pytest

from stark_bench.agents.rerank import (
    RerankAgent,
    first_relations,
    rank_names_lexically,
    ranked_relations,
)

DOC = (
    "- name: DCC mediated attractive signaling\n"
    "- type: pathway\n"
    "- relations:\n"
    "  interacts_with: {gene/protein: (RAC1, DCC, NTN1)}\n"
    "  member_of: {pathway: (Axon guidance, Metabolism)}\n"
)


def test_the_query_named_entity_wins() -> None:
    assert rank_names_lexically(
        "about DCC signalling", ["RAC1", "DCC", "NTN1"], top=1
    ) == ["DCC"]


def test_it_beats_the_substring_test_on_partial_names() -> None:
    """`"HLA-DRB1 allele".lower() in query.lower()` is False; BM25 is not
    fooled, because the tokens match."""
    got = rank_names_lexically(
        "is HLA-DRB1 implicated", ["TP53", "HLA-DRB1 allele"], top=1
    )
    assert got == ["HLA-DRB1 allele"]


def test_it_ranks_among_several_matches() -> None:
    """The substring test cannot: both names match, it cannot say which
    matches better. At top=1 that is the entire decision."""
    got = rank_names_lexically(
        "BRCA1 breast cancer", ["cancer", "BRCA1 breast cancer"], top=1
    )
    assert got == ["BRCA1 breast cancer"]


def test_a_name_in_every_candidate_does_not_outrank_a_rare_one() -> None:
    """idf is the point. A name shared by everything distinguishes nothing."""
    names = ["common"] * 8 + ["rare"]
    assert rank_names_lexically("common rare", names, top=1) == ["rare"]


def test_no_query_overlap_falls_back_to_document_order() -> None:
    """Not to an arbitrary reshuffle: with no signal, match `first_relations`
    so a null result is a null and not noise."""
    assert rank_names_lexically("zzz", ["A", "B", "C"], top=2) == ["A", "B"]


def test_an_empty_query_falls_back_rather_than_dividing_by_zero() -> None:
    assert rank_names_lexically("", ["A", "B"], top=1) == ["A"]


def test_no_names_is_empty_not_an_error() -> None:
    assert rank_names_lexically("q", [], top=1) == []


def test_top_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="top"):
        rank_names_lexically("q", ["A"], top=0)


def test_ranked_relations_changes_the_kept_name() -> None:
    """The end-to-end claim, on the shape the prompt actually sees."""
    assert "RAC1" in first_relations(DOC)
    assert "DCC" in ranked_relations(DOC, "DCC mediated attractive signaling")


def test_ranked_relations_keeps_the_same_shape_as_first_relations() -> None:
    """Only *which* names survive may differ, or the arms are not
    comparable and the prompt cost moves too."""
    a = first_relations(DOC)
    b = ranked_relations(DOC, "unrelated words")
    assert a == b


def test_ranked_relations_is_empty_without_a_block() -> None:
    assert ranked_relations("- name: X\n", "q") == ""


def test_the_agent_dispatches_the_ranked_mode() -> None:
    agent = RerankAgent(passage_mode="title_rel_ranked")
    out = agent._render_passage(DOC, "DCC mediated attractive signaling")
    assert out.startswith("DCC mediated attractive signaling (pathway) | ")
    assert "DCC" in out.split("|", 1)[1]


def test_the_registry_exposes_the_ranked_arm() -> None:
    from stark_bench.composition.agent_registry import AGENTS

    assert "rerank40titlerelranked" in AGENTS

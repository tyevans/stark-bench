"""Cutting the reranker's token bill must not cut the signal it exists to weigh.

Both knobs here trade tokens for information, and the information at risk is
the one this project's headline result rests on: PRIME queries name related
entities verbatim, and the relations block is why hybrid moved +42%. A cap
that dropped the named neighbour would buy speed with the finding.
"""

from __future__ import annotations

import pytest

from stark_bench.agents.rerank import (
    RerankAgent,
    TerseRelevance,
    TerseRelevances,
    lean_document,
)
from stark_bench.domain import Passage, Query

HUB = (
    "- name: UBC\n"
    "- type: gene/protein\n"
    "- details:\n"
    "  - summary: a gene\n"
    "- relations:\n"
    "  ppi: {gene/protein: (AAA, BBB, CCC, DDD, EEE, TARGETX)}\n"
    "  indication: {disease: (ASTHMA, GOUT)}\n"
)


def test_a_short_list_is_left_exactly_alone():
    """Catches: annotating or reordering lists that fit, which costs tokens."""
    out = lean_document(HUB, "unrelated", cap=10)

    assert "(AAA, BBB, CCC, DDD, EEE, TARGETX)" in out
    assert "+" not in out.split("- relations:")[1]


def test_a_long_list_is_capped_and_says_how_much_it_dropped():
    """Catches: silent truncation.

    A model shown 2 of 500 neighbours cannot distinguish a hub from a leaf
    unless told, and that difference is evidence about the candidate.
    """
    out = lean_document(HUB, "unrelated", cap=2)

    assert "(AAA, BBB, +4 more)" in out
    assert "(ASTHMA, GOUT)" in out, "every relation TYPE must survive the cap"


def test_a_neighbour_the_query_names_survives_the_cap():
    """Catches: capping by position alone.

    `TARGETX` is last of six. At cap=2 a positional cut drops it -- and it is
    the only token in the document that answers this query.
    """
    out = lean_document(HUB, "a gene that interacts with TARGETX", cap=2)

    assert "TARGETX" in out
    assert "+4 more" in out


def test_the_head_of_the_document_is_never_touched():
    """Catches: a character budget applied before the relations marker.

    Name, type and details are the candidate's identity; STaRK puts
    `- relations:` near the top, so everything above it is small and always
    worth its tokens.
    """
    out = lean_document(HUB, "x", cap=1)

    assert out.startswith("- name: UBC\n- type: gene/protein")
    assert "summary: a gene" in out


def test_a_document_without_relations_is_returned_unchanged():
    plain = "- name: X\n- type: drug\n"

    assert lean_document(plain, "x", cap=5) == plain


def test_an_unrecognised_relation_line_passes_through_untouched():
    """Catches: a regex that mangles what it cannot parse.

    Costing tokens is recoverable; corrupting a candidate's text is not.
    """
    odd = "- relations:\n  weird line without parens\n"

    assert lean_document(odd, "x", cap=1) == odd


def test_capping_to_zero_is_refused():
    """Catches: a 61%-saving setting that deletes the corpus's whole point."""
    with pytest.raises(ValueError, match="removes the relations signal"):
        lean_document(HUB, "x", cap=0)


def test_the_cap_is_deterministic():
    """Two runs of one arm must agree; every accuracy number here is a diff."""
    a = lean_document(HUB, "TARGETX and GOUT", cap=3)
    b = lean_document(HUB, "TARGETX and GOUT", cap=3)

    assert a == b


class FakeToolset:
    """Captures the prompt and returns a scripted judgement."""

    def __init__(self, passages, judged):
        self._passages = passages
        self._judged = judged
        self.prompt = None

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):
        self.prompt = prompt
        return self._judged


def _passages(n=3):
    return [
        Passage(node_id=str(100 + i), text=HUB, score=1.0 / (i + 1)) for i in range(n)
    ]


async def test_terse_mode_labels_candidates_by_index_not_node_id():
    """Catches: paying ~3 tokens per candidate for an id the model echoes."""
    tools = FakeToolset(_passages(), TerseRelevances(scores=[]))
    agent = RerankAgent(k=3, fetch=3, terse_scores=True)

    await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert "[1] " in tools.prompt and "[3] " in tools.prompt
    assert "[100]" not in tools.prompt


async def test_an_index_is_mapped_back_to_the_right_node():
    """Catches: an off-by-one between label and passage -- the whole risk.

    Index 2 must score node 101, not 100 or 102. A shift here would rerank
    every candidate against its neighbour's text and look entirely normal.
    """
    tools = FakeToolset(
        _passages(), TerseRelevances(scores=[TerseRelevance(i=2, s=99)])
    )
    agent = RerankAgent(k=3, fetch=3, terse_scores=True)

    ranked = await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert ranked[0].node_id == "101"


async def test_an_out_of_range_index_is_dropped_not_clamped():
    """Catches: `passages[i-1]` on an invented index, which wraps on i=0.

    Python's negative indexing turns index 0 into the LAST candidate -- a
    silently mis-scored one rather than an unscored one.
    """
    tools = FakeToolset(
        _passages(),
        TerseRelevances(scores=[TerseRelevance(i=0, s=99), TerseRelevance(i=9, s=98)]),
    )
    agent = RerankAgent(k=3, fetch=3, terse_scores=True)

    ranked = await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert [r.node_id for r in ranked] == [
        "100",
        "101",
        "102",
    ], "no score survived, so retrieval order must be preserved exactly"


async def test_a_duplicate_index_keeps_the_first_score():
    tools = FakeToolset(
        _passages(),
        TerseRelevances(scores=[TerseRelevance(i=3, s=90), TerseRelevance(i=3, s=10)]),
    )
    agent = RerankAgent(k=3, fetch=3, terse_scores=True)

    ranked = await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert ranked[0].node_id == "102"


async def test_the_relation_cap_reaches_the_prompt():
    """Catches: the classic -- helper correct, nobody calls it."""
    tools = FakeToolset(_passages(1), TerseRelevances(scores=[]))
    agent = RerankAgent(k=1, fetch=1, relation_cap=2, terse_scores=True)

    await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert "+4 more" in tools.prompt


async def test_the_default_arm_is_completely_unchanged():
    """Catches: turning an optimisation on by default.

    `rerank40`'s 0.46323 was measured with node-id labels, float scores and
    uncapped relations. If this arm quietly acquired the new behaviour, that
    number would stop being reproducible with no file recording the change.
    """
    from stark_bench.agents.rerank import Relevance, Relevances

    tools = FakeToolset(
        _passages(), Relevances(scores=[Relevance(node_id="101", score=99.0)])
    )
    agent = RerankAgent(k=3, fetch=3)

    ranked = await agent.retrieve(Query(query_id=1, text="q"), tools)

    assert "[100]" in tools.prompt, "default mode labels by node id"
    assert "+" not in tools.prompt.split("- relations:")[1].split("\n\n")[0]
    assert ranked[0].node_id == "101"


async def test_the_instruction_stays_at_the_very_front_of_the_prompt():
    """Catches: anything query-specific migrating above the invariant text.

    The instruction is the only span two consecutive rerank prompts share --
    98.2% of query pairs share zero candidates, measured on this corpus -- so
    it is the only span a server-side prefix cache can ever reuse. Small, but
    it is free, and it is free only while it is first.
    """
    tools = FakeToolset(_passages(), TerseRelevances(scores=[]))
    agent = RerankAgent(k=3, fetch=3, terse_scores=True)

    await agent.retrieve(Query(query_id=1, text="UNIQUE-QUERY-TOKEN"), tools)

    assert tools.prompt.startswith("You are ranking candidate entities")
    assert tools.prompt.index("UNIQUE-QUERY-TOKEN") > 400

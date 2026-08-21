"""Matrix scoring: one row per candidate, averaged across dimensions.

Two motivations, one measured and one structural.

**Measured.** The model quantises hard onto a few integers -- one observed
response used 5 nine times, 8 six times and 10 six times across 40
candidates. Ties break on retrieval order, so 10% of queries carry a run of
>=10 candidates ordered by hybrid rather than by the reranker.

**Structural.** PRIME's queries are conjunctive: "a drug that targets X and
is indicated for Y". A single score collapses two judgements into one
number; separate dimensions ask for them apart.

The premise is that the dimensions are ORTHOGONAL, and that is exactly what
could quietly fail -- a model that writes the same number three times has
tripled the decode bill for nothing while every test below still passes. So
the degenerate share is counted and logged.
"""

from __future__ import annotations

import logging

import pytest

from stark_bench.agents.rerank import (
    _SCORE_DIMENSIONS,
    MatrixRelevances,
    RerankAgent,
)
from stark_bench.domain import Passage, Query

DOC = "- name: E{}\n- type: gene/protein\n"


class FakeToolset:
    def __init__(self, passages, judged):
        self._passages = passages
        self._judged = judged
        self.schema = None
        self.prompt = None

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):
        self.schema, self.prompt = schema, prompt
        return self._judged


def _passages(n=3):
    return [
        Passage(node_id=str(100 + i), text=DOC.format(i), score=1.0 / (i + 1))
        for i in range(n)
    ]


async def _run(judged, n=3):
    tools = FakeToolset(_passages(n), judged)
    agent = RerankAgent(k=n, fetch=n, matrix_scores=True, passage_mode="title")
    ranked = await agent.retrieve(Query(query_id=1, text="q"), tools)
    return ranked, tools


async def test_the_matrix_schema_is_requested() -> None:
    """Catches the field being set while the old schema still goes out."""
    _, tools = await _run(MatrixRelevances(scores=[[1, 10, 10, 10]]))
    assert tools.schema is MatrixRelevances


async def test_matrix_wins_over_pair_when_both_are_set() -> None:
    tools = FakeToolset(_passages(), MatrixRelevances(scores=[[1, 5, 5, 5]]))
    agent = RerankAgent(
        k=3, fetch=3, pair_scores=True, matrix_scores=True, passage_mode="title"
    )
    await agent.retrieve(Query(query_id=1, text="q"), tools)
    assert tools.schema is MatrixRelevances


async def test_the_dimensions_are_averaged() -> None:
    """90/0/0 must not outrank 40/40/40: mean 30 against 40."""
    ranked, _ = await _run(MatrixRelevances(scores=[[1, 90, 0, 0], [2, 40, 40, 40]]))
    assert ranked[0].node_id == "101"


async def test_the_average_breaks_a_tie_a_single_score_could_not() -> None:
    """The measured defect: identical first scores, resolved by the rest."""
    ranked, _ = await _run(MatrixRelevances(scores=[[1, 50, 10, 10], [2, 50, 90, 90]]))
    assert ranked[0].node_id == "101"


async def test_a_row_of_the_wrong_width_is_dropped() -> None:
    ranked, _ = await _run(MatrixRelevances(scores=[[1, 90, 90], [2, 10, 10, 10]]))
    assert ranked[0].node_id == "101"


async def test_an_out_of_range_index_is_dropped_not_wrapped() -> None:
    ranked, _ = await _run(MatrixRelevances(scores=[[0, 99, 99, 99], [2, 80, 80, 80]]))
    assert ranked[0].node_id == "101"


async def test_scores_are_clamped_not_dropped() -> None:
    ranked, _ = await _run(
        MatrixRelevances(scores=[[1, 500, 500, 500], [2, 10, 10, 10]])
    )
    assert ranked[0].node_id == "100"


async def test_the_prompt_names_every_dimension() -> None:
    """A schema constrains shape; only the prompt says what to put in it --
    the defect that made an earlier arm return empty scores."""
    _, tools = await _run(MatrixRelevances(scores=[[1, 1, 1, 1]]))
    for name, _how in _SCORE_DIMENSIONS:
        assert name in tools.prompt


async def test_the_prompt_asks_for_the_scores_to_differ() -> None:
    """Without this the model has no reason not to write one number three
    times, which is the failure mode that makes the encoding pointless.

    Asserted on a phrase unique to this instruction. A substring test for
    "differ" passed while the instruction was deleted, because the base
    template already says "give close candidates different scores"."""
    _, tools = await _run(MatrixRelevances(scores=[[1, 1, 1, 1]]))
    assert "SEPARATE judgements" in tools.prompt
    assert "often differ from each other" in tools.prompt


async def test_degenerate_rows_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tripling the decode bill for one number must not be silent."""
    with caplog.at_level(logging.WARNING):
        await _run(MatrixRelevances(scores=[[1, 7, 7, 7], [2, 9, 9, 9], [3, 4, 4, 4]]))
    assert any("orthogonally" in r.getMessage() for r in caplog.records)


async def test_genuinely_varied_rows_log_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning on every query is a warning nobody reads."""
    with caplog.at_level(logging.WARNING):
        await _run(
            MatrixRelevances(scores=[[1, 90, 10, 50], [2, 10, 80, 20], [3, 5, 5, 60]])
        )
    assert not [r for r in caplog.records if "orthogonally" in r.getMessage()]


def test_the_row_width_follows_the_dimension_list() -> None:
    """Adding a dimension must not silently start dropping every row."""
    assert len(_SCORE_DIMENSIONS) >= 2


def test_the_registry_exposes_the_matrix_arm() -> None:
    from stark_bench.composition.agent_registry import AGENTS

    assert "rerank40titlerelmatrix" in AGENTS

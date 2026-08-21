"""The prompt must say what to return, and an empty answer must be loud.

Found live on 2026-08-20. With `pair_scores` the schema is a bare
`list[list[int]]`, and the prompt still carried the `node_id`-era sentence
"Return one score for every candidate id, and invent no ids" -- naming ids
the model could no longer see. It responded `{"scores": []}` in 1.55s with
no decode at all.

That is the worst failure this agent has, because it is not a failure
anywhere visible: empty scores fall through to retrieval order, which scores
*exactly* `hybrid`, and `run_queries` logs `0 empty` because a full ranking
was returned.

A JSON schema constrains shape. It does not say what to put in the shape.
"""

from __future__ import annotations

import logging

import pytest

from stark_bench.agents.rerank import (
    PairRelevances,
    RerankAgent,
    TerseRelevances,
    _OUTPUT_INSTRUCTION,
)
from stark_bench.domain import Passage, Query


class FakeToolset:
    def __init__(self, judged, n=3):
        self._passages = [
            Passage(node_id=str(100 + i), text=f"- name: E{i}\n- type: t\n", score=1.0)
            for i in range(n)
        ]
        self._judged = judged
        self.prompt = None

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):
        self.prompt = prompt
        return self._judged


async def _run(agent, judged):
    tools = FakeToolset(judged)
    ranked = await agent.retrieve(Query(query_id=7, text="q"), tools)
    return ranked, tools


async def test_pair_mode_asks_for_pairs_not_ids() -> None:
    """The exact defect: an instruction naming ids the model cannot see."""
    _, tools = await _run(
        RerankAgent(k=3, fetch=3, pair_scores=True, passage_mode="title"),
        PairRelevances(scores=[[1, 10]]),
    )
    assert "[index, score]" in tools.prompt
    assert "invent no ids" not in tools.prompt


async def test_terse_mode_asks_for_objects() -> None:
    _, tools = await _run(
        RerankAgent(k=3, fetch=3, terse_scores=True, passage_mode="title"),
        TerseRelevances(scores=[]),
    )
    assert '{"i": index, "s": score}' in tools.prompt


async def test_every_mode_forbids_an_empty_list_in_words() -> None:
    for mode in ("pairs", "terse"):
        assert "empty list" in _OUTPUT_INSTRUCTION[mode]


async def test_the_instruction_survives_template_formatting() -> None:
    """Doubled braces would reach the model literally: the instruction is
    substituted into the template, not formatted itself."""
    _, tools = await _run(
        RerankAgent(k=3, fetch=3, terse_scores=True, passage_mode="title"),
        TerseRelevances(scores=[]),
    )
    assert "{{" not in tools.prompt


async def test_an_empty_judgement_is_logged_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silent is the whole problem. Retrieval order is a valid-looking
    answer that scores as `hybrid`."""
    with caplog.at_level(logging.WARNING):
        ranked, _ = await _run(
            RerankAgent(k=3, fetch=3, pair_scores=True, passage_mode="title"),
            PairRelevances(scores=[]),
        )
    assert len(ranked) == 3, "the run must continue; one bad response is not fatal"
    assert any("empty scores" in r.getMessage() for r in caplog.records)


async def test_a_normal_judgement_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning on every query is a warning nobody reads."""
    with caplog.at_level(logging.WARNING):
        await _run(
            RerankAgent(k=3, fetch=3, pair_scores=True, passage_mode="title"),
            PairRelevances(scores=[[1, 10], [2, 20], [3, 30]]),
        )
    assert not [r for r in caplog.records if "empty scores" in str(r.msg)]

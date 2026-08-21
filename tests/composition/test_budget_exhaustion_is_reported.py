"""`exhausted_queries` must reach the report.

Closes B-BUDGET-REPORT-1. `PerQueryDeepAgent` counted it and nothing read
it, so a `deep` run where 90% of queries hit the cap was indistinguishable
in the artefacts from one where the agent decided it was finished. Those are
different findings: the first is a result about `MAX_TOOL_CALLS`.

The distinction between `None` and `0` is the same one `tokens_per_query`
makes, and for the same reason -- `dense` has no budget to exhaust, and
reporting 0 would claim it ran to completion under a cap it does not have.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stark_bench.composition.cli as cli_mod
from stark_bench.application.summarise import Row, render
from stark_bench.composition.agent_registry import AGENTS, PerQueryDeepAgent
from stark_bench.domain.run_config import RunConfig

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")

_CONFIG = RunConfig(
    name="c",
    dataset="prime",
    split="test-0.1",
    chunker="whole-document",
    embeddings="e",
    dimension=8,
    aggregation="max",
    agent="deep",
    k=20,
    raw="",
)


def test_the_deep_agent_still_exposes_the_counter() -> None:
    """The report reads this by name. A rename would leave `getattr`
    returning None forever, which renders as `--` and looks deliberate."""
    agent = AGENTS["deep"](_CONFIG)
    assert isinstance(agent, PerQueryDeepAgent)
    assert agent.exhausted_queries == 0


def test_the_run_puts_it_in_the_cost_block() -> None:
    assert 'cost["exhausted_queries"]' in _SRC


def test_it_is_read_off_the_agent_not_invented() -> None:
    """A literal 0 would render identically on a healthy run and hide every
    unhealthy one."""
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and getattr(node.targets[0].slice, "value", None) == "exhausted_queries"
        ):
            assert isinstance(node.value, ast.Call)
            assert node.value.func.id == "getattr"
            # default must be None, not 0
            assert node.value.args[2].value is None
            return
    raise AssertionError("no assignment of cost['exhausted_queries'] found")


def test_an_agent_without_a_budget_reports_none() -> None:
    assert getattr(AGENTS["dense"](_CONFIG), "exhausted_queries", None) is None


def _row(value: object) -> str:
    return render(
        [
            Row(
                config="c",
                agent="deep",
                dataset="prime",
                chunker="w",
                embeddings="e",
                chat_model="m",
                metrics={"mrr": 0.1},
                cost={"exhausted_queries": value},
                ingest={},
            )
        ]
    )


def test_the_table_has_a_cut_off_column() -> None:
    assert "cut off" in _row(7)


def test_a_cut_off_count_is_rendered() -> None:
    assert "| 7 |" in _row(7)


def test_no_budget_renders_a_dash_not_a_zero() -> None:
    """0 would claim `dense` ran to completion under a cap it does not have."""
    out = _row(None)
    assert out.rstrip().endswith("| -- |")


def test_zero_and_none_render_differently() -> None:
    assert _row(0) != _row(None)


def test_header_and_rows_agree_on_column_count() -> None:
    lines = [line for line in _row(3).splitlines() if line.startswith("|")]
    assert len({line.count("|") for line in lines}) == 1

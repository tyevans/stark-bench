"""A run that reported on itself is not a run that failed half its queries.

## The measurement this comes from

Four `rerank40titlerelmatrix` arms, 2026-08-21, all four exiting non-zero
under a gate that counted every WARNING equally:

| cell | warnings | what they were | usable |
|---|---|---|---|
| gemma x whole | 171 | all `not behaving orthogonally` | **yes** |
| gemma x sliding1k | 206 | all `not behaving orthogonally` | **yes** |
| qwen x whole | 147 | 124 `empty scores` | no -- 44.6% fallback |
| qwen x sliding1k | 162 | 131 `empty scores` | no -- 47.1% fallback |

The totals do not separate them. Neither does `llm_calls_per_query`, which
was **1.0000 on all four**: an `empty scores` call succeeded, it just
returned `{"scores": []}`. Neither does the mrr -- the qwen cells scored
0.36942 and 0.37780, comfortably inside the range a slightly-worse reranker
produces.

A gate that fails a clean arm is worse than no gate, because the next
person routes around it. So `degraded` gates and `diagnostics` is recorded.

## Why these assert on behaviour rather than the call site

`test_a_degraded_agent_run_is_not_reported_as_clean.py` already holds the
AST checks that `--run` builds the handler and raises on it. What that
cannot see is *which* warnings it counts as which, and that is the whole
content of this change -- the classification is a data question, testable
directly, so it is tested directly.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import stark_bench.composition.cli as cli_module
from stark_bench.composition.cli import _AgentWarnings


def _record(message: str) -> logging.LogRecord:
    """A record carrying `message` as its format string, as the agents emit it."""
    return logging.LogRecord(
        name="stark_bench.agents.rerank",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_a_failed_extract_call_degrades_the_run() -> None:
    handler = _AgentWarnings()
    handler.emit(_record("rerank: extract failed for query %s"))
    assert handler.degraded == 1
    assert handler.diagnostics == 0


def test_empty_scores_degrades_the_run() -> None:
    """The call succeeded and accomplished nothing, which nothing else shows.

    `llm_calls_per_query` counts the call and stays 1.0. This is the only
    place the distinction survives.
    """
    handler = _AgentWarnings()
    handler.emit(
        _record(
            "rerank: empty scores for query %s -- falling back to "
            "retrieval order, which scores as hybrid"
        )
    )
    assert handler.degraded == 1, "an empty-scores fallback is a degraded query"
    assert handler.diagnostics == 0


def test_a_non_orthogonality_report_does_not_degrade_the_run() -> None:
    """171 of these came from an arm whose every query ranked correctly."""
    handler = _AgentWarnings()
    handler.emit(
        _record(
            "rerank: %.0f%% of rows for query %s scored every dimension "
            "identically -- the dimensions are not behaving orthogonally"
        )
    )
    assert handler.degraded == 0, (
        "an encoding diagnostic gated the run; a gate that fails clean arms "
        "gets routed around"
    )
    assert handler.diagnostics == 1


def test_an_unrecognised_warning_is_a_diagnostic_not_a_failure() -> None:
    """Unknown warnings must not gate, or every new log line breaks the suite.

    The asymmetry is deliberate and is the safe direction: a missed
    degradation shows up as a fallback rate in the log, while a false gate
    stops arms that worked.
    """
    handler = _AgentWarnings()
    handler.emit(_record("rerank: something nobody has classified yet"))
    assert handler.degraded == 0
    assert handler.diagnostics == 1


def test_the_two_counters_still_total_every_warning() -> None:
    """`count` stays the sum, so reports written before the split compare."""
    handler = _AgentWarnings()
    for message in (
        "rerank: extract failed for query %s",
        "rerank: empty scores for query %s",
        "rerank: dimensions are not behaving orthogonally",
    ):
        handler.emit(_record(message))
    assert (handler.degraded, handler.diagnostics) == (2, 1)
    assert handler.count == 3


def test_the_gate_reads_the_degraded_count_and_not_the_total() -> None:
    """Gating on `.count` again would re-fail every clean matrix arm."""
    tree = ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(inner, ast.Attribute)
            and inner.attr == "degraded"
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "warnings"
            for inner in ast.walk(node.test)
        )
        and any(
            isinstance(inner, ast.Raise)
            and isinstance(inner.exc, ast.Call)
            and isinstance(inner.exc.func, ast.Name)
            and inner.exc.func.id == "SystemExit"
            for inner in ast.walk(node)
        )
    ]
    assert gates, (
        "no `if warnings.degraded: raise SystemExit(...)` in cli.py -- the "
        "gate either vanished or went back to gating on the total"
    )


def test_both_counts_reach_the_report() -> None:
    """A diagnostic count nobody records is a diagnostic nobody reads."""
    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    for key, attribute in (
        ("agent_warnings_degraded", "warnings.degraded"),
        ("agent_warnings_diagnostic", "warnings.diagnostics"),
    ):
        assert (
            f'cost["{key}"] = {attribute}' in source
        ), f"cost[{key!r}] is not assigned from {attribute}"

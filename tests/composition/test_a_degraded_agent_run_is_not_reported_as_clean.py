"""A rerank arm whose LLM calls fail scores like `hybrid`, and says nothing.

This happened twice on 2026-08-21, hours apart, against a healthy-looking
endpoint. `llama-swap` answered `502 Bad Gateway` for a model it could not
serve: once for every call in the run, giving `llm_calls_per_query = 0.0`
and an mrr of 0.29188 that read as a plausible architecture result; once
for 8.2% of them, giving a number quietly depressed by an eighth of its
queries never being ranked.

`run_queries` logged `0 empty` both times, because retrieval order is not
empty. The agent logged `rerank: extract failed` exactly as designed --
and nothing read the log.

So the gate is: an agent warning makes the run exit non-zero, *after* the
report is written. Losing half an hour of predictions to a transient 502
would be a worse trade than keeping them; what must not happen is a
caller looping over arms recording this one as `ok`.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import stark_bench.composition.cli as cli_module
from stark_bench.composition.cli import _AgentWarnings


def test_the_handler_counts_agent_warnings() -> None:
    handler = _AgentWarnings()
    logger = logging.getLogger("stark_bench.agents.rerank")
    logger.addHandler(handler)
    try:
        logger.warning("rerank: extract failed for query %s", 7)
        logger.warning("rerank: empty scores for query %s", 9)
        logger.info("this is not a warning")
    finally:
        logger.removeHandler(handler)
    assert handler.count == 2


def test_it_ignores_anything_below_warning() -> None:
    """An INFO-level log is the normal path and must not fail a run."""
    handler = _AgentWarnings()
    logger = logging.getLogger("stark_bench.agents.rerank")
    logger.addHandler(handler)
    try:
        logger.info("280/280 queries done")
        logger.debug("candidate rendered")
    finally:
        logger.removeHandler(handler)
    assert handler.count == 0


def _run_function() -> ast.AsyncFunctionDef:
    tree = ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_run"
    )


def test_the_run_path_attaches_the_handler_and_raises_on_a_warning() -> None:
    """The handler is useless unless the run installs it and acts on it.

    An AST check because running this path needs Postgres, Neo4j and an
    endpoint -- the same reasoning as
    `test_ingest_stats_reach_the_report.py`, which exists because a correct
    helper nobody called has shipped twice in this repo.
    """
    source = ast.dump(_run_function())
    assert "_AgentWarnings" in source, (
        "the run path never constructs _AgentWarnings, so a degraded arm "
        "will report success"
    )
    assert "addHandler" in source, "the handler is constructed but never attached"
    # Not merely "a SystemExit appears somewhere": `if False: raise
    # SystemExit(...)` satisfies that and is dead code. Checked by making
    # exactly that edit and watching this test stay green, which is why it
    # now asserts the guard reads the counter.
    function = _run_function()
    guarded = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "warnings"
            for name in ast.walk(node.test)
        )
        and any(
            isinstance(inner, ast.Raise)
            and isinstance(inner.exc, ast.Call)
            and isinstance(inner.exc.func, ast.Name)
            and inner.exc.func.id == "SystemExit"
            for inner in ast.walk(node)
        )
    ]
    assert guarded, (
        "no `if warnings...: raise SystemExit(...)` in the run path -- a "
        "warning during the run must make the process exit non-zero, or a "
        "caller looping over arms records a degraded arm as ok"
    )


def test_the_report_is_written_before_the_run_fails() -> None:
    """Order matters: keep the data, fail the run.

    Re-running a rerank arm is ~30 minutes. The predictions from a
    partially-degraded run are still worth having, and `agent_warnings` in
    the cost block is what tells a later reader not to quote the number.
    """
    function = _run_function()
    write_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_report"
    ]
    raise_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "SystemExit"
    ]
    assert write_lines, "no write_report call in the run path"
    assert raise_lines, "no SystemExit raise in the run path"
    # Line numbers rather than `ast.walk` order: walk is breadth-first and
    # says nothing about source order, which is the thing under test.
    assert max(write_lines) < min(raise_lines), (
        "the run raises before writing the report, discarding predictions "
        "that cost half an hour"
    )

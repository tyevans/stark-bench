"""An approximate retrieval must say so in the file it writes.

An HNSW index makes retrieval approximate. Measured on `qwen-rel-whole`,
280 queries: the default `hnsw.ef_search = 40` costs 0.0117 of recall@20
against an exact scan, and 800 matches it to every digit. That is larger
than several of the differences this project reports between architectures,
and **nothing else in a report can see it** -- `config_verbatim` is the
config FILE's bytes, and the index lives in Postgres, which no config
mentions.

So two runs of the same arm either side of an index produce files that
differ only in the metric. That reads as an architecture result and is not
one, which is this project's signature failure shape.

The AST test below exists because the helper being correct is not the thing
that has gone wrong here before. Twice, hours apart, a correct helper was
shipped that the call site did not use, with the whole suite green both
times -- see `test_ingest_stats_reach_the_report.py`, which is this test's
template and was written after the second occurrence.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stark_bench.adapters.postgres_retrieval_stats as stats_module
import stark_bench.composition.cli as cli_module


def _cli_tree() -> ast.Module:
    return ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))


def _stats_tree() -> ast.Module:
    return ast.parse(Path(stats_module.__file__).read_text(encoding="utf-8"))


def test_the_run_path_calls_retrieval_stats() -> None:
    """The helper is useless unless `--run` invokes it."""
    calls = [
        node
        for node in ast.walk(_cli_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "retrieval_stats"
    ]
    assert calls, (
        "retrieval_stats is never called -- every report will omit whether "
        "retrieval was exact, and an indexed run will be indistinguishable "
        "from an exact one"
    )


def test_retrieval_stats_is_merged_into_cost() -> None:
    """Called is not enough; its result has to reach the report.

    `await _retrieval_stats(...)` on a line by itself would satisfy the test
    above and record nothing, which is the same defect one step further on.
    """
    merged = [
        node
        for node in ast.walk(_cli_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cost"
        and any(
            isinstance(arg, ast.Await)
            and isinstance(arg.value, ast.Call)
            and isinstance(arg.value.func, ast.Name)
            and arg.value.func.id == "retrieval_stats"
            for arg in node.args
        )
    ]
    assert merged, "the result of retrieval_stats is not merged into `cost`"


def test_the_helper_does_not_swallow_its_own_failures() -> None:
    """A provenance check that cannot fail loudly is not a check.

    The first draft wrapped the whole body in `except UndefinedTableError:
    return {}`. A typo in the query -- `s.idx_scan` against an alias named
    `i` -- raises exactly that, so the run wrote a clean, empty provenance
    block and reported success. The absence looked like "no index here".
    """
    function = next(
        node
        for node in ast.walk(_stats_tree())
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "retrieval_stats"
    )
    handlers = [
        node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
    ]
    assert not handlers, (
        "retrieval_stats catches an exception; a failed provenance query must "
        "fail the run rather than return an empty block indistinguishable from "
        "an unindexed corpus"
    )

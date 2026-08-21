"""`seconds_total` and `seconds_wall` are different numbers now.

`seconds_total` sums per-call durations and was wall time only while
`run_queries.run` was serial. Under concurrency the calls overlap, so the
sum counts the same seconds up to N times: a real run reported 1933s while
taking ~480s, rendered under a column headed `seconds`.

The failure this guards is not a crash. It is a cost column whose meaning
silently depends on a flag not shown beside it.
"""

from __future__ import annotations

from pathlib import Path


import stark_bench.composition.cli as cli_mod
from stark_bench.adapters.report_file import summarise_cost
from stark_bench.domain import ToolCall


def _calls(n: int, each: float) -> list[ToolCall]:
    return [ToolCall(tool="extract", duration_s=each, result_count=1) for _ in range(n)]


def test_compute_seconds_still_sum_the_calls() -> None:
    cost = summarise_cost(_calls(4, 2.0), queries=4)
    assert cost["seconds_total"] == 8.0


def test_wall_seconds_are_recorded_when_given() -> None:
    cost = summarise_cost(_calls(4, 2.0), queries=4, wall_s=2.5, concurrency=4)
    assert cost["seconds_wall"] == 2.5


def test_wall_and_compute_disagree_under_concurrency() -> None:
    """The whole point: four 2s calls at concurrency 4 take ~2s, not 8s."""
    cost = summarise_cost(_calls(4, 2.0), queries=4, wall_s=2.1, concurrency=4)
    assert cost["seconds_total"] == 8.0
    assert cost["seconds_wall"] == 2.1


def test_the_concurrency_is_recorded_beside_them() -> None:
    """A wall figure without it invites a comparison it cannot support."""
    assert (
        summarise_cost(_calls(1, 1.0), queries=1, concurrency=4)["query_concurrency"]
        == 4
    )


def test_wall_is_none_rather_than_zero_when_not_measured() -> None:
    """Same rule as `tokens_per_query`: not measured and measured as nothing
    are different facts, and 0.0 would render as an instant run."""
    assert summarise_cost(_calls(1, 1.0), queries=1)["seconds_wall"] is None


def test_the_zero_query_branch_carries_both_keys() -> None:
    """It is a separate return statement and has drifted from the main one
    before."""
    empty = summarise_cost([], queries=0, wall_s=1.0, concurrency=2)
    full = summarise_cost(_calls(1, 1.0), queries=1, wall_s=1.0, concurrency=2)
    assert set(empty) == set(full)


def test_the_run_measures_wall_time_around_the_query_set() -> None:
    """AST, because running the call site needs Postgres, Neo4j and an
    endpoint. Catches wall time being derived from the calls instead."""
    src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    body = src.split("async def _do_run")[1].split("\nasync def ")[0]
    assert "perf_counter()" in body
    assert "wall_s=run_wall_s" in body
    assert "concurrency=config.query_concurrency" in body


def test_summarise_renders_both_columns_and_the_concurrency() -> None:
    from stark_bench.application.summarise import Row, render

    row = Row(
        config="c",
        agent="a",
        dataset="d",
        chunker="k",
        embeddings="e",
        chat_model="m",
        metrics={"mrr": 0.5},
        cost={"seconds_total": 1933.0, "seconds_wall": 480.0, "query_concurrency": 4},
        ingest={},
    )
    out = render([row])
    assert "wall seconds" in out and "gpu seconds" in out and "conc" in out
    assert "1933.0" in out and "480.0" in out and "| 4 |" in out
    # Header, separator and row must agree on column count. A header patch
    # that silently failed to apply left the row with 14 fields under a
    # 12-field header, which renders as a broken table rather than an error.
    lines = [line for line in out.splitlines() if line.startswith("|")]
    assert len({line.count("|") for line in lines}) == 1

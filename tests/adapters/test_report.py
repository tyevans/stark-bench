import json

from stark_bench.domain.run_config import RunConfig
from stark_bench.adapters.report_file import summarise_cost, write_report
from stark_bench.domain import ToolCall


def test_cost_is_per_query_not_total_calls():
    calls = [ToolCall("search_chunks", 0.1, 5) for _ in range(10)]
    calls += [ToolCall("extract", 0.2, 1) for _ in range(5)]
    cost = summarise_cost(calls, queries=5)
    assert cost["tool_calls_per_query"] == 3.0
    assert cost["llm_calls_per_query"] == 1.0


def test_zero_queries_does_not_divide_by_zero():
    assert summarise_cost([], queries=0)["tool_calls_per_query"] == 0.0


def test_the_report_embeds_the_config_verbatim(tmp_path):
    config = RunConfig(
        "vss-control",
        "prime",
        "test-0.1",
        "whole-document",
        "precomputed-ada002",
        1536,
        "max",
        "dense",
        20,
        raw="name: vss-control\n",
    )
    out = tmp_path / "r.json"
    write_report(
        out,
        config=config,
        metrics={"mrr": 0.4},
        cost={"tool_calls_per_query": 1.0},
        ingest={"nodes": 12},
        queries=3,
    )
    written = json.loads(out.read_text())
    assert written["config_verbatim"] == "name: vss-control\n"
    assert written["metrics"]["mrr"] == 0.4

"""The table builder's checks must fire, not merely exist.

Every check in `scripts/results_table.py` corresponds to a defect this
project actually shipped, so each one is tested by constructing the defect
and asserting it is caught. A checker whose checks have never fired is
indistinguishable from a checker with no checks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def table(monkeypatch, tmp_path):
    """The script, loaded by path, pointed at a throwaway results tree."""
    spec = importlib.util.spec_from_file_location(
        "results_table", ROOT / "scripts" / "results_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["results_table"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    (tmp_path / "results").mkdir()
    (tmp_path / "config").mkdir()
    module._test_root = tmp_path
    return module


def _write(table, name: str, data: dict) -> None:
    (table.RESULTS / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def _good(**overrides) -> dict:
    base = {
        "config_name": "arm",
        "config_verbatim": "name: arm\n",
        "queries": 280,
        "metrics": {"mrr": 0.23, "hit@1": 0.15, "hit@5": 0.31, "recall@20": 0.38},
        "cost": {"tool_calls_per_query": 1.0, "llm_calls_per_query": 0.0},
        # A *realistic* corpus size, not 100. The staleness check compares
        # against the real node count of prime/test-0.1, so a toy fixture
        # would make every clean report look stale -- and, before that check
        # existed, a toy fixture is what let a 3000-node probe's report
        # render as the full arm's ingest stats without any test noticing.
        "ingest": {"nodes": 129375, "chunks": 136803, "edges_ingested": True},
    }
    base.update(overrides)
    return base


def test_a_clean_report_raises_nothing(table, monkeypatch):
    monkeypatch.chdir(table._test_root)
    (table._test_root / "config" / "arm.yaml").write_text("name: arm\n")
    _write(table, "arm.dense", _good())
    # The staleness check reads config/ relative to the script, not cwd, so a
    # config it cannot find is skipped -- assert only on the other checks.
    assert [p for p in table.check(table.load()) if "STALE" not in p] == []


def test_all_zero_metrics_are_caught(table):
    """280 queries retrieving nothing, with no error logged -- a real incident."""
    _write(table, "arm.dense", _good(metrics={"mrr": 0.0, "hit@1": 0.0}))
    assert any("every metric is zero" in p for p in table.check(table.load()))


def test_an_empty_ingest_block_is_caught(table):
    """`write_report(ingest={})` emptied the cost column of every report."""
    _write(table, "arm.dense", _good(ingest={}))
    assert any("empty ingest block" in p for p in table.check(table.load()))


def test_a_deep_run_on_an_edgeless_corpus_is_caught(table):
    """B-DEEP-EDGES-1: traversal returns empty and it reads as an architecture result."""
    _write(
        table,
        "arm.deep",
        _good(ingest={"nodes": 1, "chunks": 1, "edges_ingested": False}),
    )
    assert any("edgeless" in p for p in table.check(table.load()))


def test_the_same_report_under_dense_is_not_flagged_edgeless(table):
    """Only `deep` traverses -- flagging `dense` would train people to ignore it."""
    _write(
        table,
        "arm.dense",
        _good(ingest={"nodes": 1, "chunks": 1, "edges_ingested": False}),
    )
    assert not any("edgeless" in p for p in table.check(table.load()))


def test_zero_queries_is_caught(table):
    _write(table, "arm.dense", _good(queries=0))
    assert any("scored 0 queries" in p for p in table.check(table.load()))


def test_a_missing_metric_renders_as_a_dash_not_a_zero(table):
    """A cost of zero and an absent cost are different claims."""
    _write(table, "arm.dense", _good(cost={}))
    rendered = table.render(table.load())
    assert "| -- | -- |" in rendered
    assert "0.00 | 0.00" not in rendered


def test_an_ingest_report_from_a_smaller_run_is_caught(table):
    """A report describing a different, smaller corpus than the one scored.

    `resume_is_safe.py` cannot see this: it compares the recorded config
    text, and a `--limit 3000` probe writes the *same* config bytes as the
    full run, so the comparison passes.

    Observed 2026-08-19: RESULTS.md published `chunks/node 1.000` and
    `ingest s 207.7` for native-wholedoc, both taken from a 3000-node
    probe, while the real corpus held 136,803 chunks over 129,375 nodes.
    """
    _write(
        table,
        "arm.dense",
        _good(ingest={"nodes": 3000, "chunks": 3000, "edges_ingested": False}),
    )
    problems = table.check(table.load())
    assert any("3000 nodes" in p and "129375" in p for p in problems), problems


def test_a_full_corpus_report_is_not_flagged_as_stale(table):
    """The check must discriminate, not fire on everything.

    Without this, raising EXPECTED_NODES above every real corpus would
    'pass' the test above while flagging every honest run.
    """
    _write(table, "arm.dense", _good())
    assert not [p for p in table.check(table.load()) if "stale report" in p]


def test_granularity_below_one_is_caught(table):
    """Chunks/node under 1.0 is impossible, so it means broken arithmetic.

    RESULTS.md rendered `0.381` for native-wholedoc on 2026-08-19, from a
    resumed ingest whose `chunks` counted only what that run wrote. Nothing
    flagged it, and the number is the one the whole chunking sweep is about.
    """
    _write(
        table,
        "arm.hybrid",
        _good(
            ingest={
                "nodes": 129375,
                "chunks": 49280,
                "skipped": 0,
                "edges_ingested": True,
            }
        ),
    )
    problems = table.check(table.load())
    assert any("below 1.0 is" in p for p in problems), problems


def test_a_resumed_ingest_renders_the_corpus_granularity(table):
    """chunks + skipped, not chunks. The real arm-1 numbers."""
    _write(
        table,
        "arm.hybrid",
        _good(
            ingest={
                "nodes": 129375,
                "chunks": 49280,
                "skipped": 87523,
                "edges_ingested": True,
            }
        ),
    )
    rendered = table.render(table.load())
    assert "1.057" in rendered, rendered
    assert "0.381" not in rendered, rendered

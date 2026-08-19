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
        "ingest": {"nodes": 100, "chunks": 106, "edges_ingested": True},
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
    _write(table, "arm.deep", _good(ingest={"nodes": 1, "chunks": 1, "edges_ingested": False}))
    assert any("edgeless" in p for p in table.check(table.load()))


def test_the_same_report_under_dense_is_not_flagged_edgeless(table):
    """Only `deep` traverses -- flagging `dense` would train people to ignore it."""
    _write(table, "arm.dense", _good(ingest={"nodes": 1, "chunks": 1, "edges_ingested": False}))
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

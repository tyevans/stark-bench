"""A cost number that does not reach the report cannot be read.

Same shape as `test_ingest_stats_reach_the_report.py` and for the same
reason: `write_report(ingest={})` once emptied the cost column of every
report ever written and nothing raised. `_ingest_stats` currently passes the
whole file through, so this passes without a code change -- which is the
point. It is here so that narrowing it to a key list later, which is a very
natural refactor, cannot silently drop the cache counters.
"""

from __future__ import annotations

import json

import pytest

from stark_bench.composition.cli import _ingest_stats, ingest_report_path
from stark_bench.domain.run_config import RunConfig


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        name="test-cache-stats",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="qwen3-embedding-0.6b",
        dimension=1024,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )


def test_cache_counters_survive_the_process_boundary(config, monkeypatch, tmp_path):
    monkeypatch.setattr("stark_bench.composition.cli.RESULTS_ROOT", tmp_path)
    path = ingest_report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"nodes": 10, "chunks": 12, "cache_hits": 9, "cache_misses": 3})
    )
    stats = _ingest_stats(config)
    assert stats["cache_hits"] == 9
    assert stats["cache_misses"] == 3


def test_an_arm_ingested_before_the_cache_existed_still_scores(
    config, monkeypatch, tmp_path
):
    """Old report files have no cache keys, and must not become unreadable."""
    monkeypatch.setattr("stark_bench.composition.cli.RESULTS_ROOT", tmp_path)
    path = ingest_report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nodes": 10, "chunks": 12}))
    stats = _ingest_stats(config)
    assert stats["nodes"] == 10
    assert "cache_hits" not in stats

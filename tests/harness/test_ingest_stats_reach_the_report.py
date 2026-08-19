"""An ingest's cost has to survive the process boundary to be reported.

`--ingest` and `--run` are separate processes on purpose -- one is hours and
the other is minutes -- so the run cannot observe what the ingest cost and
has to read it back from disk. Until now it passed `{}` and every report
carried an empty `ingest` block, which is indistinguishable from an ingest
that did nothing.
"""

from __future__ import annotations

import json

import pytest

from stark_bench.harness.cli import _ingest_stats, ingest_report_path
from stark_bench.harness.config import RunConfig


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        name="test-ingest-stats",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="Nemotron-3-Embed-1B",
        dimension=2048,
        aggregation="max",
        agent="dense",
        k=20,
        raw="",
    )


def test_missing_file_is_empty_rather_than_an_error(config, monkeypatch, tmp_path):
    """Scoring a hand-ingested corpus must not be refused over a cost column."""
    monkeypatch.setattr("stark_bench.harness.cli.RESULTS_ROOT", tmp_path)
    assert _ingest_stats(config) == {}


def test_the_written_stats_come_back(config, monkeypatch, tmp_path):
    """Values chosen to differ from each other and from any plausible default.

    A block of zeroes and `False`s could not distinguish "read the file"
    from "returned a fresh dict of the same shape".
    """
    monkeypatch.setattr("stark_bench.harness.cli.RESULTS_ROOT", tmp_path)
    written = {
        "nodes": 129375,
        "chunks": 147329,
        "skipped": 7,
        "edges": 3,
        "self_loops_dropped": 1,
        "edges_ingested": True,
        "resume": False,
        "existing_ids_load_s": 0.25,
        "wall_time_s": 2564.78,
    }
    ingest_report_path(config).parent.mkdir(parents=True, exist_ok=True)
    ingest_report_path(config).write_text(json.dumps(written), encoding="utf-8")

    assert _ingest_stats(config) == written


def test_the_path_is_keyed_on_the_config_not_the_agent(config, monkeypatch, tmp_path):
    """One ingest serves all four agents, so four runs read one file."""
    monkeypatch.setattr("stark_bench.harness.cli.RESULTS_ROOT", tmp_path)
    from dataclasses import replace

    assert ingest_report_path(config) == ingest_report_path(replace(config, agent="deep"))


def test_a_float_and_a_bool_survive_the_round_trip(config, monkeypatch, tmp_path):
    """The block was annotated `Mapping[str, int]`, which was never true of it.

    `wall_time_s` is a float and `edges_ingested` a bool. The annotation was
    accurate only while the block was always empty -- an exemption that
    matched nothing, and passed for exactly that reason.
    """
    monkeypatch.setattr("stark_bench.harness.cli.RESULTS_ROOT", tmp_path)
    ingest_report_path(config).parent.mkdir(parents=True, exist_ok=True)
    ingest_report_path(config).write_text(
        json.dumps({"wall_time_s": 90.55539454508107, "edges_ingested": True}),
        encoding="utf-8",
    )
    stats = _ingest_stats(config)
    assert stats["wall_time_s"] == 90.55539454508107
    assert stats["edges_ingested"] is True


def test_the_run_actually_passes_the_stats_it_read():
    """Reverting `--run` to `ingest={}` passed all 45 harness tests.

    That is the second time in this session the same shape appeared: a
    helper covered from every angle, and nothing asserting the call site
    uses it. Both times the defect being prevented is not "the helper is
    wrong" but "nobody calls the helper", and no test of the helper can
    see it.

    Checked structurally rather than by running `_do_run`, which needs
    Postgres, Neo4j and an embedding endpoint. The claim is narrow and it
    is exactly the claim: whatever is passed as `ingest=` to `write_report`
    is not an empty literal.
    """
    import ast
    from pathlib import Path

    import stark_bench.harness.cli as cli_module

    tree = ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_report"
    ]
    assert calls, "no write_report call found -- this test would pass vacuously"
    for call in calls:
        ingest = next((kw.value for kw in call.keywords if kw.arg == "ingest"), None)
        assert ingest is not None, "write_report called without ingest="
        assert not (isinstance(ingest, ast.Dict) and not ingest.keys), (
            "write_report is passed an empty ingest literal -- the ingest cost "
            "will be missing from every report"
        )

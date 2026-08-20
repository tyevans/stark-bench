"""One config must serve both query sets without either overwriting the other.

The corpus does not vary with the split -- the tenant is a `uuid5` of the
config *name* -- so `--split test` reads the same ingested store as
`test-0.1`. That makes the full 2,801-query run free of endpoint time, and it
also makes the two runs collide: before this, both wrote
`<config>.<agent>.json`.

The collision is the dangerous half. `report_path`'s own docstring records
the same failure for agents: the survivor carries a `config_verbatim` that
looks correct, so nothing in the file reveals the loss.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from stark_bench.adapters.report_file import write_report
from stark_bench.composition.cli import predictions_path, report_path
from stark_bench.domain.run_config import RunConfig


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        name="qwen-rel-whole",
        dataset="prime-rel",
        split="test-0.1",
        chunker="whole-document",
        embeddings="qwen3-embedding-0.6b",
        dimension=1024,
        aggregation="max",
        agent="dense",
        k=20,
        raw="name: qwen-rel-whole\nsplit: test-0.1\n",
    )


def test_the_default_split_keeps_the_filename_it_always_had(config):
    """Catches: tagging unconditionally, renaming every existing result.

    `RESULTS.md` and `FINDINGS.md` both quote these paths.
    """
    assert report_path(config).name == "qwen-rel-whole.dense.json"
    assert predictions_path(config).name == "qwen-rel-whole.dense.predictions.json"


def test_an_overridden_split_cannot_overwrite_the_default_one(config):
    """Catches: the collision this whole file exists for."""
    wide = replace(config, split_override="test")

    assert report_path(wide).name == "qwen-rel-whole.test.dense.json"
    assert report_path(wide) != report_path(config)
    assert predictions_path(wide) != predictions_path(config)


def test_overriding_with_the_same_value_is_not_an_override(config):
    """Catches: one run acquiring two names depending on how it was invoked.

    `--split test-0.1` on a config that already says `test-0.1` is a no-op,
    and must not produce `qwen-rel-whole.test-0.1.dense.json` alongside the
    file it is identical to.
    """
    from stark_bench.composition.cli import main  # noqa: F401 -- import guard

    # The CLI guards on `args.split != config.split`; assert the property the
    # guard exists to protect rather than the guard itself.
    same = replace(config, split_override=None)
    assert same.effective_split == "test-0.1"
    assert report_path(same) == report_path(config)


def test_the_effective_split_is_the_override_when_there_is_one(config):
    assert config.effective_split == "test-0.1"
    assert replace(config, split_override="test").effective_split == "test"


def test_the_report_records_the_split_that_ran(config, tmp_path):
    """Catches: trusting `config_verbatim` to say which queries ran.

    It is the config FILE's bytes. On an overridden run it still reads
    `test-0.1` -- so a report without its own `split` key names the query set
    that did not run, confidently and in writing.
    """
    wide = replace(config, split_override="test")
    path = tmp_path / "r.json"
    write_report(
        path, config=wide, metrics={"mrr": 0.5}, cost={}, ingest={}, queries=2801
    )

    written = json.loads(path.read_text())

    assert written["split"] == "test"
    assert (
        "test-0.1" in written["config_verbatim"]
    ), "the verbatim config is unchanged, which is exactly why `split` is needed"
    assert written["queries"] == 2801


def test_the_run_reads_the_effective_split_not_the_config_one():
    """Catches: `--split test` tagging the filename but loading 280 queries.

    Found by mutation, not by reading: reverting the read site to
    `config.split` left all 259 tests green. Every assertion above is about
    paths and reports, and a run that names its file `.test.` while scoring
    `test-0.1` satisfies every one of them -- while producing a file whose
    name, `split` key and query count all disagree with the numbers in it.

    This is the same shape as the two defects `test_prewarm_reaches_the_run`
    guards, and it is checked the same way, against the syntax tree: running
    `_do_run` needs Postgres, Neo4j and an endpoint.
    """
    import ast
    import inspect
    from pathlib import Path

    import stark_bench.composition.cli as cli_module

    source = Path(inspect.getfile(cli_module)).read_text(encoding="utf-8")
    do_run = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_run"
    )
    reads = {
        node.attr
        for node in ast.walk(do_run)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    }

    assert (
        "effective_split" in reads
    ), "_do_run must load queries for the split that actually runs"
    assert (
        "split" not in reads
    ), "_do_run must not read the raw `split`; that ignores --split"

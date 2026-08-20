"""The summariser must not let two corpora share a table.

This file exists because of a specific wrong conclusion. `qwen-wholedoc`
(dense mrr 0.183) was compared against `vss-control` (0.231) and read as
"qwen3-embedding is a weak model". It is not a model comparison: vss-control
embeds STaRK's `add_rel=True` documents, which name every neighbour, and the
prime arms embed `add_rel=False` text, which does not. A flat table invites
that read; a table per corpus does not.
"""

from __future__ import annotations

import json

from stark_bench.application.summarise import Row, read_rows, render, summarise


def _report(
    name, agent, dataset, mrr, *, chunker="whole-document", nodes=10, chunks=12
):
    return name + "." + agent + ".json", {
        "config_name": name,
        "config_verbatim": (
            "# a comment mentioning dataset: not-this-one\n"
            f"name: {name}\ndataset: {dataset}\nchunker: {chunker}\n"
            "embeddings: qwen3-embedding-0.6b\n"
        ),
        "queries": 280,
        "metrics": {"mrr": mrr, "hit@1": 0.1, "hit@5": 0.2, "recall@20": 0.3},
        "cost": {
            "llm_calls_per_query": 0.0,
            "tokens_per_query": None,
            "seconds_total": 1.0,
        },
        "ingest": {"nodes": nodes, "chunks": chunks, "skipped": 0},
    }


def _write(tmp_path, *reports):
    for filename, body in reports:
        (tmp_path / filename).write_text(json.dumps(body))
    return tmp_path


def test_two_corpora_do_not_share_a_table(tmp_path):
    _write(
        tmp_path,
        _report("qwen-wholedoc", "dense", "prime", 0.183),
        _report("qwen-rel-whole", "dense", "prime-rel", 0.31),
    )
    out = summarise(tmp_path)
    assert "## `prime`" in out and "## `prime-rel`" in out
    # The two rows must be separated by the second heading, not adjacent.
    first = out.index("qwen-wholedoc")
    second = out.index("qwen-rel-whole")
    between = out[min(first, second) : max(first, second)]
    assert "## `" in between, "two corpora rendered into one table"


def test_ingest_and_prediction_files_are_not_rows(tmp_path):
    _write(tmp_path, _report("a", "dense", "prime", 0.2))
    (tmp_path / "a.ingest.json").write_text(json.dumps({"nodes": 5, "chunks": 6}))
    (tmp_path / "a.dense.predictions.json").write_text(json.dumps({"1": ["2"]}))
    assert len(read_rows(tmp_path)) == 1


def test_a_malformed_file_does_not_stop_the_summary(tmp_path):
    """The directory is busiest exactly when a run has just crashed."""
    _write(tmp_path, _report("a", "dense", "prime", 0.2))
    (tmp_path / "broken.json").write_text("{not json")
    assert len(read_rows(tmp_path)) == 1


def test_the_agent_comes_from_the_filename(tmp_path):
    """One config serves every agent, so config `agent:` is not what ran."""
    _write(tmp_path, _report("a", "rerank", "prime", 0.2))
    assert read_rows(tmp_path)[0].agent == "rerank"


def test_dataset_is_read_from_the_config_not_a_comment(tmp_path):
    """These configs carry long comments that mention the same keys."""
    _write(tmp_path, _report("a", "dense", "prime-rel", 0.2))
    assert read_rows(tmp_path)[0].dataset == "prime-rel"


def test_chunks_per_node_counts_skipped_chunks():
    """A resumed arm holds chunks it did not write this run.

    An arm once rendered 0.381 for a corpus whose real granularity was 1.058
    -- below the 1.0 every chunker here must exceed -- because this counted
    writes only.
    """
    row = Row(
        config="a",
        agent="dense",
        dataset="prime",
        chunker="w",
        embeddings="m",
        metrics={"mrr": 0.2},
        cost={},
        ingest={"nodes": 129375, "chunks": 49280, "skipped": 87523},
    )
    assert round(row.chunks_per_node, 3) == 1.057


def test_unmeasured_tokens_render_as_dashes_not_zero(tmp_path):
    """`None` and `0` are different findings, and ToolCall.tokens says so."""
    _write(tmp_path, _report("a", "dense", "prime", 0.2))
    assert "| -- |" in summarise(tmp_path)


def test_an_empty_directory_says_so(tmp_path):
    assert "No scored arms found" in summarise(tmp_path)


def test_rows_are_ordered_by_mrr_within_a_corpus(tmp_path):
    _write(
        tmp_path,
        _report("low", "dense", "prime", 0.10),
        _report("high", "dense", "prime", 0.30),
    )
    out = render(read_rows(tmp_path))
    assert out.index("`high`") < out.index("`low`")

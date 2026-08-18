"""The exporter's output must be readable by the harness's own reader.

This is the seam between two interpreters, and it is the one place where a
mismatch produces a confusing failure hours into an ingest.
"""

import json

import pytest

from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries


@pytest.fixture
def exported(tmp_path):
    (tmp_path / "nodes.jsonl").write_text(
        json.dumps({"node_id": "0", "node_type": "drug", "name": "x", "document": "d"})
        + "\n"
    )
    (tmp_path / "edges.jsonl").write_text(
        json.dumps({"source": "0", "target": "1", "relation": "targets"}) + "\n"
    )
    (tmp_path / "queries.test.jsonl").write_text(
        json.dumps({"query_id": 5, "text": "q", "answer_ids": ["0"]}) + "\n"
    )
    return tmp_path


def test_the_readers_accept_the_exporter_schema(exported):
    assert list(read_nodes(exported / "nodes.jsonl"))[0].node_id == "0"
    assert list(read_edges(exported / "edges.jsonl"))[0].relation == "targets"
    query, answers = list(read_queries(exported / "queries.test.jsonl"))[0]
    assert query.query_id == 5
    assert answers == ["0"]

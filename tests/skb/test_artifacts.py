from pathlib import Path

from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_skb"


def test_nodes_round_trip():
    nodes = list(read_nodes(FIXTURE / "nodes.jsonl"))
    assert len(nodes) == 12
    assert nodes[0].node_id == "1"
    assert nodes[0].node_type == "drug"
    assert "cyclooxygenase" in nodes[0].document


def test_edges_include_the_self_loop_unfiltered():
    """Reading is not filtering.

    The self-loop must survive to the loader, which drops it and records the
    count. A reader that silently dropped it would make a recall ceiling look
    like a retrieval failure later.
    """
    edges = list(read_edges(FIXTURE / "edges.jsonl"))
    assert len(edges) == 11
    assert any(e.source == e.target for e in edges)


def test_queries_carry_answers_separately_from_the_query():
    pairs = list(read_queries(FIXTURE / "queries.jsonl"))
    query, answers = pairs[0]
    assert query.query_id == 1
    assert not hasattr(query, "answer_ids")
    assert answers == ["6"]

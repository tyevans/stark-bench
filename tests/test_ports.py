from stark_bench.domain import Query, Ranked
from stark_bench.ports import Agent


def test_ranked_is_a_node_id_and_a_score():
    r = Ranked(node_id="12345", score=0.5)
    assert r.node_id == "12345"
    assert r.score == 0.5


def test_query_carries_no_answer():
    """An agent must never be able to see ground truth."""
    q = Query(query_id=7, text="which drugs target PTGS2?")
    assert not hasattr(q, "answer_ids")
    assert q.query_id == 7


def test_agent_is_runtime_checkable():
    class Stub:
        async def retrieve(self, query, tools):
            return []

    assert isinstance(Stub(), Agent)

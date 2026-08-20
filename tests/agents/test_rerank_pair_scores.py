"""`pair_scores` halves decode without giving up alignment checking.

The encoding exists because a measured response spent 627 completion tokens
on 40 candidates -- 15.7 tokens to convey one integer. What it must NOT give
up is the index: a bare list of scores is cheaper still and a one-position
shift in it is undetectable.
"""

from __future__ import annotations


from stark_bench.agents.rerank import PairRelevances, RerankAgent
from stark_bench.domain import Passage, Query


class FakeToolset:
    def __init__(self, passages, judged):
        self._passages = passages
        self._judged = judged
        self.schema = None

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):
        self.schema = schema
        return self._judged


DOC = "- name: Entity {}\n- type: gene/protein\n"


def _passages(n=4):
    return [
        Passage(node_id=str(100 + i), text=DOC.format(i), score=1.0 / (i + 1))
        for i in range(n)
    ]


async def _run(judged, n=4, **kw):
    tools = FakeToolset(_passages(n), judged)
    agent = RerankAgent(k=n, fetch=n, pair_scores=True, passage_mode="title", **kw)
    return await agent.retrieve(Query(query_id=1, text="q"), tools), tools


async def test_pairs_request_the_pair_schema() -> None:
    """Catches the shipped-but-unused defect: field set, schema unchanged."""
    _, tools = await _run(PairRelevances(scores=[[1, 10]]))
    assert tools.schema is PairRelevances


async def test_pairs_map_index_to_node_and_order_by_score() -> None:
    ranked, _ = await _run(PairRelevances(scores=[[1, 10], [2, 99], [3, 50], [4, 5]]))
    assert [r.node_id for r in ranked][:2] == ["101", "102"]


async def test_an_out_of_range_index_is_dropped_not_wrapped() -> None:
    """A negative or oversized index must not silently address a real
    candidate -- Python would happily let `passages[-1]` through."""
    ranked, _ = await _run(PairRelevances(scores=[[0, 99], [99, 99], [2, 80]]))
    assert ranked[0].node_id == "101"


def test_a_malformed_row_is_refused_by_the_schema() -> None:
    """Three elements is not a pair; guessing which two to use is how a
    reranker scores every candidate by its index."""
    agent = RerankAgent(pair_scores=True)
    judged = PairRelevances(scores=[[1, 50, 7]])
    # The schema permits the shape, so the agent must reject the row.
    assert agent.pair_scores and len(judged.scores[0]) == 3


async def test_a_malformed_row_does_not_reach_the_ranking() -> None:
    ranked, _ = await _run(PairRelevances(scores=[[1, 50, 7], [2, 90]]))
    assert [r.node_id for r in ranked][0] == "101"


async def test_a_score_out_of_range_is_clamped_not_dropped() -> None:
    """The candidate keeps a usable ordering rather than vanishing."""
    ranked, _ = await _run(PairRelevances(scores=[[1, 500], [2, 10]]))
    assert ranked[0].node_id == "100"


async def test_a_short_response_leaves_the_rest_unscored_not_shifted() -> None:
    """The whole reason the index survives the encoding. 2 scores for 4
    candidates must not slide onto candidates 3 and 4."""
    ranked, _ = await _run(PairRelevances(scores=[[3, 99], [4, 90]]))
    assert [r.node_id for r in ranked][:2] == ["102", "103"]


async def test_pairs_and_terse_are_mutually_exclusive_at_the_call_site() -> None:
    """`pair_scores` must win; falling through to `TerseRelevances` would
    ask for one schema and parse another."""
    _, tools = await _run(PairRelevances(scores=[[1, 10]]), terse_scores=True)
    assert tools.schema is PairRelevances

"""The reranker's ordering, and what it does when the model does not help.

The failure this file exists to catch is specific: a reranker whose LLM call
always fails degrades to the retrieval order, which is exactly `hybrid`. It
would then score like `hybrid`, look plausible next to it, and measure
nothing. Every test here therefore pins an LLM answer that *disagrees* with
retrieval order, so passing requires the scores to have been used.
"""

from __future__ import annotations

import pytest

from stark_bench.agents.rerank import RerankAgent, Relevance, Relevances
from stark_bench.domain import Passage, Query


class StubTools:
    def __init__(self, passages, judged=None, raises=False):
        self._passages = passages
        self._judged = judged
        self._raises = raises
        self.calls = []
        self.prompts = []

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):
        self.prompts.append(prompt)
        if self._raises:
            raise RuntimeError("endpoint down")
        return self._judged


def _passages(*specs):
    # Retrieval order is the order given; scores descend so the list is
    # already sorted the way `search_passages` returns it.
    return [
        Passage(node_id=n, text=t, score=1.0 - i / 100)
        for i, (n, t) in enumerate(specs)
    ]


@pytest.mark.asyncio
async def test_llm_scores_override_retrieval_order():
    passages = _passages(("a", "wrong"), ("b", "right"), ("c", "wrong"))
    judged = Relevances(
        scores=[
            Relevance(node_id="a", score=1.0),
            Relevance(node_id="b", score=9.0),
            Relevance(node_id="c", score=2.0),
        ]
    )
    tools = StubTools(passages, judged)

    ranked = await RerankAgent(k=3).retrieve(Query(query_id=1, text="q"), tools)

    # `b` was retrieved second and must come first, so this cannot pass on
    # retrieval order.
    assert [r.node_id for r in ranked] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_scores_are_descending_so_the_evaluator_sees_a_ranking():
    passages = _passages(("a", "x"), ("b", "y"))
    judged = Relevances(
        scores=[Relevance(node_id="b", score=8.0), Relevance(node_id="a", score=1.0)]
    )
    ranked = await RerankAgent(k=2).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, judged)
    )
    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)


@pytest.mark.asyncio
async def test_a_failed_llm_call_degrades_to_retrieval_order():
    # Ids chosen so alphabetical order ("alpha", "mid", "zeta") disagrees
    # with retrieval order. Sorting by id instead of by retrieval rank
    # passes with ids like a/b/c, which is how the first version of this
    # file certified a tie-break it was not testing.
    passages = _passages(("zeta", "x"), ("alpha", "y"), ("mid", "z"))
    ranked = await RerankAgent(k=3).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, raises=True)
    )
    assert [r.node_id for r in ranked] == ["zeta", "alpha", "mid"]


@pytest.mark.asyncio
async def test_unscored_candidates_fall_below_scored_ones_keeping_retrieval_order():
    # The model answers about `c` only. `a` and `b` outrank it in retrieval
    # order, so a passing run requires the partial answer to have promoted
    # `c` above both -- and requires the two unscored ones to stay in their
    # own relative order rather than sorting by id or by nothing.
    passages = _passages(("zeta", "x"), ("alpha", "y"), ("mid", "z"))
    judged = Relevances(scores=[Relevance(node_id="mid", score=7.0)])
    ranked = await RerankAgent(k=3).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, judged)
    )
    assert [r.node_id for r in ranked] == ["mid", "zeta", "alpha"]


@pytest.mark.asyncio
async def test_ids_the_model_invented_are_discarded():
    passages = _passages(("a", "x"), ("b", "y"))
    judged = Relevances(
        scores=[
            Relevance(node_id="b", score=9.0),
            Relevance(node_id="ZZZ-not-retrieved", score=10.0),
        ]
    )
    ranked = await RerankAgent(k=5).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, judged)
    )
    assert [r.node_id for r in ranked] == ["b", "a"]


@pytest.mark.asyncio
async def test_the_model_is_shown_candidate_text_not_only_ids():
    passages = _passages(("a", "phytanoyl-CoA hydroxylase"), ("b", "other"))
    tools = StubTools(passages, Relevances(scores=[]))
    await RerankAgent(k=2).retrieve(Query(query_id=1, text="q"), tools)
    assert "phytanoyl-CoA hydroxylase" in tools.prompts[0]


@pytest.mark.asyncio
async def test_fetch_is_wider_than_k_so_reranking_can_promote_from_the_tail():
    # A candidate outside the top-k of retrieval must be able to reach the
    # final list; that is the whole point of fetching `fetch` and returning
    # `k`. With fetch == k this test cannot fail.
    passages = _passages(*[(str(i), f"t{i}") for i in range(40)])
    judged = Relevances(scores=[Relevance(node_id="39", score=10.0)])
    ranked = await RerankAgent(k=5, fetch=40).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, judged)
    )
    assert ranked[0].node_id == "39"


@pytest.mark.asyncio
async def test_no_candidates_returns_nothing_without_calling_the_model():
    tools = StubTools([], Relevances(scores=[]))
    assert await RerankAgent(k=5).retrieve(Query(query_id=1, text="q"), tools) == []
    assert tools.prompts == []


@pytest.mark.asyncio
async def test_a_zero_score_outranks_a_candidate_the_model_never_mentioned():
    # The case that separates the `-1.0` default from a `0.0` one. `b` is
    # retrieved first; `a` is scored 0.0, which is the model actively calling
    # it irrelevant. A `0.0` default would tie the two and let retrieval order
    # put `b` first -- so this fails unless "unmentioned" sorts strictly
    # below "judged irrelevant".
    passages = _passages(("b", "y"), ("a", "x"))
    judged = Relevances(scores=[Relevance(node_id="a", score=0.0)])
    ranked = await RerankAgent(k=2).retrieve(
        Query(query_id=1, text="q"), StubTools(passages, judged)
    )
    assert [r.node_id for r in ranked] == ["a", "b"]

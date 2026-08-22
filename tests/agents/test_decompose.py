"""The parts of `DecomposeAgent` that would fail without raising.

The unify step is an LLM judgment, not arithmetic, and that is the property
this file exists to defend. The first version summed reciprocal ranks
across the sub-query result lists, which promotes a candidate for being
found by several searches -- including several **tangential** ones. It
scored 0.37127 mrr and lifted recall@20 by 0.010 over `hybrid` where a
plain reranker over one list gained 0.038 by reordering alone.

So the pool must carry retrieval *evidence* to the model and must not
pre-decide with it.
"""

from __future__ import annotations

from stark_bench.agents.decompose import Candidate, DecomposeAgent, _pool
from stark_bench.domain import Passage


def _passage(node_id: str) -> Passage:
    return Passage(node_id=node_id, score=1.0, text=f"- name: {node_id}\n")


def _candidate(node_id: str, matches: dict[int, int]) -> Candidate:
    return Candidate(passage=_passage(node_id), matches=matches)


def test_being_found_by_more_searches_does_not_promote_a_candidate() -> None:
    """The whole point of the rewrite.

    `many` is found by two searches, at ranks 5 and 6. `one` is found by a
    single search, at rank 1. Additive fusion ranks `many` above `one`
    (2/(60+5) + ... beats 1/(60+1)); a union ordered by best rank does not,
    and the model -- which sees both `(found by ...)` annotations -- makes
    the call instead.

    Asserts relative order rather than first place: other searches' own
    rank-1 hits are legitimately at the front, and pinning `pooled[0]`
    would be testing the filler.
    """
    a = [_passage(f"a{i}") for i in range(4)] + [_passage("many")]
    b = [_passage(f"b{i}") for i in range(5)] + [_passage("many")]
    c = [_passage("one")]
    pooled = [cand.passage.node_id for cand in _pool([a, b, c], limit=20)]
    assert pooled.index("one") < pooled.index("many"), (
        "a candidate found by two searches at ranks 5 and 6 outranked one "
        "found by a single search at rank 1 -- that is the additive bias "
        "the LLM unify step is supposed to replace"
    )


def test_the_pool_records_which_search_found_each_candidate_and_where() -> None:
    """A count would lose the distinction that makes a match tangential."""
    a = [_passage("shared"), _passage("only_a")]
    b = [_passage("only_b"), _passage("shared")]
    by_id = {c.passage.node_id: c for c in _pool([a, b], limit=10)}
    assert by_id["shared"].matches == {0: 1, 1: 2}
    assert by_id["only_a"].matches == {0: 2}
    assert by_id["only_b"].matches == {1: 1}


def test_the_pool_keeps_every_candidate_exactly_once() -> None:
    a = [_passage("x"), _passage("y")]
    b = [_passage("y"), _passage("z")]
    pooled = [c.passage.node_id for c in _pool([a, b], limit=10)]
    assert sorted(pooled) == ["x", "y", "z"]
    assert len(pooled) == len(set(pooled))


def test_a_single_list_pools_to_exactly_that_order() -> None:
    """With no usable decomposition this must degrade to `hybrid`."""
    only = [_passage("a"), _passage("b"), _passage("c")]
    assert [c.passage.node_id for c in _pool([only], limit=10)] == ["a", "b", "c"]


def test_ties_on_best_rank_break_toward_the_original_query() -> None:
    """Search 0 is always the original, and it settles an exact tie.

    Both candidates are rank 1 of their own search, so best rank cannot
    separate them. The original query's own hit wins, which is what makes
    an empty or useless decomposition degrade to `hybrid` rather than to
    something arbitrary.

    Note this applies ONLY at equal best rank: a sub-query's rank-1 hit
    does outrank the original's rank-2 hit, deliberately -- reaching those
    is why the query was decomposed.
    """
    original = [_passage("from_original")]
    sub = [_passage("from_sub")]
    pooled = [c.passage.node_id for c in _pool([original, sub], limit=10)]
    assert pooled == [
        "from_original",
        "from_sub",
    ], f"tie at best rank resolved to {pooled}, not toward the original query"


def test_a_sub_query_can_outrank_the_original_when_it_ranks_higher() -> None:
    """Reaching what the original ranked poorly is the reason to decompose."""
    original = [_passage("filler"), _passage("original_second")]
    sub = [_passage("sub_first")]
    pooled = [c.passage.node_id for c in _pool([original, sub], limit=10)]
    assert pooled.index("sub_first") < pooled.index("original_second")


def test_the_pool_is_truncated_to_the_limit() -> None:
    lists = [[_passage(f"n{i}") for i in range(50)]]
    assert len(_pool(lists, limit=12)) == 12


def test_the_pool_is_wider_than_any_single_search() -> None:
    """A pool capped at one list's depth can only rearrange, never widen."""
    assert DecomposeAgent().fetch > DecomposeAgent().per_query_fetch


def test_unscored_candidates_are_kept_and_sort_below_scored_ones() -> None:
    """Returning only what the model mentioned would throw away recall@20."""
    agent = DecomposeAgent(k=5)
    candidates = [_candidate(f"n{i}", {0: i + 1}) for i in range(5)]
    ranked = agent._rank(candidates, {3: 90.0})
    assert len(ranked) == 5, "backfill dropped candidates the model did not score"
    assert ranked[0].node_id == "n2"
    assert [r.node_id for r in ranked[1:]] == ["n0", "n1", "n3", "n4"]


def test_an_unmentioned_candidate_is_not_confused_with_a_rejected_one() -> None:
    """`-1.0`, not `0.0`: judged irrelevant and never seen differ."""
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_candidate("a", {0: 1}), _candidate("b", {0: 2})], {1: 0.0})
    scores = {r.node_id: r.score for r in ranked}
    assert scores["a"] == 0.0
    assert scores["b"] == -1.0


def test_no_scores_at_all_falls_back_to_pooled_order() -> None:
    agent = DecomposeAgent(k=3)
    candidates = [_candidate(n, {0: i + 1}) for i, n in enumerate("abc")]
    assert [r.node_id for r in agent._rank(candidates, None)] == ["a", "b", "c"]


def test_a_hallucinated_index_cannot_promote_a_candidate() -> None:
    agent = DecomposeAgent(k=3)
    ranked = agent._rank(
        [_candidate("a", {0: 1}), _candidate("b", {0: 2})], {99: 100.0}
    )
    assert [r.node_id for r in ranked] == ["a", "b"]
    assert all(r.score == -1.0 for r in ranked)


def test_the_rendered_candidates_carry_their_retrieval_evidence() -> None:
    """`(found by ...)` is the input the unify prompt reasons over."""
    agent = DecomposeAgent()
    rendered = agent._render(
        [_candidate("aspirin", {0: 3, 2: 1})], ["aspirin", "x", "y"]
    )
    assert "[1]" in rendered
    assert "aspirin" in rendered
    assert (
        "0@3" in rendered and "2@1" in rendered
    ), f"retrieval evidence missing from the prompt: {rendered!r}"


def test_the_scoring_prompt_warns_that_a_search_may_be_tangential() -> None:
    """Without this the model counts matches, which is what was replaced."""
    from stark_bench.agents.decompose import _SCORE_PROMPT

    # The whole clause, not the word: a bare `"tangential" in prompt`
    # survived deleting the sentence that carries the instruction, because
    # the word recurs in the next fragment.
    assert "A search may be tangential" in _SCORE_PROMPT
    assert "Being retrieved by it does not make a candidate an answer" in _SCORE_PROMPT
    assert "EVIDENCE, not as a score" in _SCORE_PROMPT
    assert "found by only one search may still be the best answer" in _SCORE_PROMPT

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


class _Query:
    """Just the `query_id` and `text` the agent reads."""

    def __init__(self, query_id: int) -> None:
        self.query_id = query_id
        self.text = "a query"


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


def test_a_candidate_only_a_supplemental_search_found_can_reach_the_pool() -> None:
    """The union's whole purpose, and it survives narrowing fetch to 40.

    `fetch` was 100 and is now 40, because widening 40 -> 100 bought
    +0.001 recall@20. That is a size change, not an abandonment: a
    candidate the original query never returned still reaches the prompt
    if a supplemental search ranked it well.
    """
    original = [_passage(f"o{i}") for i in range(5)]
    sub = [_passage("only_sub")]
    pooled = [c.passage.node_id for c in _pool([original, sub], limit=6)]
    assert (
        "only_sub" in pooled
    ), "a candidate found only by a supplemental search did not reach the pool"


def test_unnamed_candidates_are_kept_behind_the_ones_the_model_chose() -> None:
    """The model may return fewer than 20; recall@20 needs all of them.

    Backfill is the difference between an arm that scores what it ranked
    and one whose recall collapses while MRR looks fine.
    """
    agent = DecomposeAgent(k=5)
    candidates = [_candidate(f"n{i}", {0: i + 1}) for i in range(5)]
    ranked = agent._rank(candidates, [3])
    assert len(ranked) == 5, "backfill dropped candidates the model did not name"
    assert ranked[0].node_id == "n2", "the model's choice did not rank first"
    assert [r.node_id for r in ranked[1:]] == [
        "n0",
        "n1",
        "n3",
        "n4",
    ], "backfilled candidates lost the pool's own order"


def test_a_candidate_the_model_never_named_is_marked_as_such() -> None:
    """`-1.0` for backfill: declined-to-name and ranked-last differ."""
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_candidate("a", {0: 1}), _candidate("b", {0: 2})], [1])
    scores = {r.node_id: r.score for r in ranked}
    assert scores["a"] > 0.0, "a chosen candidate must carry a positive score"
    assert scores["b"] == -1.0


def test_the_models_order_is_preserved_by_the_scores() -> None:
    """Scores are positional; they exist to keep `Ranked` sortable."""
    agent = DecomposeAgent(k=4)
    candidates = [_candidate(n, {0: i + 1}) for i, n in enumerate("abcd")]
    ranked = agent._rank(candidates, [3, 1])
    assert [r.node_id for r in ranked[:2]] == ["c", "a"]
    assert ranked[0].score > ranked[1].score, "the ordering was not preserved"


def test_no_ordering_at_all_falls_back_to_pooled_order() -> None:
    agent = DecomposeAgent(k=3)
    candidates = [_candidate(n, {0: i + 1}) for i, n in enumerate("abc")]
    assert [r.node_id for r in agent._rank(candidates, None)] == ["a", "b", "c"]


def test_a_repeated_index_does_not_displace_a_real_candidate() -> None:
    """A model naming an index twice meant it once.

    Honouring the repeat would push a genuine candidate off the end of
    `k` -- the failure is a shorter answer, not a crash.
    """
    agent = DecomposeAgent(k=3)
    candidates = [_candidate(n, {0: i + 1}) for i, n in enumerate("abc")]
    ranked = agent._rank(candidates, [2, 2])
    assert [r.node_id for r in ranked] == ["b", "a", "c"]
    assert len({r.node_id for r in ranked}) == 3


def test_a_hallucinated_index_is_dropped_before_it_reaches_the_ranking() -> None:
    """Exercises the real filter in `_order`, not a restatement of it.

    A model naming index 99 of 2 candidates must not silently shift the
    ranking, and must not raise either -- one bad response should not lose
    the query.
    """
    import asyncio

    from stark_bench.agents.decompose import Ordering

    class _Tools:
        async def extract(self, prompt, schema):  # noqa: ANN001, ANN202, ARG002
            return Ordering(indexes=[99, 2, 99, 1])

    agent = DecomposeAgent(k=3)
    ordering = asyncio.run(
        agent._order(_Query(1), "groups", 2, _Tools())  # type: ignore[arg-type]
    )
    assert ordering == [2, 1], f"filter let something through: {ordering}"


def test_an_ordering_of_only_bad_indexes_degrades_rather_than_ranking_noise() -> None:
    """All-invalid is a degradation and must be greppable as one."""
    import asyncio

    from stark_bench.agents.decompose import Ordering

    class _Tools:
        async def extract(self, prompt, schema):  # noqa: ANN001, ANN202, ARG002
            return Ordering(indexes=[99, 100])

    agent = DecomposeAgent(k=3)
    ordering = asyncio.run(
        agent._order(_Query(1), "groups", 2, _Tools())  # type: ignore[arg-type]
    )
    assert ordering is None, "all-invalid indexes must fall back, not rank noise"


def test_rank_ignores_an_out_of_range_index_without_raising() -> None:
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_candidate("a", {0: 1}), _candidate("b", {0: 2})], [99])
    assert [r.node_id for r in ranked] == ["a", "b"]
    assert all(r.score == -1.0 for r in ranked)


def test_candidates_are_grouped_under_the_search_that_found_them() -> None:
    """The structure is stated, not left for the model to reconstruct."""
    agent = DecomposeAgent()
    rendered = agent._render(
        [_candidate("from_query", {0: 1}), _candidate("from_sub", {2: 1})],
        ["the whole query", "sub one", "sub two"],
    )
    assert "THE QUERY ITSELF found:" in rendered
    assert "Supplemental search (2)" in rendered
    assert rendered.index("from_query") < rendered.index(
        "Supplemental search (2)"
    ), "the original query's results must come first"


def test_a_candidate_found_twice_is_listed_once_with_the_other_search_noted() -> None:
    """Repeating it would make one entity look like several."""
    agent = DecomposeAgent()
    rendered = agent._render([_candidate("both", {0: 4, 1: 1})], ["q", "sub"])
    assert rendered.count("both") == 1, "a candidate was listed under two groups"
    assert "also found by [0]" in rendered


def test_the_ordering_prompt_warns_that_a_search_may_be_tangential() -> None:
    """Without this the model treats every group as equally load-bearing.

    Asserts whole clauses, not keywords: a bare `"tangential" in prompt`
    survived deleting the sentence carrying the instruction, because the
    word recurs elsewhere.
    """
    from stark_bench.agents.decompose import _ORDER_PROMPT

    assert "a supplemental search may be TANGENTIAL" in _ORDER_PROMPT
    assert "being found by one is not evidence" in _ORDER_PROMPT
    assert "Judge every candidate against the QUERY above" in _ORDER_PROMPT
    assert "may still be the best answer" in _ORDER_PROMPT


def test_the_prompt_asks_for_an_order_and_not_for_scores() -> None:
    """Scoring every candidate is the task that produced degenerate output.

    The `matrix` encoding asked for a number per dimension and logged 431
    warnings that the model gave the same one every time. `k` is 20 and the
    metric scores an ordering, so an ordering is what to ask for.
    """
    from stark_bench.agents.decompose import Ordering, _ORDER_PROMPT

    assert "most relevant first" in _ORDER_PROMPT
    assert "at most 20" in _ORDER_PROMPT
    assert (
        "score" not in _ORDER_PROMPT.lower()
    ), "the prompt still mentions scoring; it must ask only for an order"
    assert list(Ordering.model_fields) == [
        "indexes"
    ], f"the schema is not an ordering: {list(Ordering.model_fields)}"


def test_the_rephrase_prompt_asks_for_whole_questions_not_pieces() -> None:
    """The distinction is the entire point of the mode.

    A decomposed sub-query asks part of the question and can retrieve
    tangentially; a paraphrase asks all of it, so every hit is on-topic and
    the unify step has a much easier job.
    """
    from stark_bench.agents.decompose import _REPHRASE_PROMPT

    assert "THE SAME question, complete -- not a piece of it" in _REPHRASE_PROMPT
    assert "Keep every constraint" in _REPHRASE_PROMPT


def test_both_planning_prompts_demand_verbatim_entity_names() -> None:
    """The measured gain is lexical; a paraphrased name destroys it.

    A rewrite of `Chronic myeloid leukemia` to "chronic blood cancer of the
    myeloid line" reads as a good paraphrase and silently drops the arm to
    dense-only, with every count in the report clean.
    """
    from stark_bench.agents.decompose import _DECOMPOSE_PROMPT, _REPHRASE_PROMPT

    for prompt in (_DECOMPOSE_PROMPT, _REPHRASE_PROMPT):
        assert "EXACTLY" in prompt or "EXACTLY as they appear" in prompt
        assert "lexical" in prompt


def test_rephrase_is_off_by_default_so_the_measured_arm_is_unchanged() -> None:
    assert DecomposeAgent().rephrase is False

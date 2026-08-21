"""The parts of `DecomposeAgent` that would fail without raising.

Three properties carry the design, and each has a failure mode that scores
plausibly rather than crashing:

- **fusion rewards multi-list appearance**, which is what makes a
  conjunctive query work. Broken, it degrades to whichever list happened to
  be first -- a valid ranking, slightly worse.
- **the backfill fills k**, or recall@20 collapses while MRR looks fine.
- **the original query always participates**, which is what bounds a bad
  decomposition below by `hybrid`.
"""

from __future__ import annotations

from stark_bench.agents.decompose import DecomposeAgent, _fuse
from stark_bench.domain import Passage


def _passage(node_id: str) -> Passage:
    return Passage(node_id=node_id, score=1.0, text=f"- name: {node_id}\n")


def test_a_candidate_found_by_two_sub_queries_outranks_one_found_by_one() -> None:
    """The conjunction, done arithmetically.

    `both` is rank 2 in each list and `only_a` is rank 1 in one. Without
    accumulation `only_a` wins, which is the wrong answer for "X and Y" --
    and is exactly what a sum-less fusion returns.
    """
    a = [_passage("only_a"), _passage("both")]
    b = [_passage("only_b"), _passage("both")]
    fused = [p.node_id for p in _fuse([a, b])]
    assert fused[0] == "both", (
        f"fusion ranked {fused[0]!r} first; a candidate satisfying both "
        "constraints must outrank one satisfying a single constraint"
    )


def test_fusion_keeps_every_candidate_exactly_once() -> None:
    a = [_passage("x"), _passage("y")]
    b = [_passage("y"), _passage("z")]
    fused = [p.node_id for p in _fuse([a, b])]
    assert sorted(fused) == ["x", "y", "z"]
    assert len(fused) == len(set(fused)), "a candidate was emitted twice"


def test_fusion_of_one_list_preserves_its_order() -> None:
    """With a single list this arm must degrade exactly to `hybrid`."""
    only = [_passage("a"), _passage("b"), _passage("c")]
    assert [p.node_id for p in _fuse([only])] == ["a", "b", "c"]


def test_unscored_candidates_are_kept_and_sort_below_scored_ones() -> None:
    """Returning only what the LLM mentioned would throw away recall@20."""
    agent = DecomposeAgent(k=5)
    candidates = [_passage(f"n{i}") for i in range(5)]
    ranked = agent._rank(candidates, {3: 90.0})
    assert len(ranked) == 5, "backfill dropped candidates the LLM did not score"
    assert ranked[0].node_id == "n2", "the scored candidate did not rank first"
    assert [r.node_id for r in ranked[1:]] == [
        "n0",
        "n1",
        "n3",
        "n4",
    ], "unscored candidates lost their fused retrieval order"


def test_an_unmentioned_candidate_is_not_confused_with_a_rejected_one() -> None:
    """`-1.0`, not `0.0`: judged irrelevant and never seen differ."""
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_passage("a"), _passage("b")], {1: 0.0})
    scores = {r.node_id: r.score for r in ranked}
    assert scores["a"] == 0.0
    assert scores["b"] == -1.0


def test_no_scores_at_all_falls_back_to_fused_order() -> None:
    """A failed scoring call must leave the fused ranking, not nothing."""
    agent = DecomposeAgent(k=3)
    candidates = [_passage("a"), _passage("b"), _passage("c")]
    assert [r.node_id for r in agent._rank(candidates, None)] == ["a", "b", "c"]


def test_the_ranking_is_truncated_to_k() -> None:
    agent = DecomposeAgent(k=2)
    assert len(agent._rank([_passage(f"n{i}") for i in range(9)], {})) == 2


def test_a_hallucinated_index_cannot_promote_a_candidate() -> None:
    """`_rank` is bounded by the candidate list, not by what the model sent.

    This held under a deliberate break of the range filter in `_score`,
    which is how it was discovered that the filter is redundant: `_rank`
    only ever looks up `1..len(candidates)`, so an out-of-range key is
    unreachable whatever `_score` passes through. The guarantee is real and
    worth pinning; the filter is not what provides it.
    """
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_passage("a"), _passage("b")], {99: 100.0})
    assert [r.node_id for r in ranked] == ["a", "b"]
    assert all(r.score == -1.0 for r in ranked)


def test_a_duplicated_index_does_not_produce_a_duplicate_candidate() -> None:
    """Models repeat indices; the ranking must still name each node once."""
    agent = DecomposeAgent(k=3)
    ranked = agent._rank([_passage("a"), _passage("b")], {1: 90.0, 2: 10.0})
    assert [r.node_id for r in ranked] == ["a", "b"]
    assert len({r.node_id for r in ranked}) == 2


def test_the_rendered_candidates_carry_titles() -> None:
    agent = DecomposeAgent()
    rendered = agent._render([_passage("aspirin")], ["aspirin"])
    assert "[1]" in rendered
    assert "aspirin" in rendered

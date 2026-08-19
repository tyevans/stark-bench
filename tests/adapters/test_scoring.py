import pytest

from stark_bench.adapters.stark_scorer import score_predictions
from stark_bench.domain import Ranked


@pytest.mark.integration
def test_a_perfect_agent_scores_one():
    predictions = {1: [Ranked("6", 1.0), Ranked("2", 0.1)]}
    answers = {1: ["6"]}
    metrics = score_predictions(
        predictions,
        answers,
        candidate_ids=list(range(1, 1000)),
        metrics=["hit@1", "mrr"],
    )
    assert metrics["hit@1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


@pytest.mark.integration
def test_a_useless_agent_scores_zero():
    """Without this, a scoring path that returns 1.0 unconditionally passes."""
    predictions = {1: [Ranked("999", 1.0), Ranked("998", 0.5)]}
    answers = {1: ["6"]}
    metrics = score_predictions(
        predictions,
        answers,
        candidate_ids=list(range(1, 1000)),
        metrics=["hit@1", "mrr"],
    )
    assert metrics["hit@1"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.0)


@pytest.mark.integration
def test_rank_order_matters():
    """A right answer in second place must not score like first place."""
    candidates = list(range(1, 1000))
    first = score_predictions(
        {1: [Ranked("6", 1.0), Ranked("9", 0.5)]},
        {1: ["6"]},
        candidate_ids=candidates,
        metrics=["mrr"],
    )
    second = score_predictions(
        {1: [Ranked("9", 1.0), Ranked("6", 0.5)]},
        {1: ["6"]},
        candidate_ids=candidates,
        metrics=["mrr"],
    )
    assert first["mrr"] > second["mrr"]


class TestEmptyPredictions:
    """STaRK's evaluator does `min(pred)` per query and cannot take an empty one.

    Left unguarded it dies inside a 3.11 subprocess with
    `ValueError: min() arg is an empty sequence`, naming no query and no
    cause, at the end of a run that has already done all its retrieval.
    """

    def test_one_empty_query_names_that_query(self):
        with pytest.raises(ValueError, match=r"\b7\b") as caught:
            score_predictions(
                {1: [Ranked("6", 1.0)], 7: []},
                {1: ["6"], 7: ["9"]},
                candidate_ids=list(range(1, 1000)),
            )
        assert "1 of 2" in str(caught.value)

    def test_every_query_empty_says_the_corpus_is_the_suspect(self):
        with pytest.raises(ValueError, match="2 of 2") as caught:
            score_predictions(
                {1: [], 7: []},
                {1: ["6"], 7: ["9"]},
                candidate_ids=list(range(1, 1000)),
            )
        assert "every query" in str(caught.value)

import pytest

from stark_bench.harness.aggregate import aggregate


def test_max_takes_the_best_chunk_of_a_node():
    """Scores differ on purpose: equal scores make max, mean and first agree."""
    result = aggregate([("A", 0.2), ("A", 0.9), ("B", 0.5)], strategy="max")
    assert [(r.node_id, r.score) for r in result] == [("A", 0.9), ("B", 0.5)]


def test_mean_is_a_different_answer_on_the_same_input():
    result = aggregate([("A", 0.2), ("A", 0.9), ("B", 0.5)], strategy="mean")
    by_id = {r.node_id: r.score for r in result}
    assert by_id["A"] == pytest.approx(0.55)
    assert by_id["B"] == pytest.approx(0.5)


def test_results_are_ordered_best_first():
    result = aggregate([("A", 0.1), ("B", 0.7), ("C", 0.4)], strategy="max")
    assert [r.node_id for r in result] == ["B", "C", "A"]


def test_ties_break_deterministically_by_node_id():
    """Two runs must rank identically, or a metric moves for no reason."""
    first = aggregate([("B", 0.5), ("A", 0.5)], strategy="max")
    second = aggregate([("A", 0.5), ("B", 0.5)], strategy="max")
    assert [r.node_id for r in first] == [r.node_id for r in second] == ["A", "B"]


def test_an_unknown_strategy_raises():
    with pytest.raises(KeyError):
        aggregate([("A", 1.0)], strategy="whatever-scores-best")

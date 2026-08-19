import pytest

from stark_bench.domain.budget import Budget, BudgetExhausted


def test_it_permits_calls_up_to_the_cap():
    budget = Budget(max_tool_calls=2, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    budget.spend_tool()
    with pytest.raises(BudgetExhausted):
        budget.spend_tool()


def test_tool_and_llm_budgets_are_separate():
    """One counter for both would let a cheap tool loop starve the LLM."""
    budget = Budget(max_tool_calls=1, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    budget.spend_llm()


def test_exhaustion_is_recorded_not_merely_raised():
    budget = Budget(max_tool_calls=1, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    with pytest.raises(BudgetExhausted):
        budget.spend_tool()
    assert budget.exhausted is True

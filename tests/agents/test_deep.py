import pytest

from stark_bench.agents.deep import _MAX_PROMPT_CHARS, DeepAgent
from stark_bench.harness.budget import Budget
from stark_bench.domain import Query, Ranked, ToolCall


class Tools:
    def __init__(self):
        self.calls: list[ToolCall] = []
        self.searches = 0

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.searches += 1
        return [Ranked("1", 0.9)]

    async def get_node(self, node_id):
        return {"node_id": node_id, "name": "n", "node_type": "t"}

    async def neighbors(self, node_id, *, depth=1):
        return ["2", "3"]

    async def get_relationships(self, node_id):
        return [("1", "targets", "2")]

    async def extract(self, prompt, schema):
        return schema(action="search", argument="more terms")


@pytest.mark.asyncio
async def test_it_returns_best_so_far_when_the_budget_runs_out():
    """Budget exhaustion is a recorded outcome, not an exception that voids
    the run: an agent is scored on what it had at the cap."""
    tools = Tools()
    agent = DeepAgent(
        k=20, budget=Budget(max_tool_calls=3, max_llm_calls=2, max_seconds=30.0)
    )
    result = await agent.retrieve(Query(1, "x"), tools)
    assert result
    assert tools.searches <= 3


@pytest.mark.asyncio
async def test_it_terminates_even_when_the_llm_always_asks_for_more():
    """A loop whose exit depends on model output must be hard-bounded.

    A test that hangs is worse than one that fails: in CI it reads as
    infrastructure trouble and gets retried rather than investigated.
    """
    tools = Tools()
    agent = DeepAgent(
        k=20, budget=Budget(max_tool_calls=5, max_llm_calls=5, max_seconds=30.0)
    )
    result = await agent.retrieve(Query(1, "x"), tools)
    assert isinstance(result, list)


def test_a_single_observation_larger_than_the_budget_is_hard_truncated():
    """The 'keep at least the most recent observation' fallback must not let
    one oversized observation ride through the bound uncapped -- that would
    defeat the context cap in exactly the case where it matters most."""
    oversized = "x" * (_MAX_PROMPT_CHARS * 3)
    kept = DeepAgent._truncate([oversized])
    assert len(kept) == 1
    assert len(kept[0]) <= _MAX_PROMPT_CHARS


def test_the_most_recent_observation_is_still_kept_when_earlier_ones_are_dropped():
    small = "a" * 100
    huge = "b" * (_MAX_PROMPT_CHARS * 2)
    kept = DeepAgent._truncate([small, small, huge])
    assert len(kept) == 1
    assert kept[0].startswith("b")
    assert len(kept[0]) <= _MAX_PROMPT_CHARS

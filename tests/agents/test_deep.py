import pytest

from stark_bench.agents.deep import _MAX_PROMPT_CHARS, DeepAgent
from stark_bench.domain.budget import Budget
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


class RealisticScoreTools:
    """Search scores on the scale the retriever actually returns.

    redstring's hybrid channel is an RRF-style fusion, so its scores sit near
    1/(60+rank) -- measured 0.0137..0.0325 on native-wholedoc. Every other
    fixture in this file uses 0.9, which is above any constant the agent
    assigns to a traversal hit, so those fixtures cannot tell a
    relevance-ordered pool from one dominated by traversal constants. This
    one can: `hub` has many neighbours and `gold` is the only search hit.
    """

    def __init__(self, neighbour_count: int = 30):
        self.neighbours = [f"n{i}" for i in range(neighbour_count)]
        self._steps = iter([("search", "terms"), ("neighbors", "hub"), ("finish", "")])

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        return [Ranked("gold", 0.0325)]

    async def get_node(self, node_id):
        return {"node_id": node_id, "name": "n", "node_type": "t"}

    async def neighbors(self, node_id, *, depth=1):
        return list(self.neighbours)

    async def get_relationships(self, node_id):
        return [("hub", "targets", n) for n in self.neighbours]

    async def extract(self, prompt, schema):
        try:
            action, argument = next(self._steps)
        except StopIteration:
            action, argument = "finish", ""
        return schema(action=action, argument=argument)


@pytest.mark.asyncio
async def test_a_search_hit_outranks_traversal_candidates():
    """Retrieval evidence must not be buried by traversal evidence.

    The two are different quantities. Ranking them by magnitude means the
    traversal constant wins whenever it exceeds the retriever's scale, which
    for an RRF fusion is always -- see B-DEEP-SCORE-SCALE-1.
    """
    tools = RealisticScoreTools(neighbour_count=30)
    agent = DeepAgent(
        k=20, budget=Budget(max_tool_calls=8, max_llm_calls=8, max_seconds=30.0)
    )
    result = await agent.retrieve(Query(1, "x"), tools)

    assert result, "expected candidates"
    assert result[0].node_id == "gold", (
        f"search hit ranked {[r.node_id for r in result].index('gold')} "
        f"behind traversal candidates"
    )


@pytest.mark.asyncio
async def test_traversal_candidates_do_not_evict_search_hits_from_k():
    """A hub returning more neighbours than k must not fill the whole result.

    ABLIM1 in the real corpus has degree 426 against k=20.
    """
    tools = RealisticScoreTools(neighbour_count=200)
    agent = DeepAgent(
        k=20, budget=Budget(max_tool_calls=8, max_llm_calls=8, max_seconds=30.0)
    )
    result = await agent.retrieve(Query(1, "x"), tools)

    assert "gold" in [
        r.node_id for r in result
    ], "200 neighbours evicted the only search hit from the top 20"


class CorroborationTools:
    """`b` is reached from two different hops, `a` and `c` from one each.

    Ids are chosen so alphabetical order disagrees with corroboration order:
    if the tie-break were insertion order or the id, `a` would lead.
    """

    def __init__(self):
        self._steps = iter([("neighbors", "h1"), ("neighbors", "h2"), ("finish", "")])
        self._returns = iter([["a", "b"], ["b", "c"]])

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        return []

    async def get_node(self, node_id):
        return None

    async def neighbors(self, node_id, *, depth=1):
        return next(self._returns)

    async def get_relationships(self, node_id):
        return []

    async def extract(self, prompt, schema):
        try:
            action, argument = next(self._steps)
        except StopIteration:
            action, argument = "finish", ""
        return schema(action=action, argument=argument)


@pytest.mark.asyncio
async def test_traversal_only_candidates_rank_by_corroboration():
    """A node reached from several hops outranks one reached from one.

    Without this the traversal tail is ordered arbitrarily -- which is what
    a flat constant per neighbour gave (B-DEEP-SCORE-SCALE-1).
    """
    agent = DeepAgent(
        k=20, budget=Budget(max_tool_calls=8, max_llm_calls=8, max_seconds=30.0)
    )
    result = await agent.retrieve(Query(1, "x"), CorroborationTools())

    assert [r.node_id for r in result][0] == "b", [r.node_id for r in result]

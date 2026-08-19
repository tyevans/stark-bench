import pytest

from stark_bench.agents.dense import DenseAgent
from stark_bench.agents.hybrid import HybridAgent
from stark_bench.agents.lexical import LexicalAgent
from stark_bench.domain import Query, Ranked, ToolCall
from stark_bench.ports import Agent


class RecordingTools:
    def __init__(self):
        self.calls: list[ToolCall] = []
        self.modes: list[str] = []

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.modes.append(mode)
        return [Ranked("1", 0.9), Ranked("2", 0.4)]

    async def get_node(self, node_id):
        return None

    async def neighbors(self, node_id, *, depth=1):
        return []

    async def get_relationships(self, node_id):
        return []

    async def complete(self, prompt):
        raise AssertionError("baselines use no LLM")


@pytest.mark.asyncio
async def test_dense_uses_the_semantic_channel_only():
    tools = RecordingTools()
    result = await DenseAgent(k=20).retrieve(Query(1, "aspirin"), tools)
    assert tools.modes == ["semantic"]
    assert [r.node_id for r in result] == ["1", "2"]


@pytest.mark.asyncio
async def test_hybrid_uses_the_fused_channel():
    tools = RecordingTools()
    await HybridAgent(k=20).retrieve(Query(1, "aspirin"), tools)
    assert tools.modes == ["hybrid"]


@pytest.mark.asyncio
async def test_baselines_make_no_llm_call():
    """A baseline that quietly called an LLM would not be a baseline."""
    tools = RecordingTools()
    await DenseAgent(k=20).retrieve(Query(1, "x"), tools)
    await HybridAgent(k=20).retrieve(Query(1, "x"), tools)


def test_both_satisfy_the_agent_protocol():
    assert isinstance(DenseAgent(k=20), Agent)
    assert isinstance(HybridAgent(k=20), Agent)


@pytest.mark.asyncio
async def test_lexical_uses_the_lexical_channel_only():
    tools = RecordingTools()
    await LexicalAgent(k=20).retrieve(Query(1, "aspirin"), tools)
    assert tools.modes == ["lexical"]


@pytest.mark.asyncio
async def test_the_three_channels_are_mutually_distinct():
    """No two retrieval agents may run the same channel.

    The per-agent tests above each pin one mode, and together they still
    permit a copy-paste that this catches directly: the whole point of
    scoring all three is that `hybrid` is the fusion of the other two, so
    two agents sharing a channel would produce a decomposition that is
    arithmetic on a duplicated column.

    Written after a `LexicalAgent` whose body said `mode="hybrid"` passed
    the entire suite.
    """
    modes = []
    for agent in (DenseAgent(k=20), LexicalAgent(k=20), HybridAgent(k=20)):
        tools = RecordingTools()
        await agent.retrieve(Query(1, "aspirin"), tools)
        modes.extend(tools.modes)

    assert modes == ["semantic", "lexical", "hybrid"]
    assert len(set(modes)) == 3, f"two agents share a channel: {modes}"

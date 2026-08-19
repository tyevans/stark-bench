import pytest

from stark_bench.agents.zero_shot import ZeroShotAgent
from stark_bench.ports import Query, Ranked, ToolCall


class Tools:
    """A fake toolset. `extract` returns a populated schema instance, which is
    what redstring's `LlmProvider` contract actually provides."""

    def __init__(self, reply="cyclooxygenase inhibitor drug"):
        self.calls: list[ToolCall] = []
        self.reply = reply
        self.searched: list[str] = []
        self.prompts: list[str] = []

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.searched.append(text)
        return [Ranked("6", 0.9)]

    async def get_node(self, node_id):
        return None

    async def neighbors(self, node_id, *, depth=1):
        return []

    async def get_relationships(self, node_id):
        return []

    async def extract(self, prompt, schema):
        self.prompts.append(prompt)
        return schema(query=self.reply)


@pytest.mark.asyncio
async def test_it_searches_with_the_rewritten_query():
    tools = Tools()
    await ZeroShotAgent(k=20).retrieve(
        Query(1, "which COX-2 drug treats arthritis?"), tools
    )
    assert tools.searched == ["cyclooxygenase inhibitor drug"]


@pytest.mark.asyncio
async def test_it_makes_exactly_one_llm_call():
    """Fixed cost is the defining property of this architecture."""
    tools = Tools()
    await ZeroShotAgent(k=20).retrieve(Query(1, "x"), tools)
    assert len(tools.prompts) == 1


@pytest.mark.asyncio
async def test_an_empty_rewrite_falls_back_to_the_original_query():
    """A refusal or a blank completion must not become a blank search."""
    tools = Tools(reply="   ")
    await ZeroShotAgent(k=20).retrieve(Query(1, "original text"), tools)
    assert tools.searched == ["original text"]


@pytest.mark.asyncio
async def test_an_llm_failure_falls_back_rather_than_losing_the_query():
    class Failing(Tools):
        async def extract(self, prompt, schema):
            raise RuntimeError("endpoint down")

    tools = Failing()
    result = await ZeroShotAgent(k=20).retrieve(Query(1, "original text"), tools)
    assert tools.searched == ["original text"]
    assert result

"""The wiring that lets one config run all four architectures."""

from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import BaseModel
from redstring import TenantId
from redstring.llm.adapters.langchain import LangChainLlmProvider

from stark_bench.agents.zero_shot import ZeroShotAgent
from stark_bench.composition.agent_registry import (
    AGENTS,
    PerQueryDeepAgent,
    build_agent,
)
from stark_bench.harness.cli import (
    DEFAULT_CHAT_MODEL,
    _llm_for,
    report_path,
    toolset_for,
)
from stark_bench.domain.run_config import RunConfig
from stark_bench.domain import Query, Ranked, ToolCall


class _Schema(BaseModel):
    query: str


class _StubChunks:
    dimension = 4


class _StubEmbeddings:
    model = "stub"
    dimension = 4

    def embed(self, texts):
        return [[0.0] * 4 for _ in texts]


CONFIG = RunConfig(
    name="vss-control",
    dataset="prime",
    split="test-0.1",
    chunker="whole-document",
    embeddings="precomputed-ada002",
    dimension=1536,
    aggregation="max",
    agent="dense",
    k=20,
    raw="",
)


def test_all_four_architectures_are_reachable_by_name():
    """The two LLM agents shipped tested but unreachable from a config."""
    assert set(AGENTS) == {"dense", "hybrid", "zero_shot", "deep"}


@pytest.mark.parametrize(
    ("agent", "expected"),
    [("zero_shot", ZeroShotAgent), ("deep", PerQueryDeepAgent)],
)
def test_it_builds_the_llm_agents_with_the_configured_k(agent, expected):
    """`k` comes from the config, never from the agent's own default: a
    config asking for k=5 that silently got 20 would score differently."""
    built = build_agent(replace(CONFIG, agent=agent, k=5))
    assert isinstance(built, expected)
    assert built.k == 5


def test_an_unknown_agent_is_refused_rather_than_defaulted():
    with pytest.raises(NotImplementedError, match="nonesuch"):
        build_agent(replace(CONFIG, agent="nonesuch"))


def test_two_agents_on_one_config_write_to_different_files():
    """Four architectures share `vss-control`, so a path keyed on the config
    name alone would leave one file where four runs should be."""
    dense = report_path(replace(CONFIG, agent="dense"))
    deep = report_path(replace(CONFIG, agent="deep"))
    assert dense != deep
    assert dense.name == "vss-control.dense.json"
    assert deep.name == "vss-control.deep.json"


def test_the_llm_endpoint_defaults_to_the_module_constant():
    provider = _llm_for(CONFIG)
    assert isinstance(provider, LangChainLlmProvider)
    assert provider.model.endswith(DEFAULT_CHAT_MODEL)


def test_a_config_may_name_its_own_chat_model():
    provider = _llm_for(replace(CONFIG, chat_model="gemma-4-26b-qat"))
    assert provider.model.endswith("gemma-4-26b-qat")
    assert not provider.model.endswith(DEFAULT_CHAT_MODEL)


class _CountingTools:
    """Answers every `extract` with one more search, forever."""

    def __init__(self):
        self.calls: list[ToolCall] = []
        self.searches_per_query: list[int] = []
        self.searches = 0

    def next_query(self):
        self.searches_per_query.append(self.searches)
        self.searches = 0

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.searches += 1
        return [Ranked("1", 0.9)]

    async def get_node(self, node_id):
        return None

    async def neighbors(self, node_id, *, depth=1):
        return []

    async def get_relationships(self, node_id):
        return []

    async def extract(self, prompt, schema):
        return schema(action="search", argument="more")


async def test_the_deep_agent_gets_a_fresh_budget_for_every_query():
    """`runner.run` reuses one agent across the whole query set. A single
    shared `Budget` would be spent on query 1 and every query after it would
    retrieve nothing -- a near-zero score with no error to notice."""
    tools = _CountingTools()
    agent = PerQueryDeepAgent(k=20, max_tool_calls=3, max_llm_calls=3)

    first = await agent.retrieve(Query(1, "a"), tools)
    tools.next_query()
    second = await agent.retrieve(Query(2, "b"), tools)
    tools.next_query()

    assert tools.searches_per_query == [3, 3]
    assert first and second


async def test_exhaustion_is_counted_across_queries():
    tools = _CountingTools()
    agent = PerQueryDeepAgent(k=20, max_tool_calls=1, max_llm_calls=1)
    await agent.retrieve(Query(1, "a"), tools)
    await agent.retrieve(Query(2, "b"), tools)
    assert agent.exhausted_queries == 2


def test_the_toolset_is_built_with_an_llm_provider():
    """`RedstringToolset` accepts `llm=None` and stays fully useful for
    `dense` and `hybrid`, so a missing provider surfaces only when an LLM
    agent first reaches `extract` -- mid-run, thousands of queries in.

    Reads the private attribute deliberately: the public way to observe it
    is to call `extract`, and that is a network round trip.
    """
    tools = toolset_for(
        chunks=_StubChunks(),
        graph=object(),
        embeddings=_StubEmbeddings(),
        config=CONFIG,
        tenant_id=TenantId(uuid4()),
    )
    assert isinstance(tools._llm, LangChainLlmProvider)  # noqa: SLF001

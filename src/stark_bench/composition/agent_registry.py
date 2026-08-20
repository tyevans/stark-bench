"""Which agent a config name means, and how each one is built.

The four architectures do not share a constructor. `DenseAgent`, `HybridAgent`
and `ZeroShotAgent` take `k` alone; `DeepAgent` additionally requires a
`BudgetTracker` and deliberately has no default for it -- `agents/` is
forbidden from importing `domain.budget`, where the concrete `Budget`
lives, so composition is the only place that can supply one. A plain
`{name: class}` mapping cannot express that, which is why this module holds
a mapping of *builders* instead.

## A budget is per query, not per run

`application.run_queries.run` constructs one agent and calls `retrieve` once per query,
so a `DeepAgent` holding a single `Budget` would spend the whole run's
allowance on query 1 and return `[]` for the remaining eleven thousand -- a
silently near-zero score rather than a crash, which is the worst shape a
defect of this kind can take. `PerQueryDeepAgent` closes that by building a
fresh `Budget` and a fresh `DeepAgent` for every `retrieve`, which is also the
only reading of "budget" that makes the cost column comparable across
architectures: every other agent's cost is stated per query too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from stark_bench.agents.dense import DenseAgent
from stark_bench.agents.deep import DeepAgent
from stark_bench.agents.hybrid import HybridAgent
from stark_bench.agents.lexical import LexicalAgent
from stark_bench.agents.rerank import RerankAgent
from stark_bench.agents.zero_shot import ZeroShotAgent
from stark_bench.domain.budget import Budget

if TYPE_CHECKING:
    from collections.abc import Callable

    from stark_bench.domain.run_config import RunConfig
    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Agent, Toolset

#: Per *query*, not per run -- see the module docstring. Sized so a deep run
#: over the test split cannot cost an order of magnitude more than the
#: zero-shot one without that showing up as budget exhaustion rather than as
#: an endpoint melting: eight LLM rounds, eight tool calls, one minute.
MAX_TOOL_CALLS = 8
MAX_LLM_CALLS = 8
MAX_SECONDS = 60.0


@dataclass(slots=True)
class PerQueryDeepAgent:
    """A `DeepAgent` rebuilt, with a fresh budget, for every query."""

    k: int = 20
    max_tool_calls: int = MAX_TOOL_CALLS
    max_llm_calls: int = MAX_LLM_CALLS
    max_seconds: float = MAX_SECONDS
    name: str = "deep"

    #: How many queries ended at the cap. Recorded rather than merely raised,
    #: for the same reason `Budget.exhausted` outlives the exception.
    exhausted_queries: int = field(default=0, init=False)

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        budget = Budget(
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            max_seconds=self.max_seconds,
        )
        result = await DeepAgent(k=self.k, budget=budget).retrieve(query, tools)
        if budget.exhausted:
            self.exhausted_queries += 1
        return result


AGENTS: dict[str, Callable[[RunConfig], Agent]] = {
    "dense": lambda config: DenseAgent(k=config.k),
    "hybrid": lambda config: HybridAgent(k=config.k),
    "lexical": lambda config: LexicalAgent(k=config.k),
    "zero_shot": lambda config: ZeroShotAgent(k=config.k),
    "deep": lambda config: PerQueryDeepAgent(k=config.k),
    "rerank": lambda config: RerankAgent(k=config.k),
    #: The same architecture with a wider retrieval window, registered as a
    #: separate agent rather than as a flag on `rerank`.
    #:
    #: Two reasons, both about the record rather than about taste. Reports
    #: are named `<config>.<agent>.json`, so a flag would have overwritten
    #: the `fetch=20` number with the `fetch=40` one and left no way to see
    #: the pair -- and the pair IS the experiment. And `fetch` is not in
    #: `config_verbatim`, so with a flag the surviving file would not say
    #: which setting produced it; the agent key does say.
    #:
    #: What it buys: `rerank` fetches exactly `k`, which makes it a pure
    #: ordering experiment whose ceiling is `hybrid`'s recall@20 -- 0.46508
    #: on `qwen-rel-whole`, against which reranking returned 0.41948 mrr.
    #: That is efficient enough that the ceiling, not the ordering, is the
    #: binding constraint. Fetching 40 lets the model promote a gold answer
    #: from ranks 21-40, and equally lets it demote one off the end: both
    #: were seen in a 4-query probe. So this can lose, and a loss is a
    #: result about how far reranking can be trusted to reorder.
    "rerank40": lambda config: RerankAgent(k=config.k, fetch=40),
}


def build_agent(config: RunConfig) -> Agent:
    """The agent `config.agent` names, built for this config."""
    try:
        builder = AGENTS[config.agent]
    except KeyError:
        raise NotImplementedError(f"unknown agent {config.agent!r}") from None
    return builder(config)

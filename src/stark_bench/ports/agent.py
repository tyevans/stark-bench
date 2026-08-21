"""The seam between the harness and an agent under test.

`agents/` may import `ports` and `domain` and nothing else from
`stark_bench`. That is enforced by import-linter rather than by convention,
because the failure it prevents is silent and total: an agent that can
reach the runner can reach `answer_ids`, and a retrieval agent that can see
the answers scores perfectly while measuring nothing.

`Query` carries no answer field at all (see `domain.query`), so the
restriction holds structurally even if the contract were removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from stark_bench.domain import Passage, Query, Ranked, ToolCall


@runtime_checkable
class Toolset(Protocol):
    """Reader-only access to the knowledge base.

    Reader-only is a type-level guarantee, the same argument redstring's
    `Retriever` makes for holding a `VectorReader` rather than a
    `VectorStore`: an agent that cannot reach a writer cannot poison the
    knowledge base mid-run, and a benchmark whose subject can edit the
    corpus is measuring the subject's honesty rather than its retrieval.
    """

    calls: list[ToolCall]

    #: `rank_texts` exists because an agent cannot embed.
    #:
    #: `search_*` scores text that is IN the store. Scoring arbitrary strings
    #: -- the neighbour names inside a candidate's own document -- has no
    #: route through those, and an agent may not import `harness` to reach an
    #: `EmbeddingProvider`. A missing capability on this protocol is the
    #: sanctioned fix; an import would make the agent seam decorative.
    #:
    #: Returns one score per input text, in input order, higher is better.
    #: Scores are comparable within a call and NOT across calls: the lexical
    #: half's idf is relative to `texts`, which is the point -- rarity among
    #: the alternatives being ranked is what makes a name informative.

    async def search_chunks(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Ranked]: ...
    async def search_passages(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Passage]: ...
    async def get_node(self, node_id: str) -> dict[str, object] | None: ...
    async def neighbors(self, node_id: str, *, depth: int = 1) -> list[str]: ...
    async def get_relationships(self, node_id: str) -> list[tuple[str, str, str]]: ...
    async def rank_texts(
        self, query: str, texts: Sequence[str], *, mode: str = "hybrid"
    ) -> list[float]: ...
    async def extract[S: BaseModel](self, prompt: str, schema: type[S]) -> S: ...


@runtime_checkable
class Agent(Protocol):
    """Given a query and tools, return ranked STaRK node ids.

    The whole subject of the benchmark is behind this one method. An
    architecture is a different implementation of it and nothing else --
    which is what makes the accuracy-versus-cost comparison meaningful.
    """

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]: ...


@runtime_checkable
class BudgetTracker(Protocol):
    """What a looping agent needs from a budget, without importing one.

    Declared here rather than weakening the agents contract so that
    "spend-or-raise, with exhaustion left observable afterwards" is a
    promise the concrete `Budget` keeps, not a coincidence of two modules
    happening to agree.
    """

    exhausted: bool

    def spend_tool(self) -> None: ...
    def spend_llm(self) -> None: ...
    def seconds_remaining(self) -> float: ...

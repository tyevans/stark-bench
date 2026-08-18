"""The seam between the harness and an agent.

`agents/` may import this module and nothing else from `stark_bench`. That is
enforced by import-linter, not by convention: an agent that can reach the
runner can reach `answer_ids`, and a retrieval agent that can see the answers
is one accidental import away from a perfect score.

`Query` therefore carries no answer field at all. The restriction is
structural rather than a matter of discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class Query:
    """One STaRK query. Deliberately has no `answer_ids`."""

    query_id: int
    text: str


@dataclass(frozen=True, slots=True)
class Ranked:
    """One scored candidate, in `pred_dict` shape.

    `node_id` is STaRK's id as a string, not a redstring `EntityId`: this is
    what the official evaluator consumes, so nothing is reshaped between an
    agent and scoring.
    """

    node_id: str
    score: float


@dataclass(slots=True)
class ToolCall:
    """One recorded call. Cost is a reported metric, not a footnote."""

    tool: str
    duration_s: float
    result_count: int


@runtime_checkable
class Toolset(Protocol):
    """Reader-only access to the knowledge base.

    Reader-only is a type-level guarantee, the same argument redstring's own
    `Retriever` makes for holding `VectorReader` rather than `VectorStore`: an
    agent that cannot reach a writer cannot poison the KB mid-run.
    """

    calls: list[ToolCall]

    async def search_chunks(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Ranked]: ...
    async def get_node(self, node_id: str) -> dict[str, object] | None: ...
    async def neighbors(self, node_id: str, *, depth: int = 1) -> list[str]: ...
    async def get_relationships(self, node_id: str) -> list[tuple[str, str, str]]: ...
    async def extract[S: BaseModel](self, prompt: str, schema: type[S]) -> S: ...


@runtime_checkable
class Agent(Protocol):
    """Given a query and tools, return ranked STaRK node ids."""

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]: ...

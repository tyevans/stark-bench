"""One vector search, returned as-is. The control.

No LLM, so it runs the full query set cheaply and often -- which is what lets
us tell whether a moved agent number reflects the agent or the knowledge base
underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stark_bench.ports import Query, Ranked, Toolset


@dataclass(frozen=True, slots=True)
class DenseAgent:
    k: int = 20
    name: str = "dense"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        return await tools.search_chunks(query.text, k=self.k, mode="semantic")

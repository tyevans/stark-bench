"""Vector and BM25, fused by rank inside redstring.

Answers "does redstring's fusion beat dense retrieval on STaRK" with no agent
variance in the way. The fusion constant is redstring's and is not tuned here:
its docstring says exposing it would invite tuning against a benchmark the
library does not have, and this is that benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stark_bench.ports import Query, Ranked, Toolset


@dataclass(frozen=True, slots=True)
class HybridAgent:
    k: int = 20
    name: str = "hybrid"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        return await tools.search_chunks(query.text, k=self.k, mode="hybrid")

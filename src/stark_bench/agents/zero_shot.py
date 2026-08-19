"""One LLM call to rewrite the query, then one retrieval round.

Fixed cost per query, no loop -- that fixed cost is the defining property of
this architecture and what distinguishes it from the deep agent in Task 14.

The LLM seam is `extract(prompt, schema)`, not `complete(prompt) -> str`:
redstring's `LlmProvider` offers only structured extraction against a
caller-supplied pydantic schema, so this agent declares what it wants back
and gets a validated instance or an exception. There is no prose to parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Toolset


class RewrittenQuery(BaseModel):
    query: str


_PROMPT_TEMPLATE = (
    "Rewrite the following search query as a concise set of retrieval terms "
    "that best capture what it is asking about. Return only the rewritten "
    "query text.\n\nQuery: {text}"
)


@dataclass(frozen=True, slots=True)
class ZeroShotAgent:
    k: int = 20
    name: str = "zero_shot"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        search_text = query.text
        try:
            rewritten = await tools.extract(
                _PROMPT_TEMPLATE.format(text=query.text), RewrittenQuery
            )
        except Exception:
            rewritten = None

        if rewritten is not None and rewritten.query.strip():
            search_text = rewritten.query

        return await tools.search_chunks(search_text, k=self.k, mode="hybrid")

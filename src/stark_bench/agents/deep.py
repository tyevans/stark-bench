"""Plan, act, observe, repeat -- the one architecture with a loop.

`zero_shot.py` makes exactly one LLM call and its cost is therefore fixed.
This one calls `extract` to decide its next tool, calls that tool, folds the
observation back in, and asks again -- which means both its cost *and* its
prompt size grow with the query, and neither is allowed to grow unboundedly.

Two hard bounds keep that honest:

- `budget: BudgetTracker` (declared in `stark_bench.ports`, not imported from
  `stark_bench.harness` -- `agents/` may only import `ports`, per the
  import-linter contract in `pyproject.toml`) caps tool calls and LLM calls
  *separately*, so a cheap tool loop can never starve the LLM's own budget.
  Running out is a recorded outcome (`budget.exhausted`), not an exception
  that discards the run: the loop catches `BudgetExhausted` and returns
  whatever candidates it already has.
- `_MAX_PROMPT_CHARS` caps how much observation history rides along in each
  `extract` call. The backing endpoint runs at a 16k-token context window
  shared across several jobs (`-np 4`); a loop that appended every
  observation verbatim would eventually blow that window on a query that
  needs enough hops. Older observations are dropped, newest first, once the
  budget is spent -- the agent trades history for staying inside the window
  rather than crashing partway through a query.

The loop also terminates even if the LLM always asks for another step: every
iteration spends one unit of `max_llm_calls` before it is allowed to act, so
a model that never says "finish" still runs out in `max_llm_calls` rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from stark_bench.ports import BudgetTracker, Ranked

if TYPE_CHECKING:
    from stark_bench.ports import Query, Toolset

#: Characters, not tokens -- the ~4 chars/token estimate this module reports
#: on. Leaves headroom in a 16k-token window for the model's own reply.
_MAX_PROMPT_CHARS = 40_000

_PROMPT_TEMPLATE = (
    "You are answering a search query by choosing one tool call at a time.\n"
    "Query: {query}\n\n"
    "Observations so far (most recent last):\n{observations}\n\n"
    "Choose the next action: 'search' (argument = search text), "
    "'get_node' (argument = node id), 'neighbors' (argument = node id), "
    "'relationships' (argument = node id), or 'finish' (argument may be "
    "empty) once you have enough to answer."
)


class Step(BaseModel):
    action: Literal["search", "get_node", "neighbors", "relationships", "finish"]
    argument: str


@dataclass(slots=True)
class DeepAgent:
    #: No default -- a `BudgetTracker` (see `stark_bench.ports`) must be
    #: supplied by the caller. `agents/` may not import the concrete
    #: `Budget` from `stark_bench.harness`, even to build a default one, per
    #: the import-linter contract in `pyproject.toml`.
    budget: BudgetTracker
    k: int = 20
    name: str = "deep"

    #: Set after `retrieve` returns, in characters. Not part of the
    #: contract -- a measurement hook for the harness's cost report.
    peak_prompt_chars: int = field(default=0, init=False, repr=False)

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        self.peak_prompt_chars = 0
        candidates: dict[str, float] = {}
        observations: list[str] = []

        while True:
            try:
                self.budget.spend_llm()
            except Exception:
                break

            prompt = self._build_prompt(query.text, observations)
            self.peak_prompt_chars = max(self.peak_prompt_chars, len(prompt))

            try:
                step = await tools.extract(prompt, Step)
            except Exception:
                break

            if step.action == "finish":
                break

            try:
                self.budget.spend_tool()
            except Exception:
                break

            observation = await self._act(step, tools, candidates)
            observations.append(observation)
            observations = self._truncate(observations)

        if not candidates:
            return []

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [Ranked(node_id, score) for node_id, score in ranked[: self.k]]

    async def _act(
        self, step: Step, tools: Toolset, candidates: dict[str, float]
    ) -> str:
        if step.action == "search":
            results = await tools.search_chunks(step.argument, k=self.k, mode="hybrid")
            for ranked in results:
                candidates[ranked.node_id] = max(
                    candidates.get(ranked.node_id, float("-inf")), ranked.score
                )
            return f"search({step.argument!r}) -> {[r.node_id for r in results]}"

        if step.action == "get_node":
            node = await tools.get_node(step.argument)
            if node is not None:
                candidates.setdefault(step.argument, 0.5)
            return f"get_node({step.argument!r}) -> {node}"

        if step.action == "neighbors":
            neighbor_ids = await tools.neighbors(step.argument, depth=1)
            for node_id in neighbor_ids:
                candidates.setdefault(node_id, 0.25)
            return f"neighbors({step.argument!r}) -> {neighbor_ids}"

        # "relationships" -- reveals edge type/direction that `neighbors`
        # deliberately omits, per the toolset's own contract.
        edges = await tools.get_relationships(step.argument)
        for _source, _edge_type, target in edges:
            candidates.setdefault(target, 0.25)
        return f"relationships({step.argument!r}) -> {edges}"

    @staticmethod
    def _build_prompt(query_text: str, observations: list[str]) -> str:
        body = "\n".join(observations) if observations else "(none yet)"
        return _PROMPT_TEMPLATE.format(query=query_text, observations=body)

    @staticmethod
    def _truncate(observations: list[str]) -> list[str]:
        """Drop the oldest observations until the joined history fits.

        Keeps the newest entries -- the most recent hop is the one most
        likely to matter for the next decision -- and always leaves at least
        the single most recent observation, even if it alone exceeds the
        budget, so one oversized observation can't wedge the loop.
        """
        kept = list(observations)
        while len(kept) > 1 and sum(len(o) for o in kept) > _MAX_PROMPT_CHARS:
            kept.pop(0)
        return kept

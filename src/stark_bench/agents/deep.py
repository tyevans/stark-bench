"""Plan, act, observe, repeat -- the one architecture with a loop.

`zero_shot.py` makes exactly one LLM call and its cost is therefore fixed.
This one calls `extract` to decide its next tool, calls that tool, folds the
observation back in, and asks again -- which means both its cost *and* its
prompt size grow with the query, and neither is allowed to grow unboundedly.

Two hard bounds keep that honest:

- `budget: BudgetTracker` (declared in `stark_bench.ports`, not the concrete
  `stark_bench.domain.budget.Budget`, which `agents/` is forbidden from
  importing by name in the `pyproject.toml` contract) caps tool calls and
  LLM calls *separately*, so a cheap tool loop can never starve the LLM's
  own budget.
  Running out is a recorded outcome (`budget.exhausted`), not an exception
  that discards the run: the loop catches `BudgetExhausted` and returns
  whatever candidates it already has.
- `_MAX_PROMPT_CHARS` caps how much observation history rides along in each
  `extract` call. That window covers the prompt *and* the generated output
  together -- a bound sized to leave no headroom for a response is not a
  bound. Older observations are dropped, newest
  first, once the budget is spent; a single observation larger than the
  whole budget is hard-truncated rather than passed through whole, so no one
  oversized tool result can defeat the cap in the one case it matters most.
  The agent trades history (and, in the worst case, some of one
  observation's detail) for staying inside the window rather than crashing
  partway through a query.

The loop also terminates even if the LLM always asks for another step: every
iteration spends one unit of `max_llm_calls` before it is allowed to act, so
a model that never says "finish" still runs out in `max_llm_calls` rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from stark_bench.domain import Ranked
from stark_bench.ports import BudgetTracker

if TYPE_CHECKING:
    from stark_bench.domain import Query
    from stark_bench.ports import Toolset

#: Characters, not tokens -- estimated at ~4 chars/token (conservative for
#: English text; a real tokenizer would be model-specific and is not worth
#: the dependency here). The backing window covers the prompt *and* the
#: generated output together, so this stays under half of it.
#:
#: Raised from 24,000 to 48,000 when the endpoint moved to a 32k window on
#: 2026-08-19. The old value was never the server's limit -- 24,000 chars is
#: ~6k tokens against a 16k window -- it was this agent's own, and it bound
#: first and bound hard. The measured peak against an adversarial fake was
#: 24,774 chars, so the cap was being hit in real runs, and what it drops is
#: the *oldest* observations: on a hub node like PRIME's ABLIM1 (degree 426)
#: a single `neighbors` result runs to thousands of characters, so the agent
#: reached its eighth decision having forgotten what its second and third
#: calls returned. That is a handicap on precisely the multi-hop queries
#: traversal is supposed to win.
#:
#: Deep numbers measured under the old cap are not comparable with numbers
#: measured under this one. Re-run every arm rather than mixing them.
_MAX_PROMPT_CHARS = 48_000

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
    #: supplied by the caller. `agents/` may not import
    #: `stark_bench.domain.budget`, even to build a default one: holding a
    #: `Budget` would let an agent read the caps it is being judged against,
    #: or construct its own. Named in the forbidden contract, not merely
    #: asked for here.
    budget: BudgetTracker
    k: int = 20
    name: str = "deep"

    #: Set after `retrieve` returns, in characters. Not part of the
    #: contract -- a measurement hook for the harness's cost report.
    peak_prompt_chars: int = field(default=0, init=False, repr=False)

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        self.peak_prompt_chars = 0
        retrieved: dict[str, float] = {}
        discovered: dict[str, int] = {}
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

            observation = await self._act(step, tools, retrieved, discovered)
            observations.append(observation)
            observations = self._truncate(observations)

        return self._rank(retrieved, discovered)

    def _rank(
        self, retrieved: dict[str, float], discovered: dict[str, int]
    ) -> list[Ranked]:
        """Retrieval evidence first, then traversal evidence.

        The two are different quantities and comparing them by magnitude is
        what B-DEEP-SCORE-SCALE-1 was: any constant assigned to a traversal
        hit either always beats the retriever's scores or never does,
        depending only on the retriever's scale. redstring's hybrid channel
        is an RRF fusion scoring ~0.02, so a 0.25 constant always won and a
        hub's neighbours filled the whole result.

        So order by *source* first and by evidence within it. A node the
        retriever scored outranks one only reached by traversal, whatever
        the numbers. Among traversal-only nodes the signal is corroboration:
        a node arrived at from several different hops is more likely to
        matter than one seen once, and a raw neighbour list carries no
        internal order to use instead.

        Emitted scores are rank-derived and strictly decreasing, so nothing
        downstream can re-derive the comparison this method exists to avoid.
        """
        by_score = sorted(retrieved.items(), key=lambda kv: (-kv[1], kv[0]))
        traversal_only = [
            (node_id, hits)
            for node_id, hits in discovered.items()
            if node_id not in retrieved
        ]
        by_corroboration = sorted(traversal_only, key=lambda kv: (-kv[1], kv[0]))

        ordered = [n for n, _ in by_score] + [n for n, _ in by_corroboration]
        return [
            Ranked(node_id, 1.0 / (1 + rank))
            for rank, node_id in enumerate(ordered[: self.k])
        ]

    async def _act(
        self,
        step: Step,
        tools: Toolset,
        retrieved: dict[str, float],
        discovered: dict[str, int],
    ) -> str:
        if step.action == "search":
            results = await tools.search_chunks(step.argument, k=self.k, mode="hybrid")
            for ranked in results:
                retrieved[ranked.node_id] = max(
                    retrieved.get(ranked.node_id, float("-inf")), ranked.score
                )
            return f"search({step.argument!r}) -> {[r.node_id for r in results]}"

        if step.action == "get_node":
            node = await tools.get_node(step.argument)
            if node is not None:
                discovered[step.argument] = discovered.get(step.argument, 0) + 1
            return f"get_node({step.argument!r}) -> {node}"

        if step.action == "neighbors":
            neighbor_ids = await tools.neighbors(step.argument, depth=1)
            for node_id in neighbor_ids:
                discovered[node_id] = discovered.get(node_id, 0) + 1
            return f"neighbors({step.argument!r}) -> {neighbor_ids}"

        # "relationships" -- reveals edge type/direction that `neighbors`
        # deliberately omits, per the toolset's own contract.
        edges = await tools.get_relationships(step.argument)
        for _source, _edge_type, target in edges:
            discovered[target] = discovered.get(target, 0) + 1
        return f"relationships({step.argument!r}) -> {edges}"

    @staticmethod
    def _build_prompt(query_text: str, observations: list[str]) -> str:
        body = "\n".join(observations) if observations else "(none yet)"
        return _PROMPT_TEMPLATE.format(query=query_text, observations=body)

    @staticmethod
    def _truncate(observations: list[str]) -> list[str]:
        """Drop the oldest observations until the joined history fits.

        Keeps the newest entries -- the most recent hop is the one most
        likely to matter for the next decision. If the single most recent
        observation alone still exceeds the budget, it is hard-truncated
        rather than passed through whole: letting one oversized tool result
        ride along uncapped would defeat the bound in exactly the case where
        it matters, which is the failure mode this method exists to close.
        """
        kept = list(observations)
        while len(kept) > 1 and sum(len(o) for o in kept) > _MAX_PROMPT_CHARS:
            kept.pop(0)

        if kept and len(kept[-1]) > _MAX_PROMPT_CHARS:
            marker = "...[truncated]"
            kept[-1] = kept[-1][: _MAX_PROMPT_CHARS - len(marker)] + marker

        return kept

"""A per-query cost cap, with separate counters for tools and the LLM.

A looping agent (Task 14's `DeepAgent`) needs to stop itself before it costs
more than the harness is willing to pay for one query, and it needs to keep
whatever it had found up to that point rather than throw the run away.
`BudgetExhausted` is how a single `spend_*` call communicates "no more of
this", but the loop that catches it is expected to treat that as the normal
end of a run, not a failure -- `exhausted` stays `True` afterwards precisely
so the caller can record that outcome instead of merely observing that an
exception happened to be raised somewhere.

One counter per resource, not one shared counter: a cheap `search_chunks`
loop that never calls the LLM must not be able to exhaust the LLM's budget by
proxy, and vice versa.

`stark_bench.ports.BudgetTracker` is the narrow protocol this class
satisfies. While this module lived in `harness/`, the contract forbade
`agents/` from importing it and the port was the only way an agent could
see a budget at all. That is no longer the reason -- `agents/` may import
`domain` -- and the port is kept on the weaker but sufficient one: an agent
needs to spend and to ask whether it is exhausted, and nothing else here.
Depending on the class would let an agent read the caps it is being judged
against, or construct its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    """Raised by a `spend_*` call that would exceed its cap."""


@dataclass(slots=True)
class Budget:
    max_tool_calls: int
    max_llm_calls: int
    max_seconds: float

    tool_calls: int = field(default=0, init=False)
    llm_calls: int = field(default=0, init=False)
    exhausted: bool = field(default=False, init=False)

    _start: float = field(default_factory=time.monotonic, init=False, repr=False)

    def _check_time(self) -> None:
        if time.monotonic() - self._start > self.max_seconds:
            self.exhausted = True
            raise BudgetExhausted("time budget exhausted")

    def spend_tool(self) -> None:
        self._check_time()
        if self.tool_calls >= self.max_tool_calls:
            self.exhausted = True
            raise BudgetExhausted("tool-call budget exhausted")
        self.tool_calls += 1

    def spend_llm(self) -> None:
        self._check_time()
        if self.llm_calls >= self.max_llm_calls:
            self.exhausted = True
            raise BudgetExhausted("llm-call budget exhausted")
        self.llm_calls += 1

    def seconds_remaining(self) -> float:
        remaining = self.max_seconds - (time.monotonic() - self._start)
        return max(0.0, remaining)

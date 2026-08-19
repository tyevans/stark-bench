"""What an answer cost, alongside how good it was.

Accuracy alone cannot rank retrieval architectures. A deep agent budgeted at
eight LLM calls and eight tool calls per query can buy a better number at
ten to a hundred times the cost of a dense one, and a benchmark reporting
only accuracy scores that as an unqualified win.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolCall:
    """One recorded call."""

    tool: str
    duration_s: float
    result_count: int
    #: `None` means the endpoint reported no usage. Distinct from `0`, which
    #: claims the call consumed nothing -- and the distinction is not
    #: pedantic: an ingest block of zeroes is exactly what a *missing* cost
    #: column looked like for the entire life of this project's reports.
    tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Cost:
    """What a whole run spent, per query where per query is meaningful.

    Rates rather than totals, because the architectures are compared against
    each other and one may be run over a different number of queries than
    another during development.
    """

    tool_calls_per_query: float
    llm_calls_per_query: float
    seconds_total: float
    #: `None` when no call reported usage. See `ToolCall.tokens`.
    tokens_per_query: float | None = None

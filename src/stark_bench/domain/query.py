"""What an agent is asked, and what it may answer with."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Query:
    """One STaRK query. Deliberately has no `answer_ids`.

    The omission is the point and it is structural rather than a matter of
    discipline: an agent that can see the answers is one accidental
    attribute access away from a perfect score, and a retrieval benchmark
    that can be cheated by autocomplete is not measuring anything.
    """

    query_id: int
    text: str


@dataclass(frozen=True, slots=True)
class Ranked:
    """One scored candidate, in STaRK's `pred_dict` shape.

    `node_id` is STaRK's id as a string, not a redstring `EntityId`. That is
    what the official evaluator consumes, so nothing is reshaped between an
    agent and scoring -- a translation layer there would be one more place
    for an id scheme to diverge, and id schemes diverging is how a benchmark
    silently scores zero.
    """

    node_id: str
    score: float

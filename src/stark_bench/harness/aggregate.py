"""Scored chunks up to scored nodes.

Retrieval returns chunks; STaRK scores nodes. The strategy is a *named,
recorded* config value rather than an implicit default, because an
aggregation function that is an unrecorded knob turns a benchmark into a
search for its best accident.

On `vss-control` there is one chunk per node, so every strategy degenerates
to identity -- the control exercises this code without aggregation being able
to change its answer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from stark_bench.ports import Ranked

if TYPE_CHECKING:
    from collections.abc import Sequence

AGGREGATIONS = {
    "max": max,
    "mean": lambda scores: sum(scores) / len(scores),
    "sum": sum,
}


def aggregate(
    scored: Sequence[tuple[str, float]], *, strategy: str = "max"
) -> list[Ranked]:
    """Fold per-chunk scores into per-node scores, best first.

    Ties break on `node_id` so two runs over the same data rank identically;
    without it a metric can move between runs for no reason at all.
    """
    reducer = AGGREGATIONS[strategy]
    grouped: dict[str, list[float]] = defaultdict(list)
    for node_id, score in scored:
        grouped[node_id].append(score)

    ranked = [Ranked(node_id=n, score=float(reducer(s))) for n, s in grouped.items()]
    ranked.sort(key=lambda r: (-r.score, r.node_id))
    return ranked

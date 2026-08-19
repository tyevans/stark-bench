"""One JSON file per run, carrying its own config.

Cost sits beside accuracy deliberately: a deep agent that buys four points of
Hit@1 for forty times the tokens is a different finding depending on which
number you needed, and an accuracy table alone cannot express it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from stark_bench.harness.config import RunConfig
    from stark_bench.ports import ToolCall


def summarise_cost(calls: Sequence[ToolCall], queries: int) -> dict[str, float]:
    if queries == 0:
        return {
            "tool_calls_per_query": 0.0,
            "llm_calls_per_query": 0.0,
            "tokens_per_query": None,
            "seconds_total": 0.0,
        }
    llm_calls = [c for c in calls if c.tool == "extract"]
    measured = [c for c in llm_calls if c.tokens is not None]
    return {
        "tool_calls_per_query": len(calls) / queries,
        "llm_calls_per_query": len(llm_calls) / queries,
        # `None`, not 0.0, when nothing reported usage: a missing measurement
        # and a measurement of nothing are different facts.
        "tokens_per_query": (
            sum(c.tokens for c in measured) / queries if measured else None
        ),
        "seconds_total": sum(c.duration_s for c in calls),
    }


def write_report(
    path: Path,
    *,
    config: RunConfig,
    metrics: Mapping[str, float],
    cost: Mapping[str, float],
    ingest: Mapping[str, int],
    queries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config_name": config.name,
                "config_verbatim": config.raw,
                "queries": queries,
                "metrics": dict(metrics),
                "cost": dict(cost),
                "ingest": dict(ingest),
            },
            indent=2,
        )
    )

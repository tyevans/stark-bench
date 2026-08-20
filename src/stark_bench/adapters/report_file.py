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

    from stark_bench.domain.run_config import RunConfig
    from stark_bench.domain import Ranked, ToolCall


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


def write_predictions(path: Path, predictions: Mapping[int, Sequence[Ranked]]) -> None:
    """Persist raw rankings, in the shape the scoring sidecar already reads.

    Deliberately the sidecar's own `{qid: {node_id: score}}` format rather than
    a new one, so a rescore is a file move rather than a translation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                str(qid): {r.node_id: r.score for r in ranked}
                for qid, ranked in predictions.items()
            }
        )
    )


def write_report(
    path: Path,
    *,
    config: RunConfig,
    metrics: Mapping[str, float],
    cost: Mapping[str, float],
    # Not `Mapping[str, int]`: the ingest block carries `wall_time_s`
    # (float), `edges_ingested` and `resume` (bool). The narrower annotation
    # was true when the block was always empty.
    ingest: Mapping[str, object],
    queries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config_name": config.name,
                "config_verbatim": config.raw,
                # The split that RAN, which `config_verbatim` cannot express:
                # it is the config file's own bytes, so on a `--split test`
                # run it still reads `test-0.1`. Recording only the verbatim
                # config would make the file confidently name the wrong
                # query set.
                "split": config.effective_split,
                # The chat model that RAN, for the same reason as `split`.
                # `None` means the composition default, which is recorded
                # in `cost` only if an agent actually made an LLM call.
                "chat_model": config.effective_chat_model,
                "queries": queries,
                "metrics": dict(metrics),
                "cost": dict(cost),
                "ingest": dict(ingest),
            },
            indent=2,
        )
    )

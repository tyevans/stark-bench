"""Run one agent over a query set.

A query that raises is recorded as an empty prediction rather than aborting
the run: over eleven thousand queries, one agent failure must not discard
every result after it.

Progress is reported and, optionally, checkpointed. A `deep` or `rerank` run
is an hour or more of shared GPU during which the process previously said
nothing and wrote nothing -- so "how is it doing?" could only be answered by
counting HTTP lines in a log, and "what does it look like so far?" could not
be answered at all without paying for a second run. A crash at query 279 also
took the other 278 with it, which is not hypothetical: a PyPI outage during
scoring discarded a completed 46-minute run, and the fix for that (writing
predictions before scoring) still wrote nothing until the last query landed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Agent, Toolset

logger = logging.getLogger(__name__)


async def run(
    agent: Agent,
    queries: Sequence[Query],
    tools: Toolset,
    *,
    k: int = 20,
    checkpoint: Callable[[Mapping[int, list[Ranked]]], None] | None = None,
    checkpoint_every: int = 25,
) -> dict[int, list[Ranked]]:
    """Retrieve for every query, reporting progress and checkpointing.

    `checkpoint` is a plain callable rather than a path so this layer keeps
    knowing nothing about files; `composition` passes the same writer that
    persists the final predictions, so a checkpoint and a finished run are
    the same artifact in the same format -- a partial run is scoreable with
    `scripts/rescore.py` exactly as a whole one is.
    """
    predictions: dict[int, list[Ranked]] = {}
    total = len(queries)
    for index, query in enumerate(queries, start=1):
        try:
            predictions[query.query_id] = list(await agent.retrieve(query, tools))
        except Exception:
            logger.exception("agent failed on query %s", query.query_id)
            predictions[query.query_id] = []

        if index % checkpoint_every == 0 or index == total:
            empty = sum(1 for ranked in predictions.values() if not ranked)
            logger.info("%s/%s queries done (%s empty)", index, total, empty)
            if checkpoint is not None:
                checkpoint(predictions)
    return predictions

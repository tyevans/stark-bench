"""Run one agent over a query set.

A query that raises is recorded as an empty prediction rather than aborting
the run: over eleven thousand queries, one agent failure must not discard
every result after it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Agent, Toolset

logger = logging.getLogger(__name__)


async def run(
    agent: Agent, queries: Sequence[Query], tools: Toolset, *, k: int = 20
) -> dict[int, list[Ranked]]:
    predictions: dict[int, list[Ranked]] = {}
    for query in queries:
        try:
            predictions[query.query_id] = list(await agent.retrieve(query, tools))
        except Exception:
            logger.exception("agent failed on query %s", query.query_id)
            predictions[query.query_id] = []
    return predictions

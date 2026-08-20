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

import asyncio
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
    concurrency: int = 1,
) -> dict[int, list[Ranked]]:
    """Retrieve for every query, reporting progress and checkpointing.

    `checkpoint` is a plain callable rather than a path so this layer keeps
    knowing nothing about files; `composition` passes the same writer that
    persists the final predictions, so a checkpoint and a finished run are
    the same artifact in the same format -- a partial run is scoreable with
    `scripts/rescore.py` exactly as a whole one is.

    ## `concurrency`

    A request occupies one server slot. This loop was serial for as long as
    the chat model ran at `-np 1`, where a second in-flight request could
    only have queued -- `cli.py` says as much, and said nothing was given up.
    That stopped being true when the model moved to `-np 4`: a serial client
    against four slots leaves three quarters of the server idle, which is
    the same mistake `--embed-concurrency 1` made on the ingest side and
    cost 24% there.

    Concurrency does not change any accuracy number. Queries are
    independent, `predictions` is keyed by `query_id` rather than by
    position, and every agent here is stateless across `retrieve`
    (`PerQueryDeepAgent` rebuilds its budget per call, deliberately). It
    changes wall time and it changes what contends on Postgres, which is why
    it is a parameter rather than a default.

    `concurrency=1` is the previous behaviour exactly: `asyncio.Semaphore`
    hands out permits FIFO and the tasks are created in query order, so the
    queries run in order, one at a time.
    """
    predictions: dict[int, list[Ranked]] = {}
    total = len(queries)
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    limit = asyncio.Semaphore(concurrency)
    completed = 0

    async def _one(query: Query) -> None:
        nonlocal completed
        async with limit:
            try:
                ranked = list(await agent.retrieve(query, tools))
            except Exception:
                logger.exception("agent failed on query %s", query.query_id)
                ranked = []
        predictions[query.query_id] = ranked
        # Counted on completion, not on position: with concurrency the
        # queries finish out of order, and `index % every` would checkpoint
        # at arbitrary moments or not at all.
        completed += 1
        if completed % checkpoint_every == 0 or completed == total:
            empty = sum(1 for r in predictions.values() if not r)
            logger.info("%s/%s queries done (%s empty)", completed, total, empty)
            if checkpoint is not None:
                checkpoint(predictions)

    await asyncio.gather(*(_one(query) for query in queries))
    return predictions

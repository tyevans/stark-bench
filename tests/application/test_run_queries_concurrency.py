"""Queries may run concurrently, and `concurrency=1` is the old behaviour.

The chat model moved to `-np 4`. A request occupies one slot, so a serial
client leaves three quarters of it idle -- the same mistake
`--embed-concurrency 1` made on the ingest side, where it cost 24%.

What must NOT change is any accuracy number. Queries are independent and
`predictions` is keyed by `query_id`, so these tests pin that: same results,
same keys, regardless of how many are in flight.
"""

from __future__ import annotations

import asyncio

import pytest

from stark_bench.application.run_queries import run
from stark_bench.domain import Query, Ranked


class SlowAgent:
    """Records overlap: how many retrieves were in flight at the peak."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.order: list[int] = []

    async def retrieve(self, query: Query, tools: object) -> list[Ranked]:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            self.order.append(query.query_id)
            return [Ranked(node_id=str(query.query_id), score=1.0)]
        finally:
            self.in_flight -= 1


def _queries(n: int = 8) -> list[Query]:
    return [Query(query_id=i, text=f"q{i}") for i in range(1, n + 1)]


async def test_concurrency_one_runs_strictly_serially() -> None:
    """The historical behaviour, preserved exactly."""
    agent = SlowAgent()
    await run(agent, _queries(), tools=object(), concurrency=1)
    assert agent.peak == 1
    assert agent.order == list(range(1, 9)), "serial must also stay in order"


async def test_concurrency_four_actually_overlaps() -> None:
    """Catches a `concurrency` parameter that is accepted and ignored --
    this project's signature defect."""
    agent = SlowAgent()
    await run(agent, _queries(), tools=object(), concurrency=4)
    assert agent.peak == 4


async def test_concurrency_never_exceeds_the_limit() -> None:
    """More in flight than slots is not free: it queues on the server and
    contends on Postgres."""
    agent = SlowAgent()
    await run(agent, _queries(16), tools=object(), concurrency=3)
    assert agent.peak <= 3


@pytest.mark.parametrize("concurrency", [1, 2, 4, 16])
async def test_results_are_identical_whatever_the_concurrency(
    concurrency: int,
) -> None:
    """The claim that lets previously-scored arms stand."""
    agent = SlowAgent(delay=0.001)
    got = await run(agent, _queries(), tools=object(), concurrency=concurrency)
    assert {q: [r.node_id for r in v] for q, v in got.items()} == {
        i: [str(i)] for i in range(1, 9)
    }


async def test_a_failing_query_does_not_take_the_others() -> None:
    class Flaky:
        async def retrieve(self, query: Query, tools: object) -> list[Ranked]:
            if query.query_id == 3:
                raise RuntimeError("boom")
            return [Ranked(node_id=str(query.query_id), score=1.0)]

    got = await run(Flaky(), _queries(5), tools=object(), concurrency=4)
    assert got[3] == []
    assert len(got) == 5


async def test_checkpoints_count_completions_not_positions() -> None:
    """Out of order completion makes `index % every` fire at arbitrary
    moments; a completion counter does not."""
    seen: list[int] = []
    agent = SlowAgent(delay=0.001)
    await run(
        agent,
        _queries(10),
        tools=object(),
        concurrency=4,
        checkpoint=lambda preds: seen.append(len(preds)),
        checkpoint_every=5,
    )
    assert seen == [5, 10]


async def test_zero_concurrency_is_refused() -> None:
    """`Semaphore(0)` would hang forever, which looks exactly like a slow
    endpoint."""
    with pytest.raises(ValueError, match="concurrency"):
        await run(SlowAgent(), _queries(2), tools=object(), concurrency=0)

import pytest

from stark_bench.application.run_queries import run
from stark_bench.domain import Query, Ranked, ToolCall


class Tools:
    def __init__(self):
        self.calls: list[ToolCall] = []

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        return [Ranked("1", 1.0)]

    async def get_node(self, node_id):
        return None

    async def neighbors(self, node_id, *, depth=1):
        return []

    async def get_relationships(self, node_id):
        return []

    async def complete(self, prompt):
        return ""


class Boom:
    name = "boom"

    async def retrieve(self, query, tools):
        if query.query_id == 2:
            raise ValueError("this query breaks the agent")
        return [Ranked("1", 1.0)]


@pytest.mark.asyncio
async def test_a_failing_query_does_not_abort_the_run():
    """A bad query followed by a good one.

    With the failure last, `break` and `continue` are the same function, and a
    `break` would silently discard every later query in an 11k-query run.
    """
    queries = [Query(1, "a"), Query(2, "b"), Query(3, "c")]
    predictions = await run(Boom(), queries, Tools())
    assert set(predictions) == {1, 2, 3}
    assert predictions[2] == []


@pytest.mark.asyncio
async def test_checkpoints_land_before_the_run_finishes():
    """A checkpoint after the last query only would be worth nothing.

    The point is that an hour-long run is inspectable and survivable partway
    through, so this asserts a checkpoint arrives *while queries remain* --
    a version that wrote once at the end passes any test that only counts
    calls or checks the final contents.
    """
    seen: list[int] = []
    queries = [Query(query_id=i, text=f"q{i}") for i in range(10)]

    class _Agent:
        async def retrieve(self, query, tools):
            return [Ranked(node_id=str(query.query_id), score=1.0)]

    await run(
        _Agent(),
        queries,
        object(),
        checkpoint=lambda preds: seen.append(len(preds)),
        checkpoint_every=3,
    )

    assert seen, "no checkpoint was taken"
    assert seen[0] < len(queries), (
        f"first checkpoint held {seen[0]} of {len(queries)} queries -- "
        "checkpointing only at the end defeats the purpose"
    )
    assert seen[-1] == len(queries)


@pytest.mark.asyncio
async def test_a_checkpoint_holds_the_results_so_far_not_a_placeholder():
    captured: list[dict] = []
    queries = [Query(query_id=i, text=f"q{i}") for i in range(6)]

    class _Agent:
        async def retrieve(self, query, tools):
            return [Ranked(node_id=f"n{query.query_id}", score=0.5)]

    await run(
        _Agent(),
        queries,
        object(),
        checkpoint=lambda preds: captured.append(dict(preds)),
        checkpoint_every=2,
    )

    first = captured[0]
    assert set(first) == {0, 1}
    assert first[0][0].node_id == "n0"

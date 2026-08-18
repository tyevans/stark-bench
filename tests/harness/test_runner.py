import pytest

from stark_bench.harness.runner import run
from stark_bench.ports import Query, Ranked, ToolCall


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

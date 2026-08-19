"""`ingest_corpus` decides three things; each gets a test that can fail.

The counts test uses five *distinct* numbers on purpose. Every field is an
`int`, so a mapping that crosses two of them type-checks and passes any
fixture that reuses a value -- and this outcome is exactly the block that
reached every report as `{}` without anyone noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


from stark_bench.application.ingest_corpus import ingest_corpus

TENANT = UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT = UUID("22222222-2222-2222-2222-222222222222")


@dataclass(frozen=True)
class Counts:
    nodes: int
    edges: int
    chunks: int
    skipped: int
    self_loops_dropped: int


class RecordingEngine:
    """Satisfies `IngestEngine` and remembers how it was called."""

    def __init__(self, counts: Counts) -> None:
        self._counts = counts
        self.calls: list[dict[str, object]] = []

    async def __call__(self, nodes, edges, /, **kwargs):  # noqa: ANN001, ANN003, ANN204
        self.calls.append({"nodes": nodes, "edges": edges, **kwargs})
        return self._counts


class RecordingIndex:
    """Satisfies `ChunkIdIndex` and remembers who asked."""

    def __init__(self, ids: set[str]) -> None:
        self._ids = ids
        self.asked_for: list[UUID] = []

    async def ids_for_tenant(self, tenant_id: UUID) -> set[str]:
        self.asked_for.append(tenant_id)
        return set(self._ids)


def scripted_clock(*values: float):  # noqa: ANN201
    """A clock returning a fixed sequence, so durations are literals.

    Asserting `wall_time_s` against `end - start` computed from the same
    clock is true for any implementation, including one that returns zero.
    """
    remaining = list(values)

    def clock() -> float:
        if not remaining:
            raise AssertionError("clock called more times than the test scripted")
        return remaining.pop(0)

    return clock


async def test_every_count_lands_in_its_own_field() -> None:
    engine = RecordingEngine(
        Counts(nodes=11, edges=22, chunks=33, skipped=44, self_loops_dropped=55)
    )

    outcome = await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=TENANT,
        chunk_index=None,
        edges_ingested=True,
        config_verbatim="name: arm\n",
        clock=scripted_clock(100.0, 107.0),
    )

    assert outcome.nodes == 11
    assert outcome.edges == 22
    assert outcome.chunks == 33
    assert outcome.skipped == 44
    assert outcome.self_loops_dropped == 55
    assert outcome.edges_ingested is True
    assert outcome.config_verbatim == "name: arm\n"
    assert outcome.wall_time_s == 7.0


async def test_resuming_hands_the_loader_exactly_the_stored_ids() -> None:
    index = RecordingIndex({"chunk-a", "chunk-b"})
    engine = RecordingEngine(Counts(1, 2, 3, 4, 5))

    outcome = await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=TENANT,
        chunk_index=index,
        edges_ingested=False,
        config_verbatim="",
        clock=scripted_clock(100.0, 100.5, 102.5, 110.0),
    )

    assert index.asked_for == [TENANT], "the index is consulted once, for this tenant"
    assert engine.calls[0]["existing_chunk_ids"] == {"chunk-a", "chunk-b"}
    assert engine.calls[0]["resume"] is True
    assert outcome.resume is True
    # 102.5 - 100.5, and distinct from wall_time so a swap cannot pass.
    assert outcome.existing_ids_load_s == 2.0
    assert outcome.wall_time_s == 10.0


async def test_not_resuming_never_touches_the_index() -> None:
    engine = RecordingEngine(Counts(1, 2, 3, 4, 5))

    outcome = await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=TENANT,
        chunk_index=None,
        edges_ingested=False,
        config_verbatim="",
        clock=scripted_clock(100.0, 107.0),
    )

    assert engine.calls[0]["existing_chunk_ids"] == set()
    assert engine.calls[0]["resume"] is False
    assert outcome.resume is False
    assert outcome.existing_ids_load_s == 0.0


async def test_the_index_is_scoped_to_the_tenant_it_was_given() -> None:
    """A second tenant's ids must not be what the loader skips.

    `ids_for_tenant` takes the tenant, so an implementation that ignores it
    is one line away; this asserts the argument actually travels.
    """
    index = RecordingIndex({"x"})
    engine = RecordingEngine(Counts(1, 2, 3, 4, 5))

    await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=OTHER_TENANT,
        chunk_index=index,
        edges_ingested=False,
        config_verbatim="",
        clock=scripted_clock(0.0, 0.0, 0.0, 0.0),
    )

    assert index.asked_for == [OTHER_TENANT]
    assert engine.calls[0]["tenant_id"] == OTHER_TENANT


async def test_engine_kwargs_are_forwarded_untouched() -> None:
    engine = RecordingEngine(Counts(1, 2, 3, 4, 5))
    sentinel = object()

    await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=TENANT,
        chunk_index=None,
        edges_ingested=False,
        config_verbatim="",
        clock=scripted_clock(0.0, 0.0),
        chunker=sentinel,
        embed_batch=64,
    )

    assert engine.calls[0]["chunker"] is sentinel
    assert engine.calls[0]["embed_batch"] == 64

import logging

import pytest
from redstring import (
    FakeEmbeddingProvider,
    InMemoryChunkStore,
    InMemoryGraphStore,
    TenantId,
)
from redstring.domain.chunk import chunk_id
from redstring.domain.ids import SourceId
from uuid import uuid4

from stark_bench.adapters.stark_artifacts import SkbEdge, SkbNode
from stark_bench.domain.stark_ids import STARK_ID_KEY, entity_id_for
from stark_bench.adapters.stark_ingest_engine import ingest
from redstring.extraction.chunkers.sliding_window_chunker import SlidingWindowChunker
from stark_bench.adapters.chunkers import WholeDocumentChunker


class CountingEmbeddingProvider:
    """Wraps a real provider and counts calls to `embed`.

    A row-count assertion cannot distinguish "skipped" from "re-embedded and
    happened to write the same rows" -- this spy is what makes the
    distinction visible.
    """

    def __init__(self, inner: FakeEmbeddingProvider):
        self._inner = inner
        self.calls = 0
        self.texts_embedded: list[str] = []

    async def embed(self, texts):
        self.calls += 1
        self.texts_embedded.extend(texts)
        return await self._inner.embed(texts)


def _whole_doc_chunk_id(dataset: str, node: SkbNode):
    """The single chunk id `WholeDocumentChunker` produces for one node.

    `WholeDocumentChunker` makes a node a clean 1:1 with a chunk, so its id
    is exactly `chunk_id(source_id, node.document)`.
    """
    source_id = SourceId(f"{dataset}:{node.node_id}")
    return chunk_id(source_id, node.document)


async def _all_ids_in_store(chunks, tenant, source_ids) -> set:
    """One query's worth of ids per source, matching what a real store load
    would return -- used by tests as the caller-supplied `existing_chunk_ids`."""
    ids = set()
    for source_id in source_ids:
        for stored in await chunks.get_by_source(source_id, tenant):
            ids.add(stored.id)
    return ids


@pytest.fixture
def stores():
    return InMemoryGraphStore(), InMemoryChunkStore(dimension=8)


@pytest.mark.asyncio
async def test_it_writes_entities_carrying_their_stark_id(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [
        SkbNode("1", "drug", "aspirin", "a salicylate"),
        SkbNode("2", "gene", "PTGS2", "cyclooxygenase-2"),
    ]

    report = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.nodes == 2
    stored = await graph.get_entity(entity_id_for("prime", "1"), tenant)
    assert stored is not None
    assert stored.external_ids[STARK_ID_KEY] == "1"


@pytest.mark.asyncio
async def test_a_self_loop_is_dropped_and_counted(stores):
    """Redstring rejects self-loops. A silent drop would make a recall
    ceiling look like a retrieval failure, so the count is reported."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    edges = [SkbEdge("1", "1", "related_to")]

    report = await ingest(
        nodes,
        edges,
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.self_loops_dropped == 1
    assert report.edges == 0


@pytest.mark.asyncio
async def test_edges_referencing_unknown_nodes_do_not_abort_the_run(stores):
    """A bad edge followed by a good one.

    Stated this way on purpose: with only one bad edge at the end of the
    loop, `break` and `continue` are the same function, and a `break` would
    silently discard every later edge in a real corpus.
    """
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "x"), SkbNode("2", "gene", "PTGS2", "y")]
    edges = [SkbEdge("1", "999", "targets"), SkbEdge("1", "2", "targets")]

    report = await ingest(
        nodes,
        edges,
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.edges == 1


@pytest.mark.asyncio
async def test_ingest_is_idempotent(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    kwargs = dict(
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    first = await ingest(nodes, [], **kwargs)
    second = await ingest(nodes, [], **kwargs)

    assert first.nodes == second.nodes == 1
    assert len(await graph.find_entities(tenant)) == 1


@pytest.mark.asyncio
async def test_resume_skips_nodes_already_chunked(stores):
    """A second ingest of the same nodes, with the store's existing chunk
    ids supplied, must embed nothing and report them skipped."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [
        SkbNode("1", "drug", "aspirin", "a salicylate"),
        SkbNode("2", "gene", "PTGS2", "cyclooxygenase-2"),
    ]
    inner = FakeEmbeddingProvider(dimension=8)

    first_spy = CountingEmbeddingProvider(inner)
    first = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=first_spy,
    )
    assert first.chunks == 2
    assert first.skipped == 0
    assert first_spy.calls > 0

    existing = await _all_ids_in_store(
        chunks, tenant, [SourceId(f"prime:{n.node_id}") for n in nodes]
    )
    assert existing == {_whole_doc_chunk_id("prime", n) for n in nodes}

    second_spy = CountingEmbeddingProvider(inner)
    second = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=second_spy,
        existing_chunk_ids=existing,
    )

    assert second_spy.calls == 0
    assert second.chunks == 0
    assert second.skipped == 2
    assert second.nodes == 2


@pytest.mark.asyncio
async def test_resume_re_embeds_a_node_whose_text_changed(stores):
    """Chunk ids are content-addressed, so changed text is a different id
    and must not be skipped even when the node id is unchanged."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    original = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    changed = [SkbNode("1", "drug", "aspirin", "a salicylate, revised")]

    await ingest(
        original,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )
    existing = await _all_ids_in_store(chunks, tenant, [SourceId("prime:1")])

    spy = CountingEmbeddingProvider(FakeEmbeddingProvider(dimension=8))
    report = await ingest(
        changed,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=spy,
        existing_chunk_ids=existing,
    )

    assert spy.calls == 1
    assert report.chunks == 1
    assert report.skipped == 0


@pytest.mark.asyncio
async def test_resume_false_re_embeds_everything(stores):
    """`resume=False` ignores `existing_chunk_ids` and re-embeds every node --
    the escape hatch for a deliberate full re-ingest."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]

    await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )
    existing = await _all_ids_in_store(chunks, tenant, [SourceId("prime:1")])
    assert existing  # sanity: the store really has the chunk

    spy = CountingEmbeddingProvider(FakeEmbeddingProvider(dimension=8))
    report = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=spy,
        existing_chunk_ids=existing,
        resume=False,
    )

    assert spy.calls == 1
    assert report.chunks == 1
    assert report.skipped == 0


@pytest.mark.asyncio
async def test_resume_resumes_a_partial_prior_ingest(stores):
    """3 nodes ingested, then 5 including those 3: exactly 2 embedded, 3
    skipped."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    first_three = [
        SkbNode("1", "drug", "aspirin", "a"),
        SkbNode("2", "drug", "ibuprofen", "b"),
        SkbNode("3", "drug", "naproxen", "c"),
    ]
    all_five = first_three + [
        SkbNode("4", "gene", "PTGS1", "d"),
        SkbNode("5", "gene", "PTGS2", "e"),
    ]

    await ingest(
        first_three,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )
    existing = await _all_ids_in_store(
        chunks, tenant, [SourceId(f"prime:{n.node_id}") for n in first_three]
    )

    spy = CountingEmbeddingProvider(FakeEmbeddingProvider(dimension=8))
    report = await ingest(
        all_five,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=spy,
        existing_chunk_ids=existing,
    )

    assert report.nodes == 5
    assert report.chunks == 2
    assert report.skipped == 3
    assert spy.texts_embedded == ["d", "e"]


class ManyChunksPerNode:
    """A chunker that splits every document into many pieces.

    Exercises the case `BATCH` (entity count) cannot see: few nodes, each
    contributing far more than one chunk, so `chunk_batch` can grow past a
    flush threshold expressed only in nodes. `BoundaryPreferenceChunker`
    behaves this way on real documents; this fake makes the shape
    deterministic and cheap for a unit test.
    """

    def __init__(self, chunks_per_node: int):
        self._n = chunks_per_node

    @property
    def chunker_type(self) -> str:
        return "many-chunks-per-node"

    def chunk(self, text, max_chunk_size=None, overlap_size=None):
        from redstring.extraction.chunking import Chunk, ChunkingResult

        pieces = [
            Chunk(text=f"{text}-{i}", chunk_index=i, start_char=0, end_char=1)
            for i in range(self._n)
        ]
        return ChunkingResult(
            chunks=pieces,
            total_chunks=self._n,
            original_length=len(text),
            chunking_method="many-chunks-per-node",
            overlap_size=0,
        )


class RecordingChunkStore:
    """Wraps a real chunk store and records the size of every `upsert_many`
    call, so a test can assert the flush threshold was honoured without
    needing a real Postgres payload limit to trip it."""

    def __init__(self, inner):
        self._inner = inner
        self.call_sizes: list[int] = []

    async def upsert_many(self, chunks):
        self.call_sizes.append(len(chunks))
        await self._inner.upsert_many(chunks)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.asyncio
async def test_chunk_batch_is_flushed_independently_of_entity_batch(stores):
    """Few nodes, many chunks each: `upsert_many` must never be called with
    more chunks than `CHUNK_BATCH`, even though the entity-count threshold
    (`BATCH`) would not fire until far more nodes had been processed.

    This is the regression for the real ingest failure: a run resuming with
    `BoundaryPreferenceChunker` sent `chunk_batch` well past Postgres's
    jsonb-array size limit because only `len(batch) >= BATCH` triggered a
    flush.
    """
    from stark_bench.adapters.stark_ingest_engine import CHUNK_BATCH

    graph, real_chunks = stores
    chunks = RecordingChunkStore(real_chunks)
    tenant = TenantId(uuid4())
    # BATCH is 500 entities; 10 nodes each producing far more chunks than
    # CHUNK_BATCH would, without the chunk-count flush, accumulate one
    # enormous chunk_batch across the whole run.
    chunks_per_node = CHUNK_BATCH // 2
    nodes = [SkbNode(str(i), "drug", f"node{i}", f"doc{i}") for i in range(10)]

    report = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=ManyChunksPerNode(chunks_per_node),
        embeddings=FakeEmbeddingProvider(dimension=8),
    )

    assert report.chunks == 10 * chunks_per_node
    assert chunks.call_sizes, "upsert_many was never called"
    assert all(size <= CHUNK_BATCH for size in chunks.call_sizes), chunks.call_sizes


@pytest.mark.asyncio
async def test_it_reports_a_final_progress_line_with_the_counts(stores, caplog):
    """The summary line must carry the numbers, not just say it finished.

    A log line that says "ingest done" and nothing else is what this
    replaced: it cannot answer whether the corpus is complete, which is the
    one question asked of it.

    It says "nodes done" rather than "done" because the edge phase runs
    AFTER it and takes ~28 minutes on PRIME (B-EDGE-PROGRESS-1). A line
    claiming the ingest had finished, half an hour before it had, is what
    produced a confident wrong diagnosis of a hang.
    """
    graph, chunks = stores
    nodes = [SkbNode(str(i), "drug", f"d{i}", f"doc {i}") for i in range(3)]

    with caplog.at_level(
        logging.INFO, logger="stark_bench.adapters.stark_ingest_engine"
    ):
        await ingest(
            nodes,
            [],
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=FakeEmbeddingProvider(dimension=8),
            total_nodes=3,
        )

    done = [
        r.getMessage() for r in caplog.records if "ingest nodes done" in r.getMessage()
    ]
    assert len(done) == 1, f"expected exactly one summary line, got {done}"
    assert "3/3 nodes" in done[0]
    assert "3 chunks" in done[0]


@pytest.mark.asyncio
async def test_progress_survives_an_unknown_node_total(stores, caplog):
    """`total_nodes=None` must log, not raise.

    The count comes from reading nodes.jsonl, which is allowed to fail --
    the whole point of returning None there is that a cosmetic figure can
    never take down an ingest. If the formatting divided by the total this
    would be a ZeroDivisionError on the happy path of a degraded read.
    """
    graph, chunks = stores

    with caplog.at_level(
        logging.INFO, logger="stark_bench.adapters.stark_ingest_engine"
    ):
        await ingest(
            [SkbNode("1", "drug", "aspirin", "a salicylate")],
            [],
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=FakeEmbeddingProvider(dimension=8),
            total_nodes=None,
        )

    done = [
        r.getMessage() for r in caplog.records if "ingest nodes done" in r.getMessage()
    ]
    assert len(done) == 1
    assert "1/? nodes" in done[0]
    assert "%" not in done[0], "no percentage is claimable without a total"


class RejectsLongTexts:
    """An embedding provider with a token limit, standing in for llama.cpp.

    Rejects on *characters* because the test needs a deterministic threshold,
    but raises the message a real server sends so the engine's matching is
    exercised rather than a test-only sentinel.
    """

    def __init__(self, inner, limit: int):
        self._inner = inner
        self._limit = limit
        self.rejections = 0

    async def embed(self, texts):
        for text in texts:
            if len(text) > self._limit:
                self.rejections += 1
                raise RuntimeError(
                    "the embeddings client raised BadRequestError: Error code: 400 - "
                    "{'error': {'message': 'input (2088 tokens) is larger than the max "
                    "context size (2048 tokens). skipping', "
                    "'type': 'exceed_context_size_error'}}"
                )
        return await self._inner.embed(texts)


@pytest.mark.asyncio
async def test_a_rejected_chunk_is_re_chunked_smaller_and_the_ingest_completes(stores):
    """The cap being wrong must cost a retry, not the run.

    Three ingests were lost to a character cap that did not bound tokens.
    """
    graph, chunks = stores
    provider = RejectsLongTexts(FakeEmbeddingProvider(dimension=8), limit=60)
    nodes = [SkbNode("1", "drug", "aspirin", "x" * 500)]

    report = await ingest(
        nodes,
        [],
        dataset="prime",
        tenant_id=TenantId(uuid4()),
        graph=graph,
        chunks=chunks,
        chunker=SlidingWindowChunker(default_chunk_size=400, default_overlap=0),
        embeddings=provider,
    )

    assert provider.rejections >= 1, "the oversize path was never exercised"
    assert report.nodes == 1
    assert report.chunks > 1, "a 500-char doc under a 60-char limit must split"


@pytest.mark.asyncio
async def test_an_unrelated_provider_error_is_not_retried(stores):
    """Only length rejections re-chunk. Everything else propagates.

    Retrying an auth failure or a dropped connection four times at halved
    chunk sizes would turn one clear error into four confusing ones.
    """
    graph, chunks = stores

    class Broken:
        def __init__(self):
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            raise RuntimeError("Error code: 401 - invalid api key")

    provider = Broken()

    with pytest.raises(RuntimeError, match="401"):
        await ingest(
            [SkbNode("1", "drug", "aspirin", "a salicylate")],
            [],
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=provider,
        )

    assert provider.calls == 1, f"retried a non-length error {provider.calls} times"


@pytest.mark.asyncio
async def test_the_node_summary_does_not_claim_the_whole_ingest_finished(
    stores, caplog
):
    """B-EDGE-PROGRESS-1. The edge phase runs after this line and is ~28
    minutes on PRIME. A message saying the work is finished, half an hour
    before it is, is not a missing feature -- it is the log actively lying,
    and it cost a real debugging detour on 2026-08-19."""
    graph, chunks = stores
    nodes = [SkbNode(str(i), "drug", f"d{i}", f"doc {i}") for i in range(2)]

    with caplog.at_level(
        logging.INFO, logger="stark_bench.adapters.stark_ingest_engine"
    ):
        await ingest(
            nodes,
            [],
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=FakeEmbeddingProvider(dimension=8),
            total_nodes=2,
        )

    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "ingest done" in m], (
        "the node phase must not announce the ingest as done; the edge "
        f"phase has not run yet. Got: {messages}"
    )


@pytest.mark.asyncio
async def test_the_edge_phase_reports_when_it_finishes(stores, caplog):
    """It logged nothing at all before -- ~28 minutes of silence after a
    line reading 'done'."""
    graph, chunks = stores
    nodes = [SkbNode(str(i), "drug", f"d{i}", f"doc {i}") for i in range(3)]
    edges = [SkbEdge("0", "1", "targets"), SkbEdge("1", "2", "targets")]

    with caplog.at_level(
        logging.INFO, logger="stark_bench.adapters.stark_ingest_engine"
    ):
        await ingest(
            nodes,
            edges,
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=FakeEmbeddingProvider(dimension=8),
            total_nodes=3,
        )

    done = [r.getMessage() for r in caplog.records if "edges done" in r.getMessage()]
    assert len(done) == 1, f"expected one edge summary, got {done}"
    assert "2 relationships" in done[0]


@pytest.mark.asyncio
async def test_no_edges_means_no_edge_summary(stores, caplog):
    """A run without `--ingest-edges` must not print an edge line claiming
    zero -- it did not do the phase, which is different from doing it and
    finding nothing."""
    graph, chunks = stores
    nodes = [SkbNode("0", "drug", "d", "doc")]

    with caplog.at_level(
        logging.INFO, logger="stark_bench.adapters.stark_ingest_engine"
    ):
        await ingest(
            nodes,
            [],
            dataset="prime",
            tenant_id=TenantId(uuid4()),
            graph=graph,
            chunks=chunks,
            chunker=WholeDocumentChunker(),
            embeddings=FakeEmbeddingProvider(dimension=8),
            total_nodes=1,
        )

    assert not [r for r in caplog.records if "edges done" in r.getMessage()]

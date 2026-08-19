"""Batching texts across nodes must not misalign vectors with their chunks.

This is the defect the whole file exists for. If a vector lands on the wrong
chunk, the store is fully populated, every count is right, no exception is
raised anywhere -- and retrieval scores like noise. It is indistinguishable
from "the embedding model is bad", which is a conclusion this project has
nearly drawn once already for a different reason.

So the provider here returns a vector that **encodes its own input text**,
and the assertions check chunk-by-chunk that the stored vector is the one
belonging to that chunk's text. A provider returning arbitrary vectors could
not tell a correct implementation from one that shuffles.
"""

from __future__ import annotations

import pytest
from redstring import InMemoryChunkStore, InMemoryGraphStore, TenantId
from redstring.domain.ids import SourceId
from uuid import uuid4

from stark_bench.adapters.stark_artifacts import SkbNode
from redstring.extraction.chunkers.sliding_window_chunker import SlidingWindowChunker
from stark_bench.adapters.chunkers import WholeDocumentChunker
from stark_bench.adapters.stark_ingest_engine import ingest

DIM = 8


def _fingerprint(text: str) -> list[float]:
    """A vector that identifies its input, so misalignment is visible.

    Deliberately not a hash into a small space: the first component is the
    text length and the rest are ordinals, so two different texts cannot
    collide into the same vector and make a shuffle look correct.
    """
    body = [float(ord(c)) for c in text[: DIM - 1]]
    return [float(len(text)), *body, *([0.0] * (DIM - 1 - len(body)))][:DIM]


class FingerprintingProvider:
    """Returns a vector derived from each input, and records batch shapes."""

    model = "fingerprint"
    dimension = DIM

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed(self, texts):
        self.batch_sizes.append(len(texts))
        return [_fingerprint(t) for t in texts]

    async def embed_query(self, texts):
        return await self.embed(texts)


class ShufflingProvider(FingerprintingProvider):
    """A provider that violates the port's positional contract.

    Exists to prove the alignment assertions can fail. If a test passes
    against this, it is not testing alignment.
    """

    async def embed(self, texts):
        vectors = await super().embed(texts)
        return vectors[::-1]


@pytest.fixture
def tenant():
    return TenantId(uuid4())


def _nodes(n: int, *, long_every: int = 0) -> list[SkbNode]:
    """Nodes with DISTINCT documents of DIFFERING lengths.

    Both properties are load-bearing. Identical documents would collide to
    one chunk id and hide any reordering; equal-length documents would make
    a per-node slice of the wrong width still land inside the right node.
    """
    out = []
    for i in range(n):
        body = f"node-{i:04d}-" + ("x" * (i % 37 + 1))
        if long_every and i % long_every == 0:
            body = body + " " + ("y" * 4000)  # forces multiple chunks
        out.append(
            SkbNode(node_id=str(i), node_type="gene", name=f"n{i}", document=body)
        )
    return out


async def _run(nodes, tenant, provider, chunker, **kwargs):
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=DIM)
    report = await ingest(
        nodes,
        iter(()),
        dataset="prime",
        tenant_id=tenant,
        graph=graph,
        chunks=chunks,
        chunker=chunker,
        embeddings=provider,
        resume=False,
        **kwargs,
    )
    return report, chunks


async def _assert_every_vector_matches_its_text(chunks, tenant, nodes):
    seen = 0
    for node in nodes:
        source_id = SourceId(f"prime:{node.node_id}")
        for stored in await chunks.get_by_source(source_id, tenant):
            assert stored.embedding == _fingerprint(
                stored.text
            ), f"node {node.node_id}: vector does not belong to its chunk text"
            seen += 1
    assert seen > 0, "no chunks stored -- the assertion would pass vacuously"
    return seen


@pytest.mark.asyncio
async def test_vectors_stay_with_their_chunks_when_batched(tenant):
    """The headline property, over more nodes than one batch holds."""
    nodes = _nodes(150)
    provider = FingerprintingProvider()
    _, chunks = await _run(
        nodes, tenant, provider, WholeDocumentChunker(), concurrency=4, embed_batch=16
    )
    await _assert_every_vector_matches_its_text(chunks, tenant, nodes)


@pytest.mark.asyncio
async def test_alignment_holds_when_nodes_produce_different_chunk_counts(tenant):
    """The case a per-node slice gets wrong.

    With every node producing exactly one chunk, a cursor that advances by a
    constant is indistinguishable from one that advances by `len(pieces)`.
    Multi-chunk nodes interleaved with single-chunk ones is what separates
    them, and it is the shape this corpus actually has.
    """
    nodes = _nodes(120, long_every=7)
    provider = FingerprintingProvider()
    _, chunks = await _run(
        nodes,
        tenant,
        provider,
        SlidingWindowChunker(default_chunk_size=1000, default_overlap=0),
        concurrency=3,
        embed_batch=8,
    )
    stored = await _assert_every_vector_matches_its_text(chunks, tenant, nodes)
    assert stored > len(nodes), "no node produced multiple chunks -- test is vacuous"


@pytest.mark.asyncio
async def test_the_assertion_catches_a_provider_that_reorders(tenant):
    """Proves the two tests above are testing something."""
    nodes = _nodes(40)
    with pytest.raises(AssertionError, match="does not belong"):
        _, chunks = await _run(
            nodes,
            tenant,
            ShufflingProvider(),
            WholeDocumentChunker(),
            concurrency=2,
            embed_batch=8,
        )
        await _assert_every_vector_matches_its_text(chunks, tenant, nodes)


@pytest.mark.asyncio
async def test_texts_are_actually_batched(tenant):
    """The point of the change: requests must carry many texts, not one.

    Without this, an implementation that quietly kept one request per node
    would pass every alignment test above -- correct, and six times slower,
    which is exactly the state this change is fixing.
    """
    provider = FingerprintingProvider()
    await _run(
        _nodes(150),
        tenant,
        provider,
        WholeDocumentChunker(),
        concurrency=4,
        embed_batch=16,
    )
    assert max(provider.batch_sizes) > 1, "every request carried one text"
    assert max(provider.batch_sizes) <= 16, "a request exceeded embed_batch"


@pytest.mark.asyncio
async def test_a_short_final_batch_is_not_dropped(tenant):
    """37 nodes at embed_batch 16 leaves a remainder of 5."""
    nodes = _nodes(37)
    provider = FingerprintingProvider()
    report, chunks = await _run(
        nodes, tenant, provider, WholeDocumentChunker(), concurrency=2, embed_batch=16
    )
    assert report.nodes == 37
    assert await _assert_every_vector_matches_its_text(chunks, tenant, nodes) == 37


@pytest.mark.asyncio
async def test_embed_batch_below_one_is_refused(tenant):
    with pytest.raises(ValueError, match="embed_batch"):
        await _run(
            _nodes(2),
            tenant,
            FingerprintingProvider(),
            WholeDocumentChunker(),
            embed_batch=0,
        )

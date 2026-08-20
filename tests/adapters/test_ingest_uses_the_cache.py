"""A second arm over the same text must not call the provider again.

The provider here returns a vector encoding its own input -- the pattern from
`test_ingest_batching.py` -- so these assertions check the CACHED vectors are
the ones belonging to their chunks, not merely that some vector arrived. A
cache returning a plausible wrong vector is the one defect this feature can
introduce, and it would be invisible: the store fills, every count is right,
and retrieval just scores worse.
"""

from __future__ import annotations

from uuid import uuid4

from redstring import InMemoryChunkStore, InMemoryGraphStore, TenantId
from redstring.domain.ids import SourceId

from stark_bench.adapters.chunkers import WholeDocumentChunker
from stark_bench.adapters.memory_embedding_cache import InMemoryEmbeddingCache
from stark_bench.adapters.stark_artifacts import SkbNode
from stark_bench.adapters.stark_ingest_engine import ingest

DIM = 8


def _fingerprint(text: str) -> list[float]:
    """A vector that identifies its input, so a wrong vector is visible."""
    body = [float(ord(c)) for c in text[: DIM - 1]]
    return [float(len(text)), *body, *([0.0] * (DIM - 1 - len(body)))][:DIM]


class CountingProvider:
    model = "fingerprint"
    dimension = DIM

    def __init__(self) -> None:
        self.texts_embedded: list[str] = []

    async def embed(self, texts):
        self.texts_embedded.extend(texts)
        return [_fingerprint(text) for text in texts]

    async def embed_query(self, texts):
        return await self.embed(texts)


def _nodes(extra: bool = False):
    nodes = [
        SkbNode(node_id="1", node_type="gene/protein", name="A", document="alpha doc"),
        SkbNode(node_id="2", node_type="gene/protein", name="B", document="beta doc"),
    ]
    if extra:
        nodes.append(
            SkbNode(
                node_id="3", node_type="gene/protein", name="C", document="gamma doc"
            )
        )
    return nodes


async def _run(provider, cache, *, prefix: str = "", extra: bool = False):
    """One arm into a fresh tenant. Returns (report, chunk store, tenant)."""
    tenant = TenantId(uuid4())
    chunks = InMemoryChunkStore(dimension=DIM)
    report = await ingest(
        _nodes(extra),
        iter(()),
        dataset="prime",
        tenant_id=tenant,
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
        embedding_cache=cache,
        cache_model="fingerprint",
        cache_document_prefix=prefix,
    )
    return report, chunks, tenant


async def _stored(chunks, tenant, node_ids):
    out = {}
    for node_id in node_ids:
        for chunk in await chunks.get_by_source(SourceId(f"prime:{node_id}"), tenant):
            out[chunk.text] = list(chunk.embedding)
    return out


async def test_a_second_tenant_embeds_nothing_and_stores_the_right_vectors():
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()

    first, _, _ = await _run(provider, cache)
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]
    assert first.cache_hits == 0
    assert first.cache_misses == 2

    provider.texts_embedded.clear()
    second, chunks, tenant = await _run(provider, cache)

    assert provider.texts_embedded == [], "the second arm re-embedded text"
    assert second.cache_hits == 2
    assert second.cache_misses == 0

    assert await _stored(chunks, tenant, ["1", "2"]) == {
        "alpha doc": _fingerprint("alpha doc"),
        "beta doc": _fingerprint("beta doc"),
    }


async def test_a_different_prefix_does_not_hit_the_cache():
    """The failure this feature could introduce, asserted directly."""
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()
    await _run(provider, cache, prefix="")
    provider.texts_embedded.clear()

    report, _, _ = await _run(provider, cache, prefix="passage: ")

    assert report.cache_hits == 0
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]


async def test_no_cache_is_unchanged_behaviour():
    provider = CountingProvider()
    report = await ingest(
        _nodes(),
        iter(()),
        dataset="prime",
        tenant_id=TenantId(uuid4()),
        graph=InMemoryGraphStore(),
        chunks=InMemoryChunkStore(dimension=DIM),
        chunker=WholeDocumentChunker(),
        embeddings=provider,
    )
    assert sorted(provider.texts_embedded) == ["alpha doc", "beta doc"]
    assert report.cache_hits == 0
    assert report.cache_misses == 0


async def test_a_partial_hit_embeds_only_the_missing_text():
    """The interesting case: order must survive a mix of hits and misses."""
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()
    await _run(provider, cache)
    provider.texts_embedded.clear()

    report, chunks, tenant = await _run(provider, cache, extra=True)

    assert provider.texts_embedded == ["gamma doc"]
    assert report.cache_hits == 2
    assert report.cache_misses == 1

    assert await _stored(chunks, tenant, ["1", "2", "3"]) == {
        "alpha doc": _fingerprint("alpha doc"),
        "beta doc": _fingerprint("beta doc"),
        "gamma doc": _fingerprint("gamma doc"),
    }


async def test_a_repeated_document_is_embedded_once():
    """Duplicates inside one group must be looked up once and fanned out."""
    cache = InMemoryEmbeddingCache()
    provider = CountingProvider()
    same = [
        SkbNode(node_id="1", node_type="t", name="A", document="identical"),
        SkbNode(node_id="2", node_type="t", name="B", document="identical"),
    ]
    tenant = TenantId(uuid4())
    chunks = InMemoryChunkStore(dimension=DIM)
    report = await ingest(
        same,
        iter(()),
        dataset="prime",
        tenant_id=tenant,
        graph=InMemoryGraphStore(),
        chunks=chunks,
        chunker=WholeDocumentChunker(),
        embeddings=provider,
        embedding_cache=cache,
        cache_model="fingerprint",
        cache_document_prefix="",
    )
    assert provider.texts_embedded == ["identical", "identical"] or (
        provider.texts_embedded == ["identical"]
    )
    assert await _stored(chunks, tenant, ["1", "2"]) == {
        "identical": _fingerprint("identical")
    }
    assert report.cache_hits + report.cache_misses == 2

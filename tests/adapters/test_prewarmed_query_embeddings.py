"""The query side is embedded once, in batches, and never twice.

Every test here has been checked by breaking the implementation on purpose;
the deliberate defect each one catches is named in its docstring, because a
test whose failure mode is unknown is a test nobody can trust.
"""

from __future__ import annotations

import pytest

from stark_bench.adapters.prewarmed_query_embeddings import PrewarmedQueryEmbeddings


class SpyProvider:
    """Records every call, and returns a vector that identifies its text.

    The vector encodes `len(text)` rather than a constant so that a wrapper
    which returned the *wrong* prewarmed vector for a text is distinguishable
    from one that returned the right one. A provider handing back identical
    vectors would let a misaligned zip pass.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.embed_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return "spy-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str, side: float) -> list[float]:
        return [side, float(len(text)), float(sum(map(ord, text)) % 97), 1.0]

    async def embed(self, texts):
        self.embed_calls.append(list(texts))
        # Side marker 0.0 vs 1.0: a wrapper that served a *document*-side
        # vector for a query would be invisible if both sides agreed, and
        # that is precisely the ADR 0043 failure -- a plausible cosine, no
        # error.
        return [self._vector(t, 0.0) for t in texts]

    async def embed_query(self, texts):
        self.query_calls.append(list(texts))
        return [self._vector(t, 1.0) for t in texts]


@pytest.fixture
def spy() -> SpyProvider:
    return SpyProvider()


async def test_prewarming_replaces_per_query_round_trips(spy):
    """Catches: forgetting to consult the map, i.e. the whole point.

    Deliberate defect: `embed_query` delegating unconditionally. Then
    `query_calls` is 4 rather than 1 and `live_calls` is 3 rather than 0.
    """
    wrapper = PrewarmedQueryEmbeddings(spy, batch_size=128)
    queries = ["what targets ACE2", "a drug for asthma", "gene near BRCA1"]

    await wrapper.prewarm(queries)
    for query in queries:
        await wrapper.embed_query([query])

    assert len(spy.query_calls) == 1, "one batched request, not one per query"
    assert spy.query_calls[0] == queries
    assert wrapper.live_calls == 0
    assert wrapper.hits == 3
    assert wrapper.misses == 0


async def test_a_prewarmed_vector_is_the_one_the_provider_gave(spy):
    """Catches: serving a stale, zeroed, or misaligned vector.

    An all-zero vector would score plausibly and fail nothing else.
    """
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.prewarm(["alpha", "beta gamma"])

    served = await wrapper.embed_query(["beta gamma", "alpha"])

    assert served == [
        spy._vector("beta gamma", 1.0),
        spy._vector("alpha", 1.0),
    ]


async def test_the_corpus_side_is_never_served_from_the_query_map(spy):
    """Catches: one cache spanning both sides -- the ADR 0043 hazard.

    Deliberate defect: `embed` delegating to `embed_query`, or sharing the
    map. The side marker then flips and this fails; without the marker the
    two sides are indistinguishable and the test proves nothing.
    """
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.prewarm(["shared text"])

    as_document = await wrapper.embed(["shared text"])

    assert as_document == [spy._vector("shared text", 0.0)]
    assert spy.embed_calls == [["shared text"]], "corpus side must reach the provider"


async def test_one_vector_per_input_in_order_including_duplicates(spy):
    """Catches: deduplicating the *response*, not just the request.

    The port promises `len(result) == len(texts)`. A wrapper returning the
    deduplicated list would hand back 2 vectors for 4 texts, and a caller
    zipping them onto queries would bind the wrong vector to the wrong query
    for the rest of the run -- `EmbeddingProviderError` exists for this.
    """
    wrapper = PrewarmedQueryEmbeddings(spy)
    texts = ["a", "b", "a", "b"]
    await wrapper.prewarm(texts)

    served = await wrapper.embed_query(texts)

    assert len(served) == 4
    assert served[0] == served[2] == spy._vector("a", 1.0)
    assert served[1] == served[3] == spy._vector("b", 1.0)
    assert spy.query_calls == [
        ["a", "b"]
    ], "the request may dedupe; the response may not"


async def test_an_unseen_text_is_embedded_live_rather_than_refused(spy):
    """Catches: raising on a miss.

    `deep` invents sub-queries that were never in the query set. Refusing
    them would make this wrapper an architecture change wearing an
    optimisation's clothes.
    """
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.prewarm(["known"])

    served = await wrapper.embed_query(["invented sub-query"])

    assert served == [spy._vector("invented sub-query", 1.0)]
    assert wrapper.live_calls == 1
    assert wrapper.misses == 1
    assert wrapper.hits == 0


async def test_a_live_miss_is_remembered(spy):
    """Catches: embedding the same sub-query on every one of N iterations."""
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.embed_query(["invented"])
    await wrapper.embed_query(["invented"])

    assert wrapper.live_calls == 1
    assert wrapper.hits == 1


async def test_prewarming_batches_at_the_configured_size(spy):
    """Catches: a batch size that is read but not used.

    17 texts at 5 per batch is 4 requests of 5, 5, 5, 2 -- sizes that a
    round number could not distinguish from an off-by-one.
    """
    wrapper = PrewarmedQueryEmbeddings(spy, batch_size=5)
    texts = [f"query number {n}" for n in range(17)]

    await wrapper.prewarm(texts)

    assert [len(call) for call in spy.query_calls] == [5, 5, 5, 2]
    assert wrapper.prewarm_requests == 4
    assert wrapper.prewarm_texts == 17


async def test_prewarming_twice_costs_nothing(spy):
    """Catches: a prewarm that re-embeds what it already holds."""
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.prewarm(["a", "b"])
    await wrapper.prewarm(["a", "b"])

    assert len(spy.query_calls) == 1


async def test_prewarming_only_the_new_texts(spy):
    """Catches: an all-or-nothing prewarm that re-sends the whole set."""
    wrapper = PrewarmedQueryEmbeddings(spy)
    await wrapper.prewarm(["a"])
    await wrapper.prewarm(["a", "b"])

    assert spy.query_calls == [["a"], ["b"]]


async def test_a_short_response_is_refused_rather_than_zipped(spy):
    """Catches: trusting the provider's count.

    An adapter that dedupes or drops a failed text internally returns fewer
    vectors than inputs; `zip` would silently truncate and bind vectors to
    the wrong texts.
    """

    async def short(texts):
        return [[1.0, 1.0, 1.0, 1.0]]

    spy.embed_query = short
    wrapper = PrewarmedQueryEmbeddings(spy)

    with pytest.raises(ValueError, match="refusing to guess the alignment"):
        await wrapper.prewarm(["a", "b", "c"])


async def test_model_and_dimension_come_from_the_wrapped_provider(spy):
    """Catches: a wrapper that invents provenance.

    The model name is written next to stored vectors; a wrapper reporting
    its own name would misattribute every one.
    """
    wrapper = PrewarmedQueryEmbeddings(spy)

    assert wrapper.model == "spy-model"
    assert wrapper.dimension == 4


async def test_empty_input_makes_no_request(spy):
    wrapper = PrewarmedQueryEmbeddings(spy)

    assert await wrapper.embed_query([]) == []
    await wrapper.prewarm([])

    assert spy.query_calls == []


async def test_the_stats_distinguish_a_served_vector_from_a_bought_one(spy):
    """Catches: reporting hits without reporting round-trips.

    `hits` alone cannot tell a prewarmed run from an unprewarmed one, which
    is the number the report exists to prove.
    """
    wrapper = PrewarmedQueryEmbeddings(spy, batch_size=2)
    await wrapper.prewarm(["a", "b", "c"])
    await wrapper.embed_query(["a"])
    await wrapper.embed_query(["z"])

    assert wrapper.stats() == {
        "query_embed_prewarm_texts": 3,
        "query_embed_prewarm_requests": 2,
        "query_embed_hits": 1,
        "query_embed_misses": 1,
        "query_embed_live_calls": 1,
    }

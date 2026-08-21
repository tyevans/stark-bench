"""`rank_texts` scores arbitrary strings; the adapter owns the mechanism.

An agent cannot embed -- `Toolset` is its whole world and importing
`harness` is forbidden by contract. So the capability goes on the port, and
the two-sided prefix rule (ADR 0043) stays here where it cannot be
forgotten: the query goes through `embed_query`, the texts through `embed`.
"""

from __future__ import annotations

import pytest

from stark_bench.adapters.redstring_toolset import (
    RedstringToolset,
    _bm25_scores,
    _cosine,
    _rrf,
)


class SpyEmbeddings:
    """Marks which side each text went through, and counts requests."""

    model = "spy"
    dimension = 2

    def __init__(self) -> None:
        self.doc_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []

    async def embed(self, texts):
        self.doc_batches.append(list(texts))
        return [[1.0, float(len(t))] for t in texts]

    async def embed_query(self, texts):
        self.query_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeChunks:
    """Enough of a chunk store for `ChunkRetriever` to construct. None of
    these tests retrieve; they exercise `rank_texts` only."""

    dimension = 2

    async def search(self, *a, **k):  # pragma: no cover - never called
        return []


def _toolset(embeddings):
    return RedstringToolset(
        chunks=FakeChunks(),
        graph=object(),
        embeddings=embeddings,
        tenant_id="t",
        dataset="d",
    )


async def test_texts_use_the_document_side_and_the_query_the_query_side() -> None:
    """ADR 0043: an asymmetric model takes different prefixes per side, and
    a corpus embedded with the wrong one still returns plausible numbers."""
    spy = SpyEmbeddings()
    await _toolset(spy).rank_texts("q", ["a", "bb"])
    assert spy.doc_batches == [["a", "bb"]]
    assert spy.query_batches == [["q"]]


async def test_one_score_per_input_in_input_order_including_duplicates() -> None:
    spy = SpyEmbeddings()
    got = await _toolset(spy).rank_texts("q", ["a", "bb", "a"])
    assert len(got) == 3
    assert got[0] == got[2]


async def test_a_repeated_text_is_embedded_once() -> None:
    spy = SpyEmbeddings()
    await _toolset(spy).rank_texts("q", ["a", "a", "a"])
    assert spy.doc_batches == [["a"]]


async def test_texts_are_memoised_across_calls() -> None:
    """47,318 distinct names serve ~3,300 per query over 280 queries."""
    spy = SpyEmbeddings()
    tools = _toolset(spy)
    await tools.rank_texts("q", ["a", "b"])
    await tools.rank_texts("other", ["a", "b", "c"])
    assert spy.doc_batches == [["a", "b"], ["c"]]


async def test_lexical_mode_touches_no_endpoint() -> None:
    """What makes the channel comparison cheap enough to bother running."""
    spy = SpyEmbeddings()
    await _toolset(spy).rank_texts("q", ["a", "b"], mode="lexical")
    assert spy.doc_batches == [] and spy.query_batches == []


async def test_an_unknown_mode_raises_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="mode"):
        await _toolset(SpyEmbeddings()).rank_texts("q", ["a"], mode="dnese")


async def test_empty_texts_makes_no_call() -> None:
    spy = SpyEmbeddings()
    assert await _toolset(spy).rank_texts("q", []) == []
    assert spy.query_batches == []


async def test_the_call_is_recorded_for_the_cost_column() -> None:
    tools = _toolset(SpyEmbeddings())
    await tools.rank_texts("q", ["a", "b"])
    assert [c.tool for c in tools.calls] == ["rank_texts"]


async def test_a_provider_returning_the_wrong_count_raises() -> None:
    class Short(SpyEmbeddings):
        async def embed(self, texts):
            return [[1.0, 1.0]]

    with pytest.raises(ValueError, match="refusing to guess"):
        await _toolset(Short()).rank_texts("q", ["a", "b"])


def test_bm25_ranks_the_query_named_text_first() -> None:
    scores = _bm25_scores("DCC signalling", ["RAC1", "DCC"])
    assert scores[1] > scores[0]


def test_bm25_without_query_terms_is_flat() -> None:
    assert _bm25_scores("", ["a", "b"]) == [0.0, 0.0]


def test_cosine_handles_a_zero_vector() -> None:
    """Raising would take down a run over one degenerate embedding."""
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_rrf_is_not_dominated_by_a_differently_scaled_channel() -> None:
    """The reason for ranks over a weighted sum: BM25 is unbounded and
    cosine sits on [-1, 1], so summing makes the weight corpus-dependent."""
    huge = [1000.0, 0.0]
    small = [0.0, 0.001]
    fused = _rrf(huge, small)
    assert abs(fused[0] - fused[1]) < 1e-9


def test_rrf_lets_a_flat_channel_drop_out() -> None:
    flat = [1.0, 1.0, 1.0]
    signal = [3.0, 2.0, 1.0]
    fused = _rrf(flat, signal)
    assert fused[0] > fused[1] > fused[2]


async def test_hybrid_fuses_by_rank_not_by_adding_the_raw_scores() -> None:
    """Asserted on `rank_texts`, not on `_rrf`.

    A test of the helper cannot see the call site swapping it for `d + l` --
    the defect this repo hits repeatedly. RRF over two channels is bounded
    by `2 / (_RRF_K + 1)`; a raw sum of cosine (<=1) and BM25 (unbounded,
    grows with idf) is not, so the bound distinguishes the two
    implementations for any input.
    """
    from stark_bench.adapters.redstring_toolset import _RRF_K

    texts = ["zz", "qqq alpha beta gamma delta epsilon zeta eta theta"]
    got = await _toolset(SpyEmbeddings()).rank_texts("qqq", texts, mode="hybrid")
    ceiling = 2.0 / (_RRF_K + 1)
    assert all(0.0 < score <= ceiling + 1e-9 for score in got), got


async def test_dense_mode_returns_cosine_not_fused_ranks() -> None:
    """`...dense` must isolate the channel. Fusing a single channel would
    still rank correctly while measuring something else."""
    got = await _toolset(SpyEmbeddings()).rank_texts("q", ["z", "zzzz"], mode="dense")
    assert max(got) > 2.0 / 61, "dense scores should be cosines, not RRF values"

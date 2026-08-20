"""The key must separate what ADR 0002 and ADR 0043 say are different vectors.

A cache keyed on the text alone would serve a nomic vector to a qwen arm, or
an unprefixed vector to a prefixed one. Both produce a fully-populated store
whose cosine similarities are perfectly plausible and quietly wrong -- the
exact failure shape CLAUDE.md records six times over.
"""

from __future__ import annotations

from stark_bench.adapters.memory_embedding_cache import InMemoryEmbeddingCache
from stark_bench.ports.embedding_cache import cache_key


def test_same_inputs_give_the_same_key():
    a = cache_key(model="m", document_prefix="p: ", text="hello")
    b = cache_key(model="m", document_prefix="p: ", text="hello")
    assert a == b


def test_a_different_model_is_a_different_key():
    a = cache_key(model="nomic-embed-text", document_prefix="", text="hello")
    b = cache_key(model="qwen3-embedding-0.6b", document_prefix="", text="hello")
    assert a != b


def test_a_different_prefix_is_a_different_key():
    a = cache_key(model="m", document_prefix="passage: ", text="hello")
    b = cache_key(model="m", document_prefix="search_document: ", text="hello")
    assert a != b


def test_an_empty_prefix_is_a_real_value():
    """qwen's document side is the empty string, not a missing prefix."""
    a = cache_key(model="m", document_prefix="", text="hello")
    b = cache_key(model="m", document_prefix="", text="hello")
    assert a == b


def test_fields_cannot_be_smeared_into_each_other():
    """Concatenation without a separator makes ("ab","c") and ("a","bc") collide.

    A cache that collides two configs serves one arm's vectors to another.
    """
    a = cache_key(model="ab", document_prefix="c", text="x")
    b = cache_key(model="a", document_prefix="bc", text="x")
    assert a != b


async def test_in_memory_cache_round_trips_and_reports_misses():
    cache = InMemoryEmbeddingCache()
    key = cache_key(model="m", document_prefix="", text="hello")
    assert await cache.get_many([key]) == {}
    await cache.put_many({key: [1.0, 2.0]})
    assert await cache.get_many([key]) == {key: [1.0, 2.0]}

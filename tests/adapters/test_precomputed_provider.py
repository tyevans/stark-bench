import pytest

from stark_bench.adapters.precomputed_embeddings import (
    PrecomputedEmbeddingProvider,
    PrecomputedLookupError,
    node_vector_lookup,
)


@pytest.fixture
def provider():
    return PrecomputedEmbeddingProvider(
        {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}, dimension=2
    )


@pytest.mark.asyncio
async def test_it_returns_one_vector_per_input_in_order(provider):
    result = await provider.embed(["beta", "alpha"])
    assert result == [[0.0, 1.0], [1.0, 0.0]]
    assert provider.dimension == 2


@pytest.mark.asyncio
async def test_empty_input_returns_empty_output(provider):
    assert await provider.embed([]) == []


@pytest.mark.asyncio
async def test_a_miss_raises_and_never_falls_back(provider):
    """A silent fallback would turn the control into a second native run
    wearing the control's label, corrupting every comparison downstream."""
    with pytest.raises(KeyError):
        await provider.embed(["a text nobody embedded"])


@pytest.mark.asyncio
async def test_the_miss_error_is_specifically_precomputed_lookup_error(provider):
    with pytest.raises(PrecomputedLookupError):
        await provider.embed(["a text nobody embedded"])


def test_node_vector_lookup_returns_the_vector_for_a_known_id():
    lookup = node_vector_lookup({"1": [0.5, 0.5]})
    assert lookup("1") == [0.5, 0.5]


def test_node_vector_lookup_raises_on_a_miss_and_never_falls_back():
    lookup = node_vector_lookup({"1": [0.5, 0.5]})
    with pytest.raises(PrecomputedLookupError):
        lookup("999")

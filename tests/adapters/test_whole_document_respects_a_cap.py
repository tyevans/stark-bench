"""`whole-document` must split when the provider says the text is too long.

The re-split path in the ingest engine exists so a character cap no longer
has to be guessed right: when the provider rejects a text for length, the
group is re-chunked at half the size and retried. That is worthless against
a chunker that ignores `max_chunk_size` -- the loop re-chunks to the
identical single chunk, fails four times, and raises.

That is not hypothetical. `qwen-rel-whole` died 46 minutes in on
prime-rel's 133,778-character outlier:

    400 exceed_context_size_error
    request (58211 tokens) exceeds the available context size (32768 tokens)

with `qwen-rel-whole.yaml` asserting in a comment that the re-split would
handle exactly that document.

The name still means what it says: with no cap, nothing is split. The cap
is only ever passed by the re-split path, i.e. only after a server has
already refused the text.
"""

from __future__ import annotations

from stark_bench.adapters.chunkers import WholeDocumentChunker


def test_no_cap_means_one_chunk():
    """The default behaviour, which every arm relies on."""
    text = "x" * 10_000
    result = WholeDocumentChunker().chunk(text)
    assert result.total_chunks == 1
    assert result.chunks[0].text == text


def test_a_cap_splits_the_document():
    result = WholeDocumentChunker().chunk("x" * 10_000, max_chunk_size=4_000)
    assert [len(c.text) for c in result.chunks] == [4_000, 4_000, 2_000]


def test_splitting_loses_no_text():
    """Splitting, never truncating -- an arm holding less text measures that."""
    text = "".join(chr(97 + i % 26) for i in range(9_999))
    result = WholeDocumentChunker().chunk(text, max_chunk_size=1_000)
    assert "".join(c.text for c in result.chunks) == text


def test_offsets_are_contiguous_and_indexed():
    """`chunk_id` is built from these, so they must be the chunker's own."""
    result = WholeDocumentChunker().chunk("x" * 2_500, max_chunk_size=1_000)
    assert [(c.chunk_index, c.start_char, c.end_char) for c in result.chunks] == [
        (0, 0, 1_000),
        (1, 1_000, 2_000),
        (2, 2_000, 2_500),
    ]


def test_a_cap_larger_than_the_document_still_gives_one_chunk():
    result = WholeDocumentChunker().chunk("x" * 100, max_chunk_size=4_000)
    assert result.total_chunks == 1


def test_the_halving_sequence_terminates():
    """The engine halves from the longest piece; this must shrink each time.

    A chunker that ignored the cap would return the same longest piece
    forever, which is precisely the defect this file was written for.
    """
    text = "x" * 10_000
    size = 10_000
    seen = []
    for _ in range(4):
        size = max(1, size // 2)
        longest = max(
            len(c.text)
            for c in WholeDocumentChunker().chunk(text, max_chunk_size=size).chunks
        )
        seen.append(longest)
    assert seen == sorted(seen, reverse=True) and seen[-1] < seen[0]

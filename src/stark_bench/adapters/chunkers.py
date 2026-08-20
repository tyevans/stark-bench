"""One chunk per document -- unless the provider refuses the document.

This is the `vss-control` chunker. STaRK's precomputed ada-002 vectors are one
per node document, so the control path must present each document as a single
chunk for those vectors to apply. `chunker_type` is recorded on results, so
the configuration labels itself in the output.

## Why it honours `max_chunk_size` despite the name

The ingest engine's re-split path exists so a character cap no longer has to
be guessed right: when the provider rejects a text for length, the group is
re-chunked at half the size and retried. That is worthless against a chunker
that ignores the cap -- the loop re-chunks to the identical single chunk,
fails `MAX_RESPLIT_ATTEMPTS` times, and raises.

This one ignored it, and `qwen-rel-whole` died 46 minutes into a 2-hour
ingest on `prime-rel`'s 133,778-character outlier:

    400 exceed_context_size_error
    request (58211 tokens) exceeds the available context size (32768)

with the config asserting in a comment that the re-split would handle that
exact document. The safety net was a guaranteed no-op for the one chunker
that most needs it, because the arms that use it are precisely the arms
feeding whole documents to a context-limited server.

The name still holds. `max_chunk_size` defaults to `None` and nothing in
normal operation passes it -- only the re-split does, and only after a server
has already refused the text. So a document is whole until it demonstrably
cannot be.

Splitting, never truncating: an arm holding less text than its neighbours
measures content loss and calls it granularity.
"""

from __future__ import annotations

from redstring.extraction.chunking import Chunk, ChunkingResult


class WholeDocumentChunker:
    """Satisfies redstring's `Chunker` protocol without splitting anything."""

    @property
    def chunker_type(self) -> str:
        return "whole-document"

    def chunk(
        self,
        text: str,
        max_chunk_size: int | None = None,
        overlap_size: int | None = None,
    ) -> ChunkingResult:
        size = max_chunk_size if max_chunk_size and max_chunk_size > 0 else len(text)
        # `or 1` keeps the empty document a single empty chunk rather than an
        # empty chunk list, which would make the node vanish without a count
        # moving anywhere.
        pieces = [text[i : i + size] for i in range(0, len(text) or 1, size or 1)]
        return ChunkingResult(
            chunks=[
                Chunk(
                    text=piece,
                    chunk_index=index,
                    start_char=index * size,
                    end_char=index * size + len(piece),
                )
                for index, piece in enumerate(pieces)
            ],
            total_chunks=len(pieces),
            original_length=len(text),
            chunking_method="whole-document",
            overlap_size=0,
        )

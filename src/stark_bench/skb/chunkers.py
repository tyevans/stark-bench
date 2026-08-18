"""One chunk per document.

This is the `vss-control` chunker. STaRK's precomputed ada-002 vectors are one
per node document, so the control path must present each document as a single
chunk for those vectors to apply. `chunker_type` is recorded on results, so
the configuration labels itself in the output.
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
        return ChunkingResult(
            chunks=[Chunk(text=text, chunk_index=0, start_char=0, end_char=len(text))],
            total_chunks=1,
            original_length=len(text),
            chunking_method="whole-document",
            overlap_size=0,
        )

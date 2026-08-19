"""Embedding access backed by STaRK's precomputed vectors, never live embedding.

The precomputed `.npz` artifacts are keyed by *id* (node id, query id), not by
text. Two different seams need that mapping, in two different shapes:

- **Ingest** writes a vector straight onto `StoredChunk.embedding` by node id
  -- it never calls `embed()`. `node_vector_lookup` builds that callable for
  `stark_bench.skb.ingest.ingest`'s `vector_for` parameter.
- **Retrieval** needs `redstring`'s `EmbeddingProvider.embed(texts)`, which is
  keyed by *text*. `PrecomputedEmbeddingProvider` closes that gap for
  queries: it is built from a `{text: vector}` table resolved once via the
  query id (query text and query id both live in `queries.test-0.1.jsonl`),
  so `ChunkRetriever` can embed a query without embedding anything.

A lookup miss raises in both cases. Falling back to live embedding here would
turn the control into a second native run wearing the control's label, and a
zero vector would corrupt scoring silently while looking fine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class PrecomputedLookupError(KeyError):
    """A text or node id had no precomputed vector."""


class PrecomputedEmbeddingProvider:
    """Satisfies redstring's `EmbeddingProvider` from a fixed `{text: vector}` table."""

    def __init__(
        self, vectors_by_text: Mapping[str, Sequence[float]], *, dimension: int
    ) -> None:
        self._vectors = vectors_by_text
        self._dimension = dimension

    @property
    def model(self) -> str:
        return "text-embedding-ada-002"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for text in texts:
            try:
                vector = self._vectors[text]
            except KeyError as error:
                raise PrecomputedLookupError(
                    f"no precomputed embedding for query text: {text[:80]!r}"
                ) from error
            out.append(list(vector))
        return out

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Identical to `embed`, because ada-002 is a symmetric model.

        Not an oversight and not a stub. `text-embedding-ada-002` has no task
        prefix: OpenAI embeds a query and a document with the same call, and
        STaRK's precomputed `.npz` artifacts were produced that way. The two
        sides of the port coincide *for this model*, which is exactly the case
        ADR 0043 leaves open by defaulting both prefixes to empty.

        It has to be written out even so. `ChunkRetriever` calls `embed_query`,
        and a provider missing it fails with `AttributeError` at the first
        query -- after a full ingest, which for the control is the cheap part
        but for anything live is an hour.
        """
        return await self.embed(texts)


def node_vector_lookup(
    doc_embeddings: Mapping[str, Sequence[float]],
) -> Callable[[str], list[float]]:
    """Build a `vector_for` callable for `skb.ingest.ingest`, raising on a miss."""

    def _lookup(node_id: str) -> list[float]:
        try:
            return list(doc_embeddings[node_id])
        except KeyError as error:
            raise PrecomputedLookupError(
                f"no precomputed embedding for node id: {node_id!r}"
            ) from error

    return _lookup

"""The corpus loader, as something the use case calls rather than imports.

## Why the engine is a port

`skb.ingest.ingest` is a *driven* dependency: the use case decides when to
resume, what to count and what to report, and the engine does the loading.
Importing it directly would tie the use case to STaRK's loader and, worse,
would make testing the resume logic require a database, an embedding server
and 129,000 nodes -- so the resume decision, the part with actual branching
in it, would be the part that never got a unit test.

Declaring it here inverts that. `IngestCorpus` depends on a signature, a
fake satisfies it in a millisecond, and the real loader satisfies it
structurally without importing anything from this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


@runtime_checkable
class IngestCounts(Protocol):
    """What a loader reports having done.

    Structural rather than a shared dataclass, so the loader may keep its
    own richer report type and this layer names only the five numbers it
    puts in the outcome.
    """

    @property
    def nodes(self) -> int: ...
    @property
    def edges(self) -> int: ...
    @property
    def chunks(self) -> int: ...
    @property
    def skipped(self) -> int: ...
    @property
    def self_loops_dropped(self) -> int: ...


class IngestEngine(Protocol):
    """Loads nodes and edges through whatever stores it was handed.

    Deliberately loose about `graph`, `chunks` and `chunker`: those are
    redstring's ports and its chunkers, and restating their types here
    would make this package the arbiter of another library's contracts.
    What this signature pins is the part the use case controls --
    `existing_chunk_ids` and `resume`.
    """

    async def __call__(
        self,
        nodes: object,
        edges: object,
        /,
        *,
        existing_chunk_ids: AbstractSet[str] = ...,
        resume: bool = ...,
        **rest: object,
    ) -> IngestCounts: ...

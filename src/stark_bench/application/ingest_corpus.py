"""Build one arm's corpus, and say what building it produced.

Lifted out of `composition/cli.py`, where it sat inside a function that also
opened two database connections, resolved a data directory, chose an
embedding provider and closed everything in a `finally`. Those are four
different jobs and only one of them is this one.

## What this owns

The resume decision and the outcome. Both were previously expressible only
by running the real thing:

- whether to consult the chunk index at all, and how long that cost;
- which ids reach the loader;
- assembling `IngestOutcome`, whose fields a report reader cannot recover
  if they are wrong.

## What this does not own

Connections, file paths, and which embedding provider a config names. The
caller opens the stores and closes them; this function borrows them for the
duration of one call and never learns what they are connected to.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from stark_bench.domain.ingest import IngestOutcome

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from stark_bench.ports.corpus import ChunkIdIndex
    from stark_bench.ports.ingest import IngestEngine

logger = logging.getLogger(__name__)


async def ingest_corpus(
    *,
    engine: IngestEngine,
    nodes: object,
    edges: object,
    tenant_id: UUID,
    chunk_index: ChunkIdIndex | None,
    edges_ingested: bool,
    config_verbatim: str,
    clock: Callable[[], float] = time.monotonic,
    **engine_kwargs: object,
) -> IngestOutcome:
    """Run one ingest and return what it produced.

    Args:
        engine: The loader. See `ports.ingest` for why this is injected.
        nodes: Node records, passed through untouched.
        edges: Edge records, passed through untouched. Pass an empty
            iterable to load none; `edges_ingested` records the intent
            separately, because "no edges loaded" and "edges loaded, corpus
            has none" are different facts and a reader cannot tell them
            apart from a zero.
        tenant_id: Whose corpus. The only thing scoping the chunk index.
        chunk_index: `None` means do not resume -- the index is not
            consulted and the loader is told to write everything. Resuming
            and having an index are the same decision, so they are one
            argument rather than a bool that can disagree with an object.
        edges_ingested: Recorded verbatim in the outcome.
        config_verbatim: Recorded verbatim in the outcome, so a later run
            can tell whether resuming is safe. A chunk id derives from
            `(source, text)`: a changed chunker writes new ids and leaves
            the old ones behind as live rows that still answer queries, so
            resuming across a chunking change yields a silent *mixture* of
            two chunkings rather than a merely stale corpus.
        clock: Monotonic source, injected so a test can assert a duration
            against a literal rather than against arithmetic on the same
            clock the code used.
        **engine_kwargs: Forwarded to the engine unchanged -- the stores,
            the chunker, the embedding provider and its knobs. This layer
            has no opinion on them and deliberately does not restate their
            types.

    Returns:
        `IngestOutcome`, never a dict. A missing cost block and a block of
        zeroes are the same shape; a type cannot be silently empty.
    """
    started = clock()

    existing_chunk_ids: set[str] = set()
    existing_ids_load_s = 0.0
    resume = chunk_index is not None
    if chunk_index is not None:
        load_started = clock()
        existing_chunk_ids = await chunk_index.ids_for_tenant(tenant_id)
        existing_ids_load_s = clock() - load_started
        logger.info(
            "loaded %d existing chunk ids for tenant in %.2fs (resume)",
            len(existing_chunk_ids),
            existing_ids_load_s,
        )

    counts = await engine(
        nodes,
        edges,
        tenant_id=tenant_id,
        existing_chunk_ids=existing_chunk_ids,
        resume=resume,
        **engine_kwargs,
    )

    return IngestOutcome(
        nodes=counts.nodes,
        chunks=counts.chunks,
        skipped=counts.skipped,
        edges=counts.edges,
        self_loops_dropped=counts.self_loops_dropped,
        edges_ingested=edges_ingested,
        resume=resume,
        existing_ids_load_s=existing_ids_load_s,
        wall_time_s=clock() - started,
        config_verbatim=config_verbatim,
        cache_hits=getattr(counts, "cache_hits", 0),
        cache_misses=getattr(counts, "cache_misses", 0),
    )

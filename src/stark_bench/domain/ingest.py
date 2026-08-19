"""What building a corpus produced, as a value rather than a dict.

It was `dict[str, object]`, and that is how `ingest: {}` reached every
report ever written without anyone noticing: a missing cost block and a
cost block of zeroes are the same shape, and neither one raises. A type
cannot be silently empty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """The result of building one arm's corpus.

    Every field is something a later reader needs and cannot recover:
    `skipped` distinguishes a resumed ingest from a slow one, and
    `self_loops_dropped` distinguishes a clean corpus from a loader that
    stopped looking -- PRIME has self-loops, so a zero there is more likely
    a bug than a property of the data.
    """

    nodes: int
    chunks: int
    skipped: int
    edges: int
    self_loops_dropped: int
    edges_ingested: bool
    resume: bool
    existing_ids_load_s: float
    wall_time_s: float
    #: The config that produced this corpus, verbatim. Resuming is safe only
    #: when the chunking has not changed: a chunk id derives from
    #: `(source, text)`, so a changed chunker writes NEW ids and leaves the
    #: old ones behind as live rows that still answer queries. The result is
    #: not a stale corpus but a silent mixture of two chunkings.
    config_verbatim: str = ""

    @property
    def chunks_per_node(self) -> float:
        """The granularity this arm actually achieved.

        The number the chunking sweep is *about*, so it is computed here
        rather than by each reader -- three call sites dividing by hand is
        three chances to divide by zero on an empty corpus.
        """
        return self.chunks / self.nodes if self.nodes else 0.0

    def as_dict(self) -> dict[str, object]:
        """The report's `ingest` block."""
        return asdict(self)

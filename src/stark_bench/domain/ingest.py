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
    stopped looking.

    This docstring used to say a zero there was "more likely a bug than a
    property of the data", because PRIME has self-loops. Measured on
    2026-08-19, `prime/test-0.1` has **none**: 8,100,498 edges, zero where
    `source == target`, and the full ingest loaded all 8,100,498. So a zero
    is the honest answer for this split, and treating it as suspicious sends
    the next reader hunting a defect that is not there.

    What makes the zero trustworthy is not this file: the drop path has its
    own test (`test_a_self_loop_is_dropped_and_counted`), so the counter is
    known to work on input that exercises it. A field whose only evidence is
    a run whose input could not exercise it is unverified no matter what
    number it shows.
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
    def corpus_chunks(self) -> int:
        """Chunks the corpus holds, not chunks this run wrote.

        `chunks` counts writes and `skipped` counts ids already present, so
        their sum is what the tenant actually holds. On a fresh ingest
        `skipped` is zero and the two are the same number, which is exactly
        why the distinction went unnoticed.
        """
        return self.chunks + self.skipped

    @property
    def chunks_per_node(self) -> float:
        """The granularity this arm actually achieved.

        The number the chunking sweep is *about*, so it is computed here
        rather than by each reader -- three call sites dividing by hand is
        three chances to divide by zero on an empty corpus.

        Computed from `corpus_chunks`, not `chunks`. It used to be
        `chunks / nodes`, which is right for a fresh ingest and wrong for
        every resumed one: arm 1 finished on 2026-08-19 having *written*
        49,280 chunks and skipped 87,523, and reported **0.381** for a
        corpus whose real granularity is 136,803 / 129,375 = **1.058**.

        A value below 1.0 is impossible for any chunker here -- every node
        yields at least one chunk -- so the number was not merely wrong but
        outside the range the metric can take, and it still rendered into
        RESULTS.md without comment.
        """
        return self.corpus_chunks / self.nodes if self.nodes else 0.0

    def as_dict(self) -> dict[str, object]:
        """The report's `ingest` block."""
        return asdict(self)

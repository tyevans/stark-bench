"""`chunks_per_node` must describe the corpus, not one run's writes.

The distinction is invisible on a fresh ingest -- `skipped` is zero, so
`chunks` and `chunks + skipped` are the same number -- which is why the
defect survived until an arm was resumed.
"""

from __future__ import annotations

from stark_bench.domain.ingest import IngestOutcome


def _outcome(**overrides) -> IngestOutcome:
    base = {
        "nodes": 129375,
        "chunks": 136803,
        "skipped": 0,
        "edges": 8100498,
        "self_loops_dropped": 0,
        "edges_ingested": True,
        "resume": False,
        "existing_ids_load_s": 0.0,
        "wall_time_s": 1.0,
    }
    base.update(overrides)
    return IngestOutcome(**base)


def test_a_fresh_ingest_reports_what_it_wrote() -> None:
    assert _outcome().chunks_per_node == 136803 / 129375


def test_a_resumed_ingest_reports_the_corpus_not_the_delta() -> None:
    """Arm 1, 2026-08-19: wrote 49,280, skipped 87,523.

    The old `chunks / nodes` gave 0.381 for a corpus whose granularity is
    1.058. The two inputs are deliberately unequal and neither is a round
    number, so an implementation returning either one alone is visible.
    """
    resumed = _outcome(chunks=49280, skipped=87523, resume=True)

    assert resumed.corpus_chunks == 136803
    assert resumed.chunks_per_node == 136803 / 129375
    # The specific wrong answer this replaced, named so a regression is
    # recognisable rather than merely failing.
    assert abs(resumed.chunks_per_node - 0.381) > 0.5


def test_granularity_is_never_below_one() -> None:
    """Every node yields at least one chunk, for every chunker here.

    So a value under 1.0 is not an unusual corpus, it is broken arithmetic
    -- which is what makes it worth asserting as an invariant rather than
    checking a particular number.
    """
    for chunks, skipped in ((136803, 0), (49280, 87523), (129375, 0)):
        assert _outcome(chunks=chunks, skipped=skipped).chunks_per_node >= 1.0


def test_an_empty_corpus_does_not_divide_by_zero() -> None:
    assert _outcome(nodes=0, chunks=0, skipped=0).chunks_per_node == 0.0

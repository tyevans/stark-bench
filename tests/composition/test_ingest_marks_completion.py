"""The ingest report is written twice, and only the second says `complete`.

B-RESUME-COMPLETE-1. Writing it twice is the whole mechanism: a single write
at the end cannot distinguish "did not finish" from "never started", because
a killed run leaves either no report (resume refused, correct) or the report
of some EARLIER run -- which then vouches for a corpus that run did not
produce.

Asserted on the syntax tree because running `_do_ingest` needs Postgres,
Neo4j and an embedding endpoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import stark_bench.composition.cli as cli_mod
from stark_bench.domain.ingest import IngestOutcome

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")


def _outcome(**over) -> IngestOutcome:
    base = dict(
        nodes=1,
        chunks=1,
        skipped=0,
        edges=0,
        self_loops_dropped=0,
        edges_ingested=False,
        resume=True,
        existing_ids_load_s=0.0,
        wall_time_s=1.0,
    )
    base.update(over)
    return IngestOutcome(**base)


def test_an_outcome_is_incomplete_by_default() -> None:
    """The safe default. A field defaulting to True would make every
    in-flight outcome claim to have finished."""
    assert _outcome().complete is False


def test_completion_reaches_the_report_dict() -> None:
    assert _outcome(complete=True).as_dict()["complete"] is True


def test_the_stub_is_written_before_the_ingest_runs() -> None:
    """Order is the mechanism, not a detail: a stub written afterwards
    records nothing a crash could be caught by."""
    body = _SRC.split("if args.ingest:")[1].split("if args.run:")[0]
    stub = body.index('"complete": False')
    run = body.index("asyncio.run")
    assert stub < run, "the incomplete stub must be written BEFORE the ingest"


def test_the_stub_carries_the_config_so_the_guard_can_read_it() -> None:
    """`resume_is_safe` checks the config first. A stub without it would
    refuse for the wrong reason and hide which check fired."""
    body = _SRC.split("if args.ingest:")[1].split("asyncio.run")[0]
    assert "config_verbatim" in body


def test_the_finished_report_is_marked_complete() -> None:
    assert "replace(outcome, complete=True)" in _SRC


def test_the_ingest_itself_does_not_mark_completion() -> None:
    """`ingest_corpus` cannot know the caller survived to write the file, so
    completion is stamped at the call site that does."""
    engine = (
        Path(cli_mod.__file__).parent.parent / "adapters" / "stark_ingest_engine.py"
    ).read_text(encoding="utf-8")
    assert "complete=True" not in engine


def test_replace_does_not_disturb_the_other_fields() -> None:
    original = _outcome(chunks=42, cache_hits=7)
    stamped = replace(original, complete=True)
    assert stamped.chunks == 42 and stamped.cache_hits == 7

"""Is it safe to resume an arm's ingest, or must it start clean?

A chunk id is derived from `(source, text)`, so a **changed chunker writes
new ids and leaves the old ones behind** -- and the old chunks are still
real rows in the tenant, still returned by search. Resuming across a
chunking change therefore does not produce a stale corpus, it produces a
corpus that is a silent mixture of two chunkings, which is worse: every
count is inflated, and the arm no longer measures the granularity its config
names.

So resume is allowed only when the config that produced the existing corpus
is byte-identical to the config on disk. Anything else -- no report, no
recorded config, unreadable JSON, any difference at all -- refuses. A report
written before `config_verbatim` was recorded has no claim to make about
what produced it and cannot vouch for the corpus.

Exit 0 to allow resume, 1 to refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resume_is_safe(config_name: str, root: Path = ROOT) -> bool:
    report = root / "results" / f"{config_name}.ingest.json"
    source = root / "config" / f"{config_name}.yaml"
    if not report.exists() or not source.exists():
        return False
    try:
        recorded = json.loads(report.read_text(encoding="utf-8")).get("config_verbatim")
    except (OSError, ValueError):
        return False
    if recorded is None:
        return False
    return recorded == source.read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(0 if resume_is_safe(sys.argv[1]) else 1)

"""Stamp reports written before source provenance was recorded.

## Why an explicit stamp rather than inferring it from absence

Absence of `redstring_commit` does mean "written before this existed", and
a renderer could say so without touching a byte. That was the first plan
and it is the weaker one, for the reason this whole campaign keeps
relearning: **inferred metadata reads exactly like recorded metadata until
the inference is wrong.** A report copied in from elsewhere, or written by
a branch that skipped the call, would render as pre-release with the same
confidence as a report that genuinely is.

So the fact is written down once, by a script that says what it did.

## What the stamp claims, and what it does not

`redstring_release` is set to the string below on every report lacking
`redstring_commit`. It asserts one thing: this number was measured before
`8de0cb2`, the merge of redstring PR #72. It does NOT claim which commit
produced it -- that is unrecoverable, and inventing a hash would be worse
than the gap it filled.

The distinction that matters for these files is PR #72's change to
`SlidingWindowChunker`, because four configs name `sliding-1000-500` and
two of their tenants are live. See B-SLIDING-CORPORA-PREDATE-THE-FIX-1.

Idempotent: a second run stamps nothing, because the first run's reports
now carry the field. Prints what it changed rather than reporting success,
since "0 files updated" and "0 files found" are the same exit status --
the failure mode this project has hit nine times.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The merge commit of redstring PR #72 on `main`. Everything stamped here
#: was measured before it, on a `SlidingWindowChunker` that emitted a
#: redundant tail chunk for every document longer than the window.
_PRE_RELEASE = "pre-8de0cb2"

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    if not _RESULTS.is_dir():
        print(f"no results directory at {_RESULTS}", file=sys.stderr)  # noqa: T201
        return 1

    stamped: list[str] = []
    already: list[str] = []
    for path in sorted(_RESULTS.glob("*.json")):
        # Predictions are raw rankings with no metadata block to stamp.
        if path.name.endswith(".predictions.json"):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"SKIP unreadable {path.name}", file=sys.stderr)  # noqa: T201
            continue
        if not isinstance(report, dict):
            continue

        # Run reports carry provenance under `cost`; ingest reports carry it
        # at the top level, because an ingest has no cost block.
        block = report.get("cost") if "cost" in report else report
        if not isinstance(block, dict):
            continue
        if block.get("redstring_commit") is not None:
            already.append(path.name)
            continue
        if block.get("redstring_release") == _PRE_RELEASE:
            already.append(path.name)
            continue

        block["redstring_release"] = _PRE_RELEASE
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        stamped.append(path.name)

    print(f"stamped {len(stamped)} report(s) as {_PRE_RELEASE}")  # noqa: T201
    for name in stamped:
        print(f"  + {name}")  # noqa: T201
    print(f"left {len(already)} report(s) alone (already have provenance)")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

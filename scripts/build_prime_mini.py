"""Build `data/prime-mini`, the 10,000-node smoke-test subset of PRIME.

PRIME is already the smallest corpus STaRK ships -- 129,375 nodes against
MAG's 1,872,968 -- so a faster cell has to be constructed rather than chosen.

The construction is the whole point, so it is stated rather than left in a
shell history: **every node that is a gold answer to one of the 280
`test-0.1` queries is kept**, and the rest of the 10,000 is sampled from the
candidate list at a fixed seed. Dropping a gold node would silently cap
recall below 1.0 for reasons that have nothing to do with retrieval, and the
assertion below is what makes that a crash rather than a plausible number.

`candidates.json` is rewritten to exactly the kept set, because `Evaluator`
scores over `candidate_ids` and a candidate that was never ingested is a
slot no arm can win. `edges.jsonl` keeps only edges with BOTH endpoints
inside the subset -- an edge to a missing entity is what
`upsert_relationships` raises `MissingEntityError` on. The query files are
symlinks, so there is one copy and no chance of the subset and the parent
disagreeing about what the answers are.

A 10,000-candidate pool is a 13x easier retrieval problem than the full
129,375, so **every metric measured here is inflated** and must never share a
table with a full-PRIME number. What it buys is a ~6-minute ingest per arm
instead of ~90, which is the difference between finding a silent defect
before committing a day of endpoint time and finding it after.

Rerunning this is idempotent at a fixed seed and overwrites in place.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path("data/prime")
DST = Path("data/prime-mini")
SPLIT = "test-0.1"

#: Total nodes in the subset. Large enough that retrieval is not trivial,
#: small enough that an arm ingests inside a coffee break.
TARGET = 10_000

#: Fixed so the subset is the same corpus every time it is rebuilt. Two arms
#: measured against two different distractor samples are not comparable, and
#: the resampling would be invisible.
SEED = 20260819


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    gold: set[str] = set()
    for line in (SRC / f"queries.{SPLIT}.jsonl").open():
        gold.update(str(a) for a in json.loads(line)["answer_ids"])

    candidates = [str(c) for c in json.load((SRC / "candidates.json").open())]
    pool = [c for c in candidates if c not in gold]
    keep = set(gold) | set(random.Random(SEED).sample(pool, TARGET - len(gold)))
    print(f"keeping {len(keep):,} nodes, of which {len(gold):,} are gold answers")

    written = 0
    with (DST / "nodes.jsonl").open("w") as out:
        for line in (SRC / "nodes.jsonl").open():
            if str(json.loads(line)["node_id"]) in keep:
                out.write(line)
                written += 1

    # Not a sanity check on the writing -- a check that `node_id` is still the
    # field the ids live in. If the export ever renames it this loop matches
    # nothing, writes an empty corpus, and every downstream stage "succeeds"
    # against it. That is exactly the shape of every silent defect this
    # project has had.
    assert written == len(keep), f"wrote {written} nodes for {len(keep)} ids"
    print(f"nodes: {written:,}")

    edges = 0
    with (DST / "edges.jsonl").open("w") as out:
        for line in (SRC / "edges.jsonl").open():
            edge = json.loads(line)
            if str(edge["source"]) in keep and str(edge["target"]) in keep:
                out.write(line)
                edges += 1
    print(f"edges: {edges:,}")

    json.dump(sorted(int(c) for c in keep), (DST / "candidates.json").open("w"))

    for name in (f"queries.{SPLIT}.jsonl", "queries.test.jsonl"):
        link = DST / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f"../prime/{name}")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()

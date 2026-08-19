"""Render `results/*.json` as the RESULTS.md table.

Reads only what the reports contain. Anything a report does not carry shows
as `--` rather than as a zero: a missing cost column and a cost of zero are
different claims, and this project has already shipped an `ingest: {}` that
read as the latter.

    uv run python scripts/results_table.py            # markdown to stdout
    uv run python scripts/results_table.py --check    # non-zero if a cell is suspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

EXPECTED_NODES = 129375
"""Nodes in prime/test-0.1. A report below this is not describing this corpus."""

METRICS = ("mrr", "hit@1", "hit@5", "recall@20")
AGENTS = ("dense", "lexical", "hybrid", "zero_shot", "deep", "rerank")


def _fmt(value: object, places: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def load() -> list[dict]:
    rows = []
    # Not rglob: `results/archive/` holds reports measured against
    # configurations that no longer exist, kept deliberately. See its README.
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.endswith(".ingest.json"):
            continue
        stem = path.stem  # "<config>.<agent>"
        config, _, agent = stem.rpartition(".")
        if not config:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"config": config, "agent": agent, "data": data, "path": path})
    rows.sort(
        key=lambda r: (
            r["config"],
            AGENTS.index(r["agent"]) if r["agent"] in AGENTS else 99,
        )
    )
    return rows


def render(rows: list[dict]) -> str:
    # Derived from one list, because the header and its separator are two
    # things that must agree and were written twice: the header had eleven
    # columns and the separator emitted `5 + len(METRICS)` = nine, so every
    # rendered table was malformed.
    columns = [
        "config",
        "agent",
        *METRICS,
        "tool/q",
        "llm/q",
        "run s",
        "chunks/node",
        "ingest s",
    ]
    out = ["| " + " | ".join(columns) + " |"]
    out.append("|" + "---|" * len(columns))
    for row in rows:
        d = row["data"]
        metrics = d.get("metrics") or {}
        cost = d.get("cost") or {}
        ingest = d.get("ingest") or {}
        nodes, chunks = ingest.get("nodes"), ingest.get("chunks")
        # `chunks` counts writes; `skipped` counts ids already present. The
        # corpus holds both, so a resumed ingest reporting `chunks / nodes`
        # understates granularity by whatever it skipped -- 0.381 for arm 1,
        # against a real 1.058.
        corpus_chunks = (chunks or 0) + (ingest.get("skipped") or 0)
        per_node = f"{corpus_chunks / nodes:.3f}" if nodes and corpus_chunks else "--"
        out.append(
            f"| {row['config']} | {row['agent']} | "
            + " | ".join(_fmt(metrics.get(m)) for m in METRICS)
            + f" | {_fmt(cost.get('tool_calls_per_query'), 2)}"
            + f" | {_fmt(cost.get('llm_calls_per_query'), 2)}"
            + f" | {_fmt(cost.get('seconds_total'), 1)}"
            + f" | {per_node}"
            + f" | {_fmt(ingest.get('wall_time_s'), 1)} |"
        )
    return "\n".join(out)


def check(rows: list[dict]) -> list[str]:
    """Cells that are suspicious rather than merely absent.

    Every one of these has been a real defect here, which is why they are
    checked rather than eyeballed -- a zero and a missing value look alike
    in a rendered table and mean opposite things.
    """
    problems = []
    config_root = Path(__file__).resolve().parent.parent / "config"
    for row in rows:
        where = f"{row['config']}/{row['agent']}"
        d = row["data"]
        metrics = d.get("metrics") or {}
        if not metrics:
            problems.append(f"{where}: no metrics at all")
            continue
        if all(v == 0 for v in metrics.values()):
            problems.append(
                f"{where}: every metric is zero -- retrieval returned nothing"
            )
        ingest = d.get("ingest") or {}
        if not ingest:
            problems.append(f"{where}: empty ingest block -- cost column is unbacked")
        elif ingest.get("nodes") and ingest["nodes"] < EXPECTED_NODES:
            # A report from a *different, smaller* run -- a --limit probe, or
            # an ingest that died. The config check in resume_is_safe.py
            # cannot see this: a 3000-node probe uses the same config file as
            # the full run, so the recorded text matches byte for byte.
            #
            # Observed 2026-08-19: this table published chunks/node 1.000 and
            # ingest 207.7s for native-wholedoc from a 3000-node probe, while
            # the real corpus held 136,803 chunks over 129,375 nodes.
            problems.append(
                f"{where}: ingest report describes {ingest['nodes']} nodes, "
                f"expected {EXPECTED_NODES} -- stale report from a smaller run, "
                f"so chunks/node and ingest s describe a different corpus"
            )
        corpus_chunks = (ingest.get("chunks") or 0) + (ingest.get("skipped") or 0)
        nodes = ingest.get("nodes")
        if nodes and corpus_chunks and corpus_chunks < nodes:
            # Impossible for any chunker here: every node yields at least one
            # chunk, so chunks/node is >= 1.0 by construction. A value below
            # it means the arithmetic is wrong, not that the corpus is
            # unusual -- which is how 0.381 rendered without comment.
            problems.append(
                f"{where}: {corpus_chunks} chunks for {nodes} nodes "
                f"({corpus_chunks / nodes:.3f} per node) -- below 1.0 is "
                f"impossible; every node yields at least one chunk"
            )
        if d.get("queries") in (0, None):
            problems.append(f"{where}: scored {d.get('queries')!r} queries")
        if ingest and not ingest.get("edges_ingested") and row["agent"] == "deep":
            problems.append(
                f"{where}: deep agent on an edgeless corpus (B-DEEP-EDGES-1)"
            )

        # A report outlives the config that produced it, and the file name
        # says nothing about which era it belongs to. Two stale
        # `redstring-native` reports from the pre-prefix nomic runs were
        # sitting in `results/` looking exactly like current ones -- same
        # name, same shape, plausible numbers, a different embedding model
        # entirely. `config_verbatim` is the whole config as it was, which
        # is what makes this checkable at all.
        source = config_root / f"{row['config']}.yaml"
        if source.exists():
            current = source.read_text(encoding="utf-8")
            if d.get("config_verbatim") != current:
                problems.append(
                    f"{where}: STALE -- written against a different "
                    f"{source.name} than the one on disk; re-run it"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = load()
    if not rows:
        print("no reports in results/", file=sys.stderr)
        return 1
    print(render(rows))
    problems = check(rows)
    if problems:
        print("\nSuspect cells:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    return 1 if (args.check and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())

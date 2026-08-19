"""Score a persisted predictions file, without re-running retrieval.

Retrieval costs ~50 minutes of shared GPU for the `deep` agent; scoring is a
subprocess that resolves `stark-qa` from PyPI every time it runs. When PyPI
returned 502 mid-run, `redstring-native/deep` lost the whole run at the last
step. `cli.py` now writes predictions before scoring, and this turns one of
those files into a report.

Cost cannot be recovered -- it lived in the tool-call log of the dead process
-- so a rescored report carries the cost block the original run wrote, if the
report exists, and an empty one otherwise. That difference is visible in the
file rather than papered over.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from stark_bench.adapters.config_file import load_config
from stark_bench.adapters.report_file import write_report
from stark_bench.adapters.stark_artifacts import read_queries
from stark_bench.adapters.stark_scorer import score_predictions
from stark_bench.composition.cli import (
    DATA_ROOT,
    _ingest_stats,
    predictions_path,
    report_path,
)
from stark_bench.domain import Ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()

    config = replace(load_config(args.config), agent=args.agent)

    raw = json.loads(predictions_path(config).read_text())
    predictions = {
        int(qid): [Ranked(node_id, score) for node_id, score in ranked.items()]
        for qid, ranked in raw.items()
    }

    data_dir = DATA_ROOT / config.dataset
    pairs = list(read_queries(data_dir / f"queries.{config.split}.jsonl"))
    answers = {q.query_id: a for q, a in pairs}
    candidate_ids = [
        int(c) for c in json.loads((data_dir / "candidates.json").read_text())
    ]

    metrics = score_predictions(predictions, answers, candidate_ids=candidate_ids)
    print(metrics)

    path = report_path(config)
    prior = json.loads(path.read_text()) if path.exists() else {}
    write_report(
        path,
        config=config,
        metrics=metrics,
        cost=prior.get("cost", {}),
        ingest=_ingest_stats(config),
        queries=len(pairs),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

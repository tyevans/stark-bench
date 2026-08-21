"""Hand predictions to STaRK's evaluator, in its own interpreter.

`stark-qa` pulls colbert-ai, gritlm, llm2vec, PyTDC, ogb, torch_geometric and
more -- all serving baselines we do not run, several of which will not resolve
on 3.13. So it lives in a 3.11 environment reached by subprocess, and the
harness stays small.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from stark_bench.domain import Ranked

DEFAULT_METRICS = ("mrr", "hit@1", "hit@5", "recall@20")

#: Invoked by path: `--no-project` means `stark_bench` is not importable there.
SIDECAR = Path(__file__).resolve().parent.parent / "sidecar" / "score.py"


#: A prebuilt Python 3.11 environment for the sidecar, if one exists.
#: Created by `scripts/build_sidecar_env.sh`; not committed, and not
#: required.
SIDECAR_VENV = Path(__file__).resolve().parents[3] / ".sidecar-venv"


def _sidecar_command() -> list[str]:
    """How to launch the 3.11 sidecar: prebuilt if present, resolved if not.

    `uv run --no-project --with stark-qa --with "numpy<2"` re-resolves 166
    packages on every scoring run. Warm, that is ~114ms and invisible. Cold,
    or with PyPI degraded, it is a hard failure AFTER all retrieval has been
    paid for -- on 2026-08-19 a `deep` arm finished 280 queries in 46
    minutes of shared GPU and then died on a 502 fetching `anthropic`, a
    transitive dependency of `stark-qa` that this sidecar never imports.

    So a prebuilt environment is used when it exists, and the run touches no
    network at all.

    **Falling back rather than requiring it is deliberate.** Making the
    prebuilt path mandatory -- or adding `--offline` -- trades a rare
    failure for a certain one: the first run on any machine, and every CI
    checkout, would fail instead. The fallback is the same command that has
    always worked.

    The version is pinned by whatever the venv was built with, which is a
    real difference from `--with stark-qa`: the resolved path floats and the
    prebuilt one does not. That is a feature for reproducibility and a trap
    for staleness, which is why the builder script is one command to re-run.
    """
    python = SIDECAR_VENV / "bin" / "python"
    if python.exists():
        return [str(python)]
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.11",
        "--with",
        "stark-qa",
        "--with",
        "numpy<2",
        "python",
    ]


def score_predictions(
    predictions: Mapping[int, Sequence[Ranked]],
    answers: Mapping[int, Sequence[str]],
    *,
    candidate_ids: Sequence[int],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Run the official evaluator over `predictions`. Raises on any failure.

    A query that retrieved nothing is rejected here rather than passed on.
    STaRK's evaluator computes `min(pred) - 1` as the floor score for every
    unranked candidate, so one empty prediction ends the run with
    `ValueError: min() arg is an empty sequence` from inside a subprocess --
    after all the retrieval has been paid for, and naming neither the query
    nor the cause.
    """
    empty = sorted(qid for qid, ranked in predictions.items() if not ranked)
    if empty:
        shown = ", ".join(str(qid) for qid in empty[:10])
        if len(empty) > 10:
            shown += ", ..."
        cause = (
            "every query retrieved nothing, which points at the corpus rather "
            "than at the agent: check that the config's chunk table holds rows "
            "for this config's tenant"
            if len(empty) == len(predictions)
            else "the agent returned no candidates for these queries"
        )
        raise ValueError(
            f"{len(empty)} of {len(predictions)} queries have no predictions "
            f"({shown}); {cause}. STaRK's evaluator cannot score an empty "
            f"prediction list."
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        preds_path, answers_path, candidates_path, out_path = (
            root / "preds.json",
            root / "answers.json",
            root / "candidates.json",
            root / "metrics.json",
        )
        preds_path.write_text(
            json.dumps(
                {
                    str(qid): {r.node_id: r.score for r in ranked}
                    for qid, ranked in predictions.items()
                }
            )
        )
        answers_path.write_text(
            json.dumps({str(qid): list(a) for qid, a in answers.items()})
        )
        candidates_path.write_text(json.dumps(list(candidate_ids)))

        completed = subprocess.run(  # noqa: S603
            [
                *_sidecar_command(),
                str(SIDECAR),
                "--predictions",
                str(preds_path),
                "--answers",
                str(answers_path),
                "--candidates",
                str(candidates_path),
                "--out",
                str(out_path),
                "--metrics",
                ",".join(metrics),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stark-qa scoring failed:\n{completed.stdout}\n{completed.stderr}"
            )
        return json.loads(out_path.read_text())

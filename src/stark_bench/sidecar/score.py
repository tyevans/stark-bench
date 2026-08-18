"""Official STaRK scoring, run under 3.11 with `stark-qa` installed.

We compute no metric ourselves. An expected value produced by the code under
test measures determinism rather than correctness, and a reimplemented MRR is
exactly that.

Invoked as a subprocess:
    uv run --no-project --python 3.11 --with stark-qa --with "numpy<2" python -m stark_bench.sidecar.score \
        --predictions preds.json --answers answers.json --out metrics.json
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--metrics", default="mrr,hit@1,hit@5,recall@20")
    args = parser.parse_args()

    import torch
    from stark_qa.evaluator import Evaluator

    with open(args.predictions) as handle:
        predictions = json.load(handle)
    with open(args.answers) as handle:
        answers = json.load(handle)

    metrics = args.metrics.split(",")
    with open(args.candidates) as handle:
        candidate_ids = [int(c) for c in json.load(handle)]
    evaluator = Evaluator(candidate_ids)

    totals: dict[str, list[float]] = {m: [] for m in metrics}
    for query_id, pred in predictions.items():
        pred_dict = {int(node_id): float(score) for node_id, score in pred.items()}
        answer_ids = torch.LongTensor([int(a) for a in answers[query_id]])
        result = evaluator.evaluate(pred_dict, answer_ids, metrics=metrics)
        for name, value in result.items():
            totals[name].append(float(value))

    averaged = {name: (sum(v) / len(v) if v else 0.0) for name, v in totals.items()}
    with open(args.out, "w") as handle:
        json.dump(averaged, handle, indent=2)


if __name__ == "__main__":
    main()

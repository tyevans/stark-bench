"""STaRK's SKB to neutral artifacts, under 3.11 with `stark-qa` installed.

Run:
    uv run --no-project --python 3.11 --with stark-qa --with "numpy<2" \
        python src/stark_bench/sidecar/export.py --dataset prime --out data/prime

Run by path, not `-m` — the sidecar interpreter has no `stark_bench` installed
and must import nothing from this package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="test-0.1,test")
    args = parser.parse_args()

    from stark_qa import load_qa, load_skb

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    skb = load_skb(args.dataset, download_processed=True)

    with (out / "nodes.jsonl").open("w", encoding="utf-8") as handle:
        for node_id in range(skb.num_nodes()):
            handle.write(
                json.dumps(
                    {
                        "node_id": str(node_id),
                        "node_type": str(skb.get_node_type_by_id(node_id)),
                        "name": str(getattr(skb[node_id], "name", node_id)),
                        "document": skb.get_doc_info(node_id, add_rel=False),
                    }
                )
                + "\n"
            )

    edge_index = skb.edge_index
    edge_types = skb.edge_types
    with (out / "edges.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(edge_index.shape[1]):
            handle.write(
                json.dumps(
                    {
                        "source": str(int(edge_index[0, i])),
                        "target": str(int(edge_index[1, i])),
                        "relation": str(skb.edge_type_dict[int(edge_types[i])]),
                    }
                )
                + "\n"
            )

    qa = load_qa(args.dataset)
    splits = qa.get_idx_split()
    for split in args.splits.split(","):
        indices = splits[split]
        with (out / f"queries.{split}.jsonl").open("w", encoding="utf-8") as handle:
            for idx in indices:
                query, query_id, answer_ids, _ = qa[int(idx)]
                handle.write(
                    json.dumps(
                        {
                            "query_id": int(query_id),
                            "text": str(query),
                            "answer_ids": [str(int(a)) for a in answer_ids],
                        }
                    )
                    + "\n"
                )

    # The answerable subset, not every node. `Evaluator` indexes by
    # `max(candidate_ids)`, and scoring against all 129k nodes would silently
    # change every metric.
    (out / "candidates.json").write_text(
        json.dumps([int(c) for c in skb.candidate_ids])
    )

    print(f"exported {args.dataset} to {out}")


if __name__ == "__main__":
    main()

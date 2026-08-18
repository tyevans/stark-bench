"""STaRK's precomputed ada-002 embeddings, under 3.11 with `torch` and `gdown`.

Run:
    uv run --no-project --python 3.11 --with stark-qa --with "numpy<2" --with gdown \
        python src/stark_bench/sidecar/embeddings.py --dataset prime --out data/prime

Run by path, not `-m` — the sidecar interpreter has no `stark_bench` installed
and must import nothing from this package.

STaRK ships `candidate_emb_dict.pt` (node_id -> tensor) and
`query_emb_dict.pt` (query_id -> tensor) on Google Drive, keyed by id rather
than by text. This script downloads both, loads them with `torch.load`, and
writes two `.npz` files (`ids`, `vectors` arrays) so the 3.13 harness can read
them back with numpy alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# From STaRK's own emb_download.py, for the `prime` dataset.
DRIVE_IDS = {
    "prime": {
        "query": "1MshwJttPZsHEM2cKA5T13SIrsLeBEdyU",
        "candidate": "16EJvCMbgkVrQ0BuIBvLBp-BYPaye-Edy",
    },
}


def _download(file_id: str, dest: Path) -> None:
    import gdown

    gdown.download(id=file_id, output=str(dest), quiet=False)


def _convert(pt_path: Path, npz_path: Path) -> tuple[int, int]:
    """Load a `{id: tensor}` dict and write it as an `(ids, vectors)` npz.

    Returns (count, dimension) for reporting.
    """
    import numpy as np
    import torch

    emb_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    ids = []
    vectors = []
    for key, tensor in emb_dict.items():
        ids.append(int(key))
        vectors.append(tensor.detach().cpu().numpy().reshape(-1))

    ids_arr = np.array(ids, dtype=np.int64)
    vectors_arr = np.stack(vectors).astype(np.float32)

    np.savez(npz_path, ids=ids_arr, vectors=vectors_arr)

    dim = int(vectors_arr.shape[1]) if vectors_arr.ndim == 2 else 0
    return len(ids_arr), dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.dataset not in DRIVE_IDS:
        raise SystemExit(f"no known Google Drive ids for dataset {args.dataset!r}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = DRIVE_IDS[args.dataset]

    query_pt = out / "query_emb_dict.pt"
    candidate_pt = out / "candidate_emb_dict.pt"

    _download(ids["query"], query_pt)
    _download(ids["candidate"], candidate_pt)

    for pt_path, label in ((query_pt, "query"), (candidate_pt, "candidate")):
        if not pt_path.exists() or pt_path.stat().st_size < 1024:
            raise SystemExit(
                f"{label} download at {pt_path} looks wrong (missing or tiny); "
                "gdown may have returned an HTML confirmation page instead of "
                "the tensor file. Refusing to fabricate embeddings."
            )

    query_count, query_dim = _convert(query_pt, out / "query_emb.npz")
    doc_count, doc_dim = _convert(candidate_pt, out / "doc_emb.npz")

    query_pt.unlink()
    candidate_pt.unlink()

    print(
        f"query embeddings: {query_count} vectors, dim {query_dim} "
        f"-> {out / 'query_emb.npz'}"
    )
    print(
        f"doc embeddings: {doc_count} vectors, dim {doc_dim} "
        f"-> {out / 'doc_emb.npz'}"
    )

    if query_dim != 1536 or doc_dim != 1536:
        raise SystemExit(
            f"expected ada-002's 1536 dimensions, got query={query_dim} "
            f"doc={doc_dim}. This is a different model; the vss-control "
            "would not be a control. Stopping without further processing."
        )


if __name__ == "__main__":
    main()

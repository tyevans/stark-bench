"""The neutral format between the sidecar and the harness.

JSON Lines, one record per line. The sidecar writes these under a 3.11
interpreter with `stark-qa` installed; everything else in this project reads
them under 3.13 with a small dependency set.

`read_queries` returns `(Query, answers)` pairs rather than a query object
carrying its answers, because `Query` is what reaches an agent and must not
carry ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from stark_bench.domain import Query

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkbNode:
    node_id: str
    node_type: str
    name: str
    document: str


@dataclass(frozen=True, slots=True)
class SkbEdge:
    source: str
    target: str
    relation: str


def read_nodes(path: Path) -> Iterator[SkbNode]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield SkbNode(**json.loads(line))


def read_edges(path: Path) -> Iterator[SkbEdge]:
    """Yields every edge, self-loops included.

    Filtering belongs to the loader, which counts what it drops.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield SkbEdge(**json.loads(line))


def read_queries(path: Path) -> Iterator[tuple[Query, list[str]]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            answers = [str(a) for a in record["answer_ids"]]
            yield Query(query_id=int(record["query_id"]), text=record["text"]), answers


def _read_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Reads an `(ids, vectors)` npz into an `{id: vector}` dict.

    Ids are string-keyed on the way out, matching every other artifact here
    (`SkbNode.node_id`, `Query.query_id` aside), so a caller never has to
    juggle two id representations to join an embedding against a node.
    """
    with np.load(path) as npz:
        ids = npz["ids"]
        vectors = npz["vectors"]
    return {str(int(node_id)): vectors[i] for i, node_id in enumerate(ids)}


def read_doc_embeddings(path: Path) -> dict[str, np.ndarray]:
    """STaRK's precomputed candidate embeddings, keyed by node id."""
    return _read_embeddings(path)


def read_query_embeddings(path: Path) -> dict[str, np.ndarray]:
    """STaRK's precomputed query embeddings, keyed by query id."""
    return _read_embeddings(path)

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

from stark_bench.ports import Query

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

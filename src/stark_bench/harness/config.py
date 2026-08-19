"""Every knob that changes a number, in one file per run.

The resolved contents are embedded verbatim in the results file. Re-running a
variant is an edit here, and a number whose config is not recorded is not
re-runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    dataset: str
    split: str
    chunker: str
    embeddings: str
    dimension: int
    aggregation: str
    agent: str
    k: int
    raw: str


def load_config(path: Path) -> RunConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return RunConfig(raw=raw, **data)

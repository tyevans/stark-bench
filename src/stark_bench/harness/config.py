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
    #: The chat model an LLM agent talks to, as the server's own model id.
    #: Optional: `dense` and `hybrid` make no LLM call at all, so a config
    #: for either would have nothing to say here. `None` means the CLI's
    #: `DEFAULT_CHAT_MODEL`.
    chat_model: str | None = None


def load_config(path: Path) -> RunConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return RunConfig(raw=raw, **data)

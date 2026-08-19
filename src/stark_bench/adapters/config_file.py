"""Reading a `RunConfig` off disk.

An adapter because it does two things the domain must not: it touches the
filesystem and it parses YAML. `raw` carries the bytes it read, unmodified,
because resume safety is decided by a byte-identical comparison and a
re-serialised config would not be one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from stark_bench.domain.run_config import RunConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_config(path: Path) -> RunConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return RunConfig(raw=raw, **data)

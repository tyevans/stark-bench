"""The seams: what the benchmark needs, stated without saying who provides it.

Protocols only. A port names a capability the application depends on; an
adapter under `adapters/` provides one. Nothing here imports an adapter, and
nothing here does I/O -- a port that could reach a concrete implementation
would be documentation rather than a boundary.

`domain` is the one thing ports may import, because a capability is stated
in terms of the values it moves.
"""

from stark_bench.ports.agent import Agent, BudgetTracker, Toolset
from stark_bench.ports.corpus import ChunkIdIndex

__all__ = ["Agent", "BudgetTracker", "ChunkIdIndex", "Toolset"]

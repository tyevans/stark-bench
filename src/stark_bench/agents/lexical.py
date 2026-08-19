"""BM25 alone, so the fusion's contribution stops being inferred.

## Why this exists

`hybrid` is vector **and** BM25, fused by rank inside redstring, and
`dense` is vector alone -- so `hybrid - dense` was being read as "what
lexical adds". That is a subtraction, not a measurement: it assumes the
fusion contributes exactly the difference and that neither channel changes
the other's behaviour, which is precisely what rank fusion does not
guarantee.

Measured on 2026-08-19 across three corpora, that difference went from
+0.0005 mrr at 1.000 chunks/node to +0.0141 at 1.139 -- a real effect, and
one nobody could attribute without knowing what BM25 scores on its own.
A corpus where lexical alone is strong and a corpus where fusion merely
reorders a strong dense result look identical in the subtraction.

There is a second reason, and it is the one that made this file get
written. The name `hybrid` was read as "graph plus lexical", and three
findings were reported on that basis before anyone opened the module. It
uses no graph at all: `dense` and `hybrid` both call `search_chunks`, and
neither reaches a relationship. An explicit third column makes the
decomposition visible in the results table instead of living in someone's
head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Toolset


@dataclass(frozen=True, slots=True)
class LexicalAgent:
    """BM25 over chunk text, folded up to STaRK nodes."""

    k: int = 20
    name: str = "lexical"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        return await tools.search_chunks(query.text, k=self.k, mode="lexical")

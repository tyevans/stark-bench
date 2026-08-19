"""Retrieve a wide candidate list, then have the LLM score each one on its text.

The architecture the other four agents leave out. `dense`, `lexical` and
`hybrid` return the bi-encoder's ordering unchanged; `zero_shot` rewrites the
query and returns the bi-encoder's ordering of the result; `deep` plans tool
calls. **None of them ever put a document in front of the LLM** -- until
`search_passages` there was no way to, since `get_node` returns a name and a
type and `search_chunks` discards the text it matched on.

That matters most where a bi-encoder is weakest. A single vector has to commit
to one reading of a document before it knows the query; a reranker sees both
at once, which is why STaRK's own leaderboard runs an LLM reranker over the
same ada-002 candidates and reports it separately.

Two decisions worth stating, because both are places this could measure the
wrong thing:

**Scores are relevance, not a permutation.** Asking for a reordered id list
invites the model to drop, invent or duplicate ids, and every one of those has
to be repaired by code that then decides the ranking itself. Scoring each
candidate independently makes a malformed answer a *missing score* for one
candidate rather than a corrupted list.

**Retrieval order breaks ties, and unscored candidates keep their place
below the scored ones.** A reranker that fails on every candidate must
degrade to exactly `hybrid`, not to alphabetical order -- otherwise a
comparison against `hybrid` measures the failure path instead of the
reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Toolset

from stark_bench.domain import Ranked as _Ranked

logger = logging.getLogger(__name__)

#: How much of a candidate's text the model is shown. Sized against the
#: *context*, not against what would be nice to show: 40 candidates at 3000
#: characters is ~120k characters, ~30k tokens, under half the 64k window,
#: leaving room for the instructions and 40 scored objects coming back.
#:
#: The number is set by where the useful text sits, not by what fits. STaRK
#: puts a node's `- relations:` block at the *end* of its document, so a
#: small budget truncates the relations corpus exactly before the neighbour
#: names -- the reranker would read name, type and details and never reach
#: the thing that arm was built to test. At 3000 characters it clears the
#: relations corpus's 1,761-char mean with room for the long tail.
#:
#: Overflowing the window is the worst failure available to this agent: the
#: extract call raises, the agent degrades to retrieval order, and the run
#: scores *exactly* `hybrid` -- a plausible-looking null saying reranking
#: does not help, when the model never saw the prompt. Probe against the live
#: endpoint after changing this or `fetch`.
_MAX_PASSAGE_CHARS = 3_000


class Relevance(BaseModel):
    """One candidate's score, on a scale the prompt pins to examples."""

    node_id: str
    score: float = Field(ge=0.0, le=10.0)


class Relevances(BaseModel):
    scores: list[Relevance]


_PROMPT_TEMPLATE = (
    "You are ranking candidate entities from a biomedical knowledge base "
    "against a search query. Score every candidate from 0 to 10 for how well "
    "it answers the query: 10 means it is exactly what the query asks for, 5 "
    "means it is the right kind of thing but fails one stated condition, 0 "
    "means it is unrelated. Judge only from the text shown. Return one score "
    "for every candidate id, and invent no ids.\n\n"
    "Query: {query}\n\nCandidates:\n{candidates}"
)


@dataclass(frozen=True, slots=True)
class RerankAgent:
    k: int = 20
    #: Candidates fetched before reranking. Reranking can only reorder what
    #: retrieval found, so this -- not `k` -- is the ceiling on what the
    #: architecture can fix, and `recall@20` of the underlying `hybrid` run
    #: is what it is bounded by.
    fetch: int = 40
    name: str = "rerank"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        passages = await tools.search_passages(query.text, k=self.fetch, mode="hybrid")
        if not passages:
            return []

        rendered = "\n\n".join(
            f"[{p.node_id}] {p.text[:_MAX_PASSAGE_CHARS]}" for p in passages
        )
        try:
            judged = await tools.extract(
                _PROMPT_TEMPLATE.format(query=query.text, candidates=rendered),
                Relevances,
            )
        except Exception:
            # Logged, not swallowed quietly. A reranker whose every call
            # fails returns retrieval order, which scores *identically* to
            # `hybrid` -- the one failure of this agent that looks like a
            # result. `grep 'rerank: extract failed'` on a run log is what
            # separates "reranking did not help" from "reranking did not run".
            logger.warning("rerank: extract failed for query %s", query.query_id)
            judged = None

        retrieval_rank = {p.node_id: i for i, p in enumerate(passages)}
        scores = (
            {r.node_id: r.score for r in judged.scores if r.node_id in retrieval_rank}
            if judged is not None
            else {}
        )

        # Unscored candidates sort below every scored one, in retrieval
        # order. `-1.0` rather than `0.0`: a candidate the model actively
        # judged irrelevant and one it never mentioned are different facts,
        # and collapsing them would let a truncated answer promote whatever
        # the model happened to reach.
        ordered = sorted(
            passages,
            key=lambda p: (-scores.get(p.node_id, -1.0), retrieval_rank[p.node_id]),
        )
        return [
            _Ranked(node_id=p.node_id, score=1.0 / (1 + rank))
            for rank, p in enumerate(ordered[: self.k])
        ]

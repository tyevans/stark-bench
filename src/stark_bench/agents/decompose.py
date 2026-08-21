"""Split a conjunctive query, retrieve each part, fuse, then score.

## The hypothesis, recorded before the run

PRIME's queries are conjunctive by construction -- "a drug that targets X
and is indicated for Y". A single dense vector has to carry both
constraints at once, and the campaign's own numbers say it cannot: adding
relational text to the corpus moved **dense by +2%** and **lexical by
+22%**. That asymmetry is the dense channel compressing a conjunction away
while BM25 matches the neighbour names as literal tokens.

Relational corpora attacked that from the document side, by putting the
neighbour names where BM25 could find them. This attacks it from the
**query** side: split the query so each constraint gets its own
full-strength retrieval, then let fusion do the conjunction. A candidate
appearing in several sub-result lists is precisely what a conjunctive
query is asking for, and RRF gives that for free.

**Prediction, on record so it can be wrong.** The bar is `rerank40` on
whole documents at **0.46323**. Lean encodings have given up 0.05-0.07
against full documents, so decomposition has to buy back more than that
gap. Expected to beat `hybrid` (0.27711 on this corpus) comfortably and
`rerank40titlerelranked` (0.39343) by a smaller margin; beating 0.46323 is
the outcome that would change the recommendation.

## Why this is not `deep` again

`deep` is the worst arm this project has run -- 0.1851 and 0.2015, below
`lexical` on one corpus -- at 7.46 LLM calls per query against a cap of 8.
Running to exhaustion is the tell: it was searching, not deciding.

Here the plan is made once and the rest is deterministic. Two LLM calls per
query, no loop, no budget to exhaust. The agency is in *what to ask*, not
in how long to look.

## Three properties that are load-bearing rather than tidy

**The original query is always in the fusion set.** That makes this a
strict superset of `hybrid`: a decomposition that adds only noise gets
diluted by RRF and the floor stays near `hybrid` rather than at "whatever
the decomposer happened to produce". A silent failure bounded below by a
known-good arm is worth a great deal here, where nine of the last ten real
defects raised no exception.

**The decomposer is told to copy entity names verbatim.** This is the
sharpest risk in the design. The entire measured gain is lexical -- BM25
matching `Chronic myeloid leukemia` as tokens. A decomposer that helpfully
paraphrases to "blood cancer of the myeloid line" destroys the exact match
that makes the channel work, degrading the arm to roughly dense-only while
every count in the report looks perfect. It would read as "decomposition
does not help", which is exactly the wrong conclusion.

**Relations are ranked against the sub-queries, not the raw query.**
FINDINGS 1b measured relation *selection* at **+0.083 mrr** -- the largest
single lever in the campaign -- and every existing arm selects using the
blurred whole query. A decomposition separates the constraints, so the
names kept per candidate can be the ones matching the constraint actually
being tested. This is information no reranking arm here has had.

It also generalises the `matrix` encoding, whose three **fixed** dimensions
logged 431 warnings on 2026-08-21 that the model scored them identically.
Prescribed dimensions are not orthogonal for an arbitrary query; decomposed
constraints are, because the query itself drew the lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from stark_bench.agents.rerank import ranked_relations, title_of
from stark_bench.domain import Ranked

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stark_bench.domain import Passage, Query
    from stark_bench.ports.agent import Toolset

logger = logging.getLogger(__name__)

#: RRF's rank offset. 60 is the constant from the original paper and the
#: one redstring's own fusion uses; changing it here would make this arm's
#: fusion incomparable to `hybrid`'s for no measured reason.
_RRF_K = 60

#: Sub-queries requested. Bounded because over-decomposition dilutes: every
#: extra list contributes rank-1 mass to whatever it found, so a spurious
#: constraint promotes a spurious candidate to the same degree a real one
#: promotes a real candidate.
_MAX_SUB_QUERIES = 4

_DECOMPOSE_PROMPT = (
    "Break this biomedical search query into its separate constraints.\n\n"
    "Each constraint becomes one standalone search query against a "
    "knowledge base of biomedical entities and their relationships.\n\n"
    "RULES:\n"
    "1. Copy entity names EXACTLY as they appear in the query -- character "
    "for character. Do NOT paraphrase, expand abbreviations, or substitute "
    "synonyms. The search is lexical and an altered name will not match.\n"
    "2. One constraint per query. If the query asks for something that "
    "targets X and treats Y, that is two constraints.\n"
    f"3. Return at most {_MAX_SUB_QUERIES} sub-queries. Fewer is fine.\n"
    "4. If the query expresses only one constraint, return it unchanged as "
    "a single sub-query.\n\n"
    "Query: {query}"
)

_SCORE_PROMPT = (
    "Rank these candidate entities against the search query.\n\n"
    "Query: {query}\n\n"
    "The query breaks into these constraints:\n{constraints}\n\n"
    "A candidate satisfying MORE constraints ranks higher. A candidate "
    "satisfying none ranks lowest.\n\n"
    "Return one [index, score] pair for EVERY candidate below, using the "
    "bracketed index exactly as shown: [[1, 90], [2, 15], ...]. Score 0-100. "
    "Return as many pairs as there are candidates. Do not return an empty "
    "list.\n\n"
    "Candidates:\n{candidates}"
)


class SubQueries(BaseModel):
    """The decomposition. Field docstrings are load-bearing.

    On 2026-08-20 a reranker returned `{"scores": []}` for most queries
    after the schema docstrings were stripped to save tokens -- the
    description was the only text telling the model what to put in the
    array, and its absence read as "titles are not enough", which was the
    hypothesis the arm existed to test. Do not trim these.
    """

    queries: list[str] = Field(
        description=(
            "Standalone search queries, one per constraint, with entity "
            "names copied verbatim from the original query."
        )
    )


class ScoredCandidates(BaseModel):
    """`[index, score]` pairs, indexed from 1 as rendered in the prompt."""

    scores: list[tuple[int, float]] = Field(
        description=(
            "One [index, score] pair per candidate, where index is the "
            "bracketed number shown beside that candidate and score is "
            "0-100. Include every candidate."
        )
    )


def _fuse(ranked_lists: Sequence[Sequence[Passage]]) -> list[Passage]:
    """Reciprocal-rank fusion over the sub-query result lists.

    The conjunction is done here rather than by the LLM: a candidate found
    by three sub-queries accumulates three contributions and outranks one
    found by a single sub-query, which is what "targets X **and** treats Y"
    means. Doing it arithmetically also means it still happens when the
    scoring call fails, so the fallback is a fused ranking rather than a
    single list.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Passage] = {}
    for ranked in ranked_lists:
        for rank, passage in enumerate(ranked, start=1):
            scores[passage.node_id] = scores.get(passage.node_id, 0.0) + 1.0 / (
                _RRF_K + rank
            )
            seen.setdefault(passage.node_id, passage)
    order = sorted(scores, key=lambda node_id: -scores[node_id])
    return [seen[node_id] for node_id in order]


@dataclass(frozen=True, slots=True)
class DecomposeAgent:
    k: int = 20
    #: Candidates surviving fusion that reach the scoring call. 40 is the
    #: measured economy knee: `rerank80titlerelranked` bought +0.011 mrr
    #: over `rerank40titlerelranked` for double the tokens.
    fetch: int = 40
    #: Retrieved per sub-query before fusion. Wider than `fetch` because
    #: fusion discards: a candidate ranked 30th by one sub-query and 30th by
    #: another should surface, and it cannot if each list stopped at 20.
    per_query_fetch: int = 40
    #: Neighbour names kept per relation type, and relation types shown.
    #: Held at the values `rerank40titlerelranked` measured (0.39343) so a
    #: difference is attributable to decomposition rather than to encoding
    #: width. Sweep after, not during.
    relation_per_type: int = 1
    relation_max_types: int = 8

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        sub_queries = await self._decompose(query, tools)

        # The original ALWAYS participates, so fusion cannot score below
        # `hybrid` on a bad decomposition. See the module docstring.
        searches = [query.text, *sub_queries]
        ranked_lists = [
            await tools.search_passages(text, k=self.per_query_fetch, mode="hybrid")
            for text in searches
        ]
        fused = _fuse(ranked_lists)
        if not fused:
            return []
        candidates = fused[: self.fetch]

        constraints = "\n".join(f"- {text}" for text in searches)
        rendered = self._render(candidates, searches)
        scores = await self._score(query, constraints, rendered, len(candidates), tools)

        return self._rank(candidates, scores)

    async def _decompose(self, query: Query, tools: Toolset) -> list[str]:
        """The sub-queries, or `[]` when the call fails or returns nothing.

        `[]` is not a defect: `retrieve` always searches the original query
        too, so an empty decomposition degrades this arm to `hybrid` plus a
        scoring pass rather than to nothing. It is logged as a degrading
        warning anyway, because a run where most decompositions failed is
        not a measurement of decomposition, and `llm_calls_per_query` would
        stay a clean 2.0 throughout.
        """
        try:
            plan = await tools.extract(
                _DECOMPOSE_PROMPT.format(query=query.text), SubQueries
            )
        except Exception:
            logger.warning(
                "decompose: extract failed for query %s -- empty scores, "
                "falling back to the original query alone",
                query.query_id,
            )
            return []
        wanted = [text.strip() for text in plan.queries if text.strip()]
        if not wanted:
            logger.warning(
                "decompose: empty scores for query %s -- the decomposer "
                "returned no sub-queries, falling back to the original alone",
                query.query_id,
            )
        return wanted[:_MAX_SUB_QUERIES]

    def _render(self, candidates: Sequence[Passage], searches: Sequence[str]) -> str:
        """Title plus the relations matching the constraints, per candidate.

        Relations are ranked against the joined sub-queries rather than the
        raw query -- the point of the design. `ranked_relations` takes a
        single string, so the constraints are joined; the lexical ranker
        scores on token overlap, so a union of constraint tokens is what a
        candidate satisfying any of them should match.
        """
        selector = " ".join(searches)
        lines: list[str] = []
        for index, passage in enumerate(candidates, start=1):
            title = title_of(passage.text)
            relations = ranked_relations(
                passage.text,
                selector,
                per_type=self.relation_per_type,
                max_types=self.relation_max_types,
            )
            body = f"{title} | {relations}" if relations else title
            lines.append(f"[{index}] {body}")
        return "\n".join(lines)

    async def _score(
        self,
        query: Query,
        constraints: str,
        rendered: str,
        count: int,
        tools: Toolset,
    ) -> dict[int, float] | None:
        try:
            judged = await tools.extract(
                _SCORE_PROMPT.format(
                    query=query.text,
                    constraints=constraints,
                    candidates=rendered,
                ),
                ScoredCandidates,
            )
        except Exception:
            logger.warning("decompose: extract failed for query %s", query.query_id)
            return None
        if not judged.scores:
            logger.warning(
                "decompose: empty scores for query %s -- falling back to "
                "fused retrieval order",
                query.query_id,
            )
            return None
        return {index: score for index, score in judged.scores if 1 <= index <= count}

    def _rank(
        self, candidates: Sequence[Passage], scores: dict[int, float] | None
    ) -> list[Ranked]:
        """Scored candidates first, then the rest in fused retrieval order.

        Backfill is mandatory rather than optional: `k=20` is scored on
        recall@20, and an agent that returned only what it scored would
        throw away every candidate the LLM declined to mention. Unscored
        candidates take `-1.0` so that "judged irrelevant" and "never
        mentioned" stay distinguishable, as in `rerank`.
        """
        scores = scores or {}
        ordered = sorted(
            range(len(candidates)),
            key=lambda i: (-scores.get(i + 1, -1.0), i),
        )
        return [
            Ranked(node_id=candidates[i].node_id, score=scores.get(i + 1, -1.0))
            for i in ordered[: self.k]
        ]

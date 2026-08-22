"""Split a conjunctive query, retrieve each part, then let the model unify.

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
full-strength retrieval, and reach candidates no single search ranks well.

## What the first version got wrong, measured

It fused the result lists with reciprocal-rank fusion, so a candidate found
by three sub-queries outranked one found by a single sub-query. **That was
the wrong step to automate.** A decomposed constraint can be genuinely part
of the question and still retrieve candidates nobody should surface, and
RRF promotes its top hit exactly as hard as the central constraint's.
Whether a match reinforces or merely co-occurs is a judgment.

| | mrr | recall@20 |
|---|---|---|
| `hybrid` | 0.28156 | 0.46821 |
| RRF fusion (this module, first version) | **0.37127** | 0.47843 |
| `rerank40titlerelranked` (one list, reordered) | 0.39343 | 0.50672 |

Fusion bought **+0.010 recall@20** over `hybrid` while a plain reranker
gained +0.038 by reordering `hybrid`'s own top 40. The pool was diluted,
not widened -- what promoting tangential matches looks like.

So the union is now the model's job. `_pool` takes the union, keeps which
searches found each candidate and at what rank, and renders that as
evidence: `(found by 0@3, 2@1)`. The prompt says explicitly that a search
may be tangential, that being retrieved by one is not an argument, and
that a candidate found by a single search may still be the best answer.

**Prediction, on record so it can be wrong.** Against `rerank40titlerelranked`
at 0.39343 on the same corpus and encoding. The mechanism that should pay
is recall: the pool is 100 wide against any single search's 40, so unlike
every `rerank*` arm this one can surface a candidate `hybrid` never
ranked. If recall@20 does not exceed 0.50672 the widening did not happen
and the idea is beaten here regardless of how the unify step scores.

## Why this is a workflow rather than an agent

One planning call, N concurrent retrievals, one unify call. No loop, no
tool selection, no budget to exhaust. The LLM is used twice as a pure
function.

That is a design choice about this problem, **not** a claim that agentic
search fails here. `deep` scored 0.1851 and 0.2015, but those runs are
weak evidence about the architecture: they ran on `prime` corpora with no
relations block -- on an agent whose premise is walking relationships --
where `hybrid` itself managed only 0.2187 against 0.34675 on the
relational corpora. They also predate lean observation encodings, ranked
relation selection, ANN indexes and `ef` tuning, and a context bound loose
enough that a 72,000-character observation reached the model untouched.
See B-DEEP-NEVER-FAIRLY-TESTED-1.

## Two properties that are load-bearing rather than tidy

**The original query is always search 0**, and an exact tie on best rank
breaks toward it. That makes a useless decomposition degrade to `hybrid`'s
own ordering rather than to something arbitrary. A silent failure bounded
below by a known-good arm is worth a great deal where nine of the last ten
real defects raised no exception.

**The decomposer is told to copy entity names verbatim.** The entire
measured gain is lexical -- BM25 matching `Chronic myeloid leukemia` as
tokens. A decomposer that paraphrases to "blood cancer of the myeloid
line" destroys the exact match, degrading the arm to roughly dense-only
while every count in the report looks perfect. Probed on gemma before the
first run: 4 of 4 queries decomposed with every sub-query verbatim.

Relation selection still ranks against the joined searches rather than
per-constraint, which is a gap this docstring used to claim was closed.
See B-DECOMPOSE-SELECTION-1.
"""

from __future__ import annotations

import logging
from asyncio import gather
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
    "The query was split into these numbered searches, and (0) is the "
    "original query itself:\n{constraints}\n\n"
    "Each candidate is marked with which searches retrieved it and at what "
    "rank -- `(found by 0@3, 2@1)` means search (0) ranked it 3rd and "
    "search (2) ranked it 1st.\n\n"
    "Use that as EVIDENCE, not as a score. In particular:\n"
    "- A search may be tangential. Being retrieved by it does not make a "
    "candidate an answer, and a candidate found by several tangential "
    "searches is still not an answer.\n"
    "- Judge each candidate against the ORIGINAL query. A good answer "
    "satisfies what was actually asked, whichever searches happened to "
    "find it.\n"
    "- A candidate found by only one search may still be the best answer.\n\n"
    "Score 0-100, where 100 is certainly the answer and 0 is irrelevant. "
    "Return one [index, score] pair for EVERY candidate below, using the "
    "bracketed index exactly as shown: [[1, 90], [2, 15], ...]. Return as "
    "many pairs as there are candidates. Do not return an empty list.\n\n"
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


@dataclass(frozen=True, slots=True)
class Candidate:
    """One node in the pool, with which constraints reached it and how well.

    `matches` maps a search's index in `searches` (0 is always the original
    query) to the best rank that search gave this node. It is the evidence
    the LLM needs to unify: "found by constraint 2 at rank 1" and "found by
    constraints 1 and 3 at ranks 30 and 34" are different facts about a
    candidate, and no single number preserves both.
    """

    passage: Passage
    matches: dict[int, int]

    @property
    def best_rank(self) -> int:
        return min(self.matches.values())


def _pool(ranked_lists: Sequence[Sequence[Passage]], *, limit: int) -> list[Candidate]:
    """The union of the result lists, with retrieval provenance kept.

    ## Why this is a union and not a fusion

    The first version summed reciprocal ranks across lists, so a candidate
    found by three sub-queries outranked one found by a single sub-query.
    That treats every constraint as equally load-bearing, and a decomposed
    constraint may be **tangential**: a sub-query can be a genuine part of
    the question and still retrieve candidates that nobody should surface,
    while its top hit gets promoted exactly as hard as the central
    constraint's. Whether a match reinforces or merely co-occurs is a
    judgment, and arithmetic cannot make it.

    Measured: RRF fusion scored **0.37127 mrr** on `qwen-rel-whole` and
    lifted recall@20 to 0.47843 against `hybrid`'s 0.46821 -- **+0.010**,
    while a plain reranker over `hybrid`'s own top 40 reached 0.50672 by
    reordering alone. The fused pool was diluted rather than widened, which
    is what promoting tangential matches looks like.

    ## Why truncation orders by best rank rather than by match count

    Something must bound the prompt, and the ordering used to bound it is
    itself a ranking decision. Best-rank asks "did any search find this
    highly", which admits a tangential constraint's strong hit into the
    pool **without promoting it** -- the LLM then sees it matched only that
    constraint and can drop it. Ordering by match count would rebuild the
    additive bias one layer down, where it is harder to see.

    Ties break toward the original query's own ranking, so with no usable
    decomposition this degrades exactly to `hybrid`'s order.
    """
    matches: dict[str, dict[int, int]] = {}
    seen: dict[str, Passage] = {}
    for index, ranked in enumerate(ranked_lists):
        for rank, passage in enumerate(ranked, start=1):
            per_node = matches.setdefault(passage.node_id, {})
            # `min`: a search that found a node twice (it cannot) or two
            # searches that both found it keep the better evidence.
            per_node[index] = min(per_node.get(index, rank), rank)
            seen.setdefault(passage.node_id, passage)

    candidates = [
        Candidate(passage=seen[node_id], matches=per_node)
        for node_id, per_node in matches.items()
    ]
    candidates.sort(key=lambda c: (c.best_rank, c.matches.get(0, 10**6)))
    return candidates[:limit]


@dataclass(frozen=True, slots=True)
class DecomposeAgent:
    k: int = 20
    #: Candidates reaching the unify call.
    #:
    #: 100, not the 40 that is the economy knee for `rerank*`. Those arms
    #: rerank ONE list, so a wider fetch buys progressively worse candidates
    #: -- `rerank80titlerelranked` gained only +0.011 mrr for double the
    #: tokens. This arm pools SEVERAL lists, so the extra slots hold each
    #: search's own strong hits rather than another search's tail, and a
    #: pool no wider than one list means the union can only rearrange which
    #: candidates survive, never reach one a single search missed. Reaching
    #: those is the entire reason to decompose.
    #:
    #: Affordable because the encoding is lean: ~200 characters a candidate,
    #: so 100 of them is ~20k characters against a 65,536-token context.
    fetch: int = 100
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
        # Concurrently: independent reads, and hybrid search is dominated by
        # BM25 over a 5.7M-row terms table, so five sequential awaits per
        # query were a large share of this arm's wall time.
        ranked_lists = await gather(
            *(
                tools.search_passages(text, k=self.per_query_fetch, mode="hybrid")
                for text in searches
            )
        )
        candidates = _pool(ranked_lists, limit=self.fetch)
        if not candidates:
            return []

        constraints = "\n".join(f"({i}) {text}" for i, text in enumerate(searches))
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

    def _render(self, candidates: Sequence[Candidate], searches: Sequence[str]) -> str:
        """Title, relations, and which constraints actually reached this node.

        The retrieval evidence is rendered verbatim -- `(found by 0@3, 2@1)`
        means the original query ranked it third and constraint 2 ranked it
        first -- rather than summarised into a score. That is the whole
        point of the rewrite: whether a match reinforces or merely
        co-occurs is a judgment, and a count throws away exactly the
        distinction that makes a constraint tangential.

        Relations are still selected against the joined searches, which is
        NOT the per-constraint selection this module's docstring argues
        for. See B-DECOMPOSE-SELECTION-1: changing both the unify step and
        the selection at once would make neither attributable.
        """
        selector = " ".join(searches)
        lines: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            passage = candidate.passage
            title = title_of(passage.text)
            relations = ranked_relations(
                passage.text,
                selector,
                per_type=self.relation_per_type,
                max_types=self.relation_max_types,
            )
            body = f"{title} | {relations}" if relations else title
            found = ", ".join(
                f"{i}@{rank}" for i, rank in sorted(candidate.matches.items())
            )
            lines.append(f"[{index}] (found by {found}) {body}")
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
        self, candidates: Sequence[Candidate], scores: dict[int, float] | None
    ) -> list[Ranked]:
        """Scored candidates first, then the rest in pooled retrieval order.

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
            Ranked(
                node_id=candidates[i].passage.node_id,
                score=scores.get(i + 1, -1.0),
            )
            for i in ordered[: self.k]
        ]

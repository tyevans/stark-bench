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

#: Characters of any one candidate that reach the unify prompt.
#:
#: `rerank` caps at 3,000 because it renders whole documents. This arm
#: renders a lean encoding -- title plus one neighbour per relation type --
#: which is ~200 characters typically, so 500 is generous rather than
#: tight, and 80 candidates fit in ~40k characters against a 65,536-token
#: context.
#:
#: It exists because the typical case is not the binding one. PRIME hub
#: entities carry dozens of relation types with long neighbour names, and
#: on 2026-08-21 exactly one query in 280 produced a prompt the endpoint
#: rejected with `400`. `rerank`'s own docstring names this as the worst
#: failure available to these agents: the extract call raises, the agent
#: degrades to retrieval order, and the arm scores like `hybrid` while
#: every count in the report looks clean. Widening the pool is what made it
#: reachable, and the pool has since widened again to 80.
_MAX_CANDIDATE_CHARS = 500

#: Planned queries requested, beside the original. The default is 2, so
#: three searches run in total.
#:
#: Fewer and deeper beats more and shallower here. The union's value is
#: reach -- candidates a single search never ranked -- and reach comes from
#: how far down each search goes, not from how many searches there are.
#: Five searches at k=40 and three at k=80 cost the same retrieval work;
#: the second reaches rank 80.
#:
#: It also bounds dilution. Every extra planned query contributes its own
#: rank-1 hit to the pool, and those displace the original query's ranks
#: 2-5 -- measured as `rephraseshort`'s recall@20 falling BELOW plain
#: `hybrid` when the model did not rerank the whole pool.
_MAX_SUB_QUERIES = 2

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
    "3. Return at most 4 sub-queries. Fewer is fine.\n"
    "4. If the query expresses only one constraint, return it unchanged as "
    "a single sub-query.\n\n"
    "Query: {query}"
)

#: The shortlist wording, kept because it is a different instrument rather
#: than a worse one.
#:
#: Measured on 50 queries: a MEDIAN of 2.5 candidates named out of 40, and
#: 16 of 50 naming exactly one. Backfill keeps recall@20 whole, so what
#: this trades is positions 3-20 -- which become retrieval order -- for a
#: model that only commits where it is confident. If hit@1 is materially
#: better this way, that is a result about precision and not a defect.
_SHORTLIST_PROMPT_TAIL = (
    "Return the bracketed indexes of the candidates you believe answer the "
    "query, most relevant first: [4, 17, 2, ...]. Return at most 20, and "
    "fewer -- even one -- when only a few are plausible. Do not pad the "
    "list with candidates you do not believe. Do not repeat an index and "
    "do not return an empty list."
)

_ORDER_PROMPT = (
    "You are choosing which entities answer a biomedical search query.\n\n"
    "QUERY: {query}\n\n"
    "Candidates are grouped by the search that found them. The first group "
    "is the query itself; the others are supplemental searches for single "
    "constraints, and a supplemental search may be TANGENTIAL -- being "
    "found by one is not evidence that a candidate answers the query.\n\n"
    "{groups}\n"
    "Judge every candidate against the QUERY above, not against the search "
    "that happened to find it. A candidate listed under one supplemental "
    "search may still be the best answer; a candidate listed under several "
    "may be no answer at all.\n\n"
    "{instruction}"
)


_RANK_ALL_INSTRUCTION = (
    "RANK the candidates -- do not select only the ones you are sure "
    "about. Return exactly 20 bracketed indexes, best first: "
    "[4, 17, 2, ...].\n\n"
    "Return 20 even when only one or two look right. Scoring counts where "
    "the answer lands in your list, so a low-confidence guess at position "
    "20 costs you nothing and can only help; leaving it out throws the "
    "position away. If fewer than 20 candidates exist, return all of them. "
    "Do not repeat an index and do not return an empty list."
)


_REPHRASE_PROMPT = (
    "Rewrite this biomedical search query {count} different ways.\n\n"
    "Each rewrite must ask THE SAME question, complete -- not a piece of "
    "it. Vary how it is asked: the word order, the framing, the general "
    "vocabulary, whether it reads as a question or a description.\n\n"
    "RULES:\n"
    "1. Copy every entity name, gene symbol, drug name, disease name and "
    "identifier EXACTLY as written -- character for character. The search "
    "is partly lexical and an altered name will not match. Rephrase the "
    "words AROUND them.\n"
    "2. Keep every constraint. If the query asks for something that targets "
    "X and treats Y, every rewrite must still ask for both.\n"
    "3. Do not answer the question or guess what the answer is.\n\n"
    "Query: {query}"
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


class Ordering(BaseModel):
    """The top candidates, best first, by their bracketed index.

    An ORDER rather than scores, which is both the easier task and the one
    the benchmark actually measures. Scoring every candidate asks for
    `fetch` numbers of decode and invites the degenerate answer: the
    `matrix` encoding logged 431 warnings on 2026-08-21 that the model gave
    the same number for every dimension. An ordered list of 20 integers is
    ~20 tokens, and "which twenty, in what order" is exactly the question
    MRR and Hit@k score.

    Field docstrings are load-bearing. On 2026-08-20 a reranker returned
    `{"scores": []}` for most queries after the schema descriptions were
    stripped to save tokens -- the description was the only text telling
    the model what to put in the array, and its absence read as an
    architecture result. Do not trim these.
    """

    indexes: list[int] = Field(
        description=(
            "Exactly 20 bracketed indexes, ranked most relevant first, or "
            "all of them if fewer than 20 exist. This is a ranking, not a "
            "shortlist: include uncertain candidates at the bottom rather "
            "than omitting them. Use each index at most once, and invent "
            "no indexes."
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
    #: Planned queries beside the original; three searches at the default.
    sub_queries: int = _MAX_SUB_QUERIES
    #: Candidates reaching the unify call.
    #:
    #: 80. Ranking 20 out of 80 rather than out of 40 doubles what the
    #: model can reach without changing what it is asked for -- it returns
    #: 20 either way, so the extra rows cost prefill and no decode. At ~200
    #: characters a candidate that is ~16k characters, roughly 5k tokens
    #: against a 65,536-token context.
    #:
    #: The earlier 100 was measured and reverted for the wrong reason: it
    #: bought +0.001 recall@20 on `decompose`, where the pool was fed by
    #: fragment-matching sub-queries and widening it added noise. Under
    #: paraphrase every search asks the whole question, so a wider pool
    #: holds more on-topic candidates rather than more distractors -- which
    #: is why `rephrase` gained +0.078 recall where `decompose` gained
    #: +0.011.
    fetch: int = 80
    #: Retrieved per search before pooling.
    #:
    #: Equal to `fetch`, so the pool is a genuine union: a candidate ranked
    #: 70th by two different searches reaches the prompt, where a narrower
    #: per-search fetch would drop it before the union ever saw it.
    per_query_fetch: int = 80
    #: Neighbour names kept per relation type, and relation types shown.
    #: Held at the values `rerank40titlerelranked` measured (0.39343) so a
    #: difference is attributable to decomposition rather than to encoding
    #: width. Sweep after, not during.
    relation_per_type: int = 1
    relation_max_types: int = 8
    #: Ask for whole-question paraphrases instead of constraint pieces.
    #:
    #: Decomposition lost on `prime-rel` (0.37947 against a plain
    #: reranker's 0.39343) and the diagnosis was that the corpus solves the
    #: conjunction at index time: the relations block puts the neighbour
    #: names in the document, so BM25 already matches "targets X and treats
    #: Y" against a document containing X and Y.
    #:
    #: Paraphrase attacks a different weakness -- **phrasing mismatch**,
    #: which is a dense-channel problem, and dense is the channel that has
    #: never moved here (0.18-0.25, and +2% from relational text against
    #: lexical's +22%).
    #:
    #: It also removes this design's worst failure mode by construction. A
    #: decomposed sub-query asks part of the question, so its hits can be
    #: tangential and the unify step has to judge that. A paraphrase asks
    #: the WHOLE question, so every search is full strength and every hit
    #: is on-topic.
    rephrase: bool = False
    #: Ask for a full ranking of 20 rather than only the candidates the
    #: model believes in.
    #:
    #: These are different instruments. Under MRR and recall@20, omitting a
    #: candidate can only lose position -- a guess at rank 20 is free -- so
    #: a full ranking should win on those. A shortlist commits only where
    #: the model is confident, which is the better shape if hit@1 is what
    #: matters. Measured separately rather than assumed.
    rank_all: bool = True

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        sub_queries = await self._plan(query, tools)

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

        groups = self._render(candidates, searches)
        ordering = await self._order(query, groups, len(candidates), tools)

        return self._rank(candidates, ordering)

    async def _plan(self, query: Query, tools: Toolset) -> list[str]:
        """The sub-queries, or `[]` when the call fails or returns nothing.

        `[]` is not a defect: `retrieve` always searches the original query
        too, so an empty decomposition degrades this arm to `hybrid` plus a
        scoring pass rather than to nothing. It is logged as a degrading
        warning anyway, because a run where most decompositions failed is
        not a measurement of decomposition, and `llm_calls_per_query` would
        stay a clean 2.0 throughout.
        """
        if self.sub_queries <= 0:
            # No planning call at all, rather than asking for zero
            # rewrites. This is the control for the whole design: one
            # search, the same pool machinery, the same grouped render and
            # the same ordering output, so the paraphrase union is the only
            # thing that differs from `rephrase`.
            return []
        try:
            plan = await tools.extract(
                (
                    _REPHRASE_PROMPT.format(query=query.text, count=_MAX_SUB_QUERIES)
                    if self.rephrase
                    else _DECOMPOSE_PROMPT.format(query=query.text)
                ),
                SubQueries,
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
        return wanted[: self.sub_queries]

    def _render(self, candidates: Sequence[Candidate], searches: Sequence[str]) -> str:
        """Candidates grouped under the search that found them.

        The flat list this replaced annotated each candidate `(found by
        0@3, 2@1)` and left the model to reconstruct the structure. Grouping
        states it: here is the original query and what it found, here are
        the supplemental searches and what they found. A candidate reached
        by more than one search appears once, under its best-ranking
        search, with the others noted -- repeating it would make the same
        entity look like several.

        Relations are still selected against the joined searches rather
        than per-constraint. See B-DECOMPOSE-SELECTION-1.
        """
        selector = " ".join(searches)
        # Each candidate belongs to the search that ranked it best; ties go
        # to the lower search index, so the original query keeps its own.
        owned: dict[int, list[tuple[Candidate, int]]] = {}
        for position, candidate in enumerate(candidates):
            owner = min(candidate.matches, key=lambda i: (candidate.matches[i], i))
            owned.setdefault(owner, []).append((candidate, position))

        lines: list[str] = []
        for search_index, text in enumerate(searches):
            group = owned.get(search_index)
            if not group:
                continue
            heading = (
                "THE QUERY ITSELF found:"
                if search_index == 0
                else f"Supplemental search ({search_index}) {text!r} found:"
            )
            lines.append(heading)
            group.sort(key=lambda pair: pair[0].matches[search_index])
            for candidate, position in group:
                passage = candidate.passage
                title = title_of(passage.text)
                relations = ranked_relations(
                    passage.text,
                    selector,
                    per_type=self.relation_per_type,
                    max_types=self.relation_max_types,
                )
                body = f"{title} | {relations}" if relations else title
                others = sorted(i for i in candidate.matches if i != search_index)
                also = f" (also found by {others})" if others else ""
                lines.append(f"  [{position + 1}]{also} {body[:_MAX_CANDIDATE_CHARS]}")
            lines.append("")
        return "\n".join(lines)

    async def _order(
        self, query: Query, groups: str, count: int, tools: Toolset
    ) -> list[int] | None:
        """The model's chosen ordering, or `None` when it gave us nothing.

        Indexes outside the candidate range are dropped and repeats keep
        their first position: a model that names an index twice meant it
        once, and honouring the repeat would push a real candidate off the
        end of `k`.
        """
        try:
            chosen = await tools.extract(
                _ORDER_PROMPT.format(
                    query=query.text,
                    groups=groups,
                    instruction=(
                        _RANK_ALL_INSTRUCTION
                        if self.rank_all
                        else _SHORTLIST_PROMPT_TAIL
                    ),
                ),
                Ordering,
            )
        except Exception:
            logger.warning("decompose: extract failed for query %s", query.query_id)
            return None
        if not chosen.indexes:
            logger.warning(
                "decompose: empty scores for query %s -- the model returned "
                "no ordering, falling back to pooled retrieval order",
                query.query_id,
            )
            return None
        seen: set[int] = set()
        ordered = [
            index
            for index in chosen.indexes
            if 1 <= index <= count and not (index in seen or seen.add(index))
        ]
        if not ordered:
            logger.warning(
                "decompose: empty scores for query %s -- every index the "
                "model returned was out of range",
                query.query_id,
            )
            return None
        return ordered

    def _rank(
        self, candidates: Sequence[Candidate], ordering: Sequence[int] | None
    ) -> list[Ranked]:
        """The model's order first, then the pool's own order behind it.

        Backfill is mandatory rather than optional. `k=20` is scored on
        recall@20, and the model is told it may return fewer than 20 -- so
        an agent that returned only what the model named would throw away
        every candidate it declined to mention, and recall would collapse
        while MRR looked fine.

        Scores here are positional, not the model's own: it was asked for
        an ORDER, and inventing a score to represent a rank would put a
        number in the report that nothing measured. Descending from 1.0
        keeps `Ranked` sortable and preserves the order the model gave.
        Backfilled candidates take `-1.0`, so "the model declined to name
        this" stays distinguishable from "the model ranked it last".
        """
        # Deduplicated and bounded HERE as well as in `_order`, because a
        # repeat or a stray index reaching this point would put the same
        # node in the ranking twice -- and `write_predictions` keys by node
        # id, so the duplicate collapses on write and the arm silently
        # returns 19 candidates where it reported 20.
        seen: set[int] = set()
        chosen = [
            index
            for index in (ordering or ())
            if 1 <= index <= len(candidates) and not (index in seen or seen.add(index))
        ]
        named = set(chosen)
        rest = [i + 1 for i in range(len(candidates)) if (i + 1) not in named]
        final = (chosen + rest)[: self.k]
        return [
            Ranked(
                node_id=candidates[index - 1].passage.node_id,
                score=(1.0 - position / len(final)) if index in named else -1.0,
            )
            for position, index in enumerate(final)
        ]

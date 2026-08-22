"""Which agent a config name means, and how each one is built.

The four architectures do not share a constructor. `DenseAgent`, `HybridAgent`
and `ZeroShotAgent` take `k` alone; `DeepAgent` additionally requires a
`BudgetTracker` and deliberately has no default for it -- `agents/` is
forbidden from importing `domain.budget`, where the concrete `Budget`
lives, so composition is the only place that can supply one. A plain
`{name: class}` mapping cannot express that, which is why this module holds
a mapping of *builders* instead.

## A budget is per query, not per run

`application.run_queries.run` constructs one agent and calls `retrieve` once per query,
so a `DeepAgent` holding a single `Budget` would spend the whole run's
allowance on query 1 and return `[]` for the remaining eleven thousand -- a
silently near-zero score rather than a crash, which is the worst shape a
defect of this kind can take. `PerQueryDeepAgent` closes that by building a
fresh `Budget` and a fresh `DeepAgent` for every `retrieve`, which is also the
only reading of "budget" that makes the cost column comparable across
architectures: every other agent's cost is stated per query too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from stark_bench.agents.dense import DenseAgent
from stark_bench.agents.decompose import DecomposeAgent
from stark_bench.agents.deep import DeepAgent
from stark_bench.agents.hybrid import HybridAgent
from stark_bench.agents.lexical import LexicalAgent
from stark_bench.agents.rerank import RerankAgent
from stark_bench.agents.zero_shot import ZeroShotAgent
from stark_bench.domain.budget import Budget

if TYPE_CHECKING:
    from collections.abc import Callable

    from stark_bench.domain.run_config import RunConfig
    from stark_bench.domain import Query, Ranked
    from stark_bench.ports import Agent, Toolset

#: Per *query*, not per run -- see the module docstring. Sized so a deep run
#: over the test split cannot cost an order of magnitude more than the
#: zero-shot one without that showing up as budget exhaustion rather than as
#: an endpoint melting: eight LLM rounds, eight tool calls, one minute.
MAX_TOOL_CALLS = 8
MAX_LLM_CALLS = 8
MAX_SECONDS = 60.0


@dataclass(slots=True)
class PerQueryDeepAgent:
    """A `DeepAgent` rebuilt, with a fresh budget, for every query."""

    k: int = 20
    max_tool_calls: int = MAX_TOOL_CALLS
    max_llm_calls: int = MAX_LLM_CALLS
    max_seconds: float = MAX_SECONDS
    name: str = "deep"

    #: How many queries ended at the cap. Recorded rather than merely raised,
    #: for the same reason `Budget.exhausted` outlives the exception.
    exhausted_queries: int = field(default=0, init=False)

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        budget = Budget(
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            max_seconds=self.max_seconds,
        )
        result = await DeepAgent(k=self.k, budget=budget).retrieve(query, tools)
        if budget.exhausted:
            self.exhausted_queries += 1
        return result


AGENTS: dict[str, Callable[[RunConfig], Agent]] = {
    "dense": lambda config: DenseAgent(k=config.k),
    "hybrid": lambda config: HybridAgent(k=config.k),
    "lexical": lambda config: LexicalAgent(k=config.k),
    "zero_shot": lambda config: ZeroShotAgent(k=config.k),
    "deep": lambda config: PerQueryDeepAgent(k=config.k),
    "rerank": lambda config: RerankAgent(k=config.k),
    #: The same architecture with a wider retrieval window, registered as a
    #: separate agent rather than as a flag on `rerank`.
    #:
    #: Two reasons, both about the record rather than about taste. Reports
    #: are named `<config>.<agent>.json`, so a flag would have overwritten
    #: the `fetch=20` number with the `fetch=40` one and left no way to see
    #: the pair -- and the pair IS the experiment. And `fetch` is not in
    #: `config_verbatim`, so with a flag the surviving file would not say
    #: which setting produced it; the agent key does say.
    #:
    #: What it buys: `rerank` fetches exactly `k`, which makes it a pure
    #: ordering experiment whose ceiling is `hybrid`'s recall@20 -- 0.46508
    #: on `qwen-rel-whole`, against which reranking returned 0.41948 mrr.
    #: That is efficient enough that the ceiling, not the ordering, is the
    #: binding constraint. Fetching 40 lets the model promote a gold answer
    #: from ranks 21-40, and equally lets it demote one off the end: both
    #: were seen in a 4-query probe. So this can lose, and a loss is a
    #: result about how far reranking can be trusted to reorder.
    "rerank40": lambda config: RerankAgent(k=config.k, fetch=40),
    #: `rerank40`'s accuracy at roughly half its cost -- or that is the
    #: hypothesis; it is a separate key because it may not be.
    #:
    #: Measured against the endpoint's own rates (1230 tok/s prefill, 60
    #: tok/s decode), a `rerank40` query is 23.3s of prefill and 12.7s of
    #: decode, summing to the 36.0s/query actually observed. `relation_cap`
    #: cuts the first by ~40% and `terse_scores` the second by ~45%,
    #: predicting ~19s.
    #:
    #: Both knobs change what the model sees or says, so this cannot share
    #: `rerank40`'s filename: the 0.46323 it is being compared against would
    #: be overwritten by its own comparison. Same reason `rerank40` is not a
    #: flag on `rerank`.
    "rerank40lean": lambda config: RerankAgent(
        k=config.k, fetch=40, relation_cap=10, terse_scores=True
    ),
    # Titles only: name and type, no body, no relations. The leanest thing
    # that is still a reranker. ~410 prompt tokens at fetch=40 against
    # `rerank40lean`'s measured 13,133 -- a 32x cut that moves the whole cost
    # of this architecture onto decode and retrieval.
    "rerank40title": lambda config: RerankAgent(
        k=config.k, fetch=40, terse_scores=True, pair_scores=True, passage_mode="title"
    ),
    # The same, plus one neighbour per relation type. Isolates whether the
    # relations signal survives being sampled down to a single name, which
    # `rerank40lean` (cap=10) cannot answer.
    "rerank40titlerel": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel",
    ),
    # The same, but the single kept neighbour per relation type is chosen by
    # BM25 against the query rather than by document order. At per_type=1
    # *which* name survives is the whole relations signal, so the selector
    # stops being a detail -- `titlerel` vs `titlerelranked` isolates it.
    "rerank40titlerelranked": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_ranked",
    ),
    # Twice the candidates, on the encoding that made them affordable.
    #
    # `fetch` is the ceiling on what reranking can fix -- it can only
    # reorder what retrieval found -- and the measured curve has not
    # flattened: recall@20 was 0.4651 at fetch=20 (identical to `hybrid`, by
    # construction: same 20 candidates, k=20, so only the order changes),
    # and 0.5369 at fetch=40 on full documents.
    #
    # Doubling used to mean doubling a 13,133-token prompt. On titles it is
    # ~350 extra prefill tokens, well under a second at aggregate rates.
    # Decode roughly doubles, which is the real bill.
    # The hybrid selector: neighbour names scored by embedding AND BM25
    # against the query, fused by reciprocal rank, in ONE batched call
    # across all 40 candidates. `...dense` and `...lexical` isolate the
    # channels -- the lexical one differs from `rerank40titlerelranked`
    # only in where idf is computed (across all candidates, not within one
    # document), which is itself worth a number.
    # Three orthogonal scores per candidate, averaged. Addresses a
    # measured defect: the model quantises hard onto a few integers (one
    # response used 5 nine times across 40 candidates), ties break on
    # retrieval order, and 10% of queries carry a run of >=10 candidates
    # ordered that way. It also matches the queries, which are conjunctive
    # -- "a drug that targets X and is indicated for Y" is two judgements.
    # Budget arms. FINDINGS 1b measured SELECTION as worth +0.083 mrr at
    # one name per relation type -- which says nothing about whether one is
    # the right number.
    #
    # The prediction is genuinely uncertain, which is what makes these worth
    # running. A second name is BY DEFINITION a worse match than the first,
    # and arbitrary names were measured 0.030 BELOW showing none at all. So
    # per_type=2 might gain (more evidence) or lose (the second name is
    # closer to noise than to signal). Same argument for showing more
    # relation types.
    "rerank40titlerel2ranked": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_ranked",
        relation_per_type=2,
    ),
    "rerank40titlerelwide": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_ranked",
        relation_max_types=16,
    ),
    "rerank40titlerelmatrix": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        matrix_scores=True,
        passage_mode="title_rel_ranked",
    ),
    "rerank40titlerelhybrid": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_hybrid",
    ),
    "rerank40titlereldense": lambda config: RerankAgent(
        k=config.k,
        fetch=40,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_dense",
    ),
    "rerank80titlerelranked": lambda config: RerankAgent(
        k=config.k,
        fetch=80,
        terse_scores=True,
        pair_scores=True,
        passage_mode="title_rel_ranked",
    ),
    # The only agent here that changes what RETRIEVAL sees rather than how
    # candidates are scored. Every `rerank*` arm above is bounded by what
    # one `hybrid` search found; this one issues several and fuses them, so
    # its recall@20 is not `hybrid`'s by construction. See
    # `agents/decompose.py` for the hypothesis and the prediction.
    "decompose": lambda config: DecomposeAgent(k=config.k),
    # Same machinery, whole-question paraphrases instead of constraint
    # pieces. Isolates phrasing mismatch from conjunction handling: every
    # search asks the complete question, so no hit can be tangential.
    "rephrase": lambda config: DecomposeAgent(k=config.k, rephrase=True),
    # Same as `rephrase`, but the model returns only the candidates it
    # believes rather than a full 20. A different instrument: it trades
    # positions 3-20 (which become retrieval order) for committing only
    # where it is confident. Whether that is better depends entirely on
    # whether hit@1 or MRR is the number you need.
    # The same three deep searches, truncated to 40 before the model sees
    # them. Isolates selection difficulty from pool reach: 3x80 ranking 20
    # of 80 gained mrr and LOST 0.023 recall@20 against 5x40 ranking 20 of
    # 40, and two variables moved at once. If recall returns to ~0.545 the
    # pool was never the problem.
    "rephrasenarrow": lambda config: DecomposeAgent(
        k=config.k, rephrase=True, fetch=40
    ),
    # Five searches AND eighty candidates. The two knobs measured
    # independent on 2026-08-21: pool size 80 -> 40 moved recall@20 by
    # 0.00003 and mrr by -0.014, while 3 -> 5 searches moved recall by
    # +0.023 and mrr by +0.003. So reach comes from search count and
    # ranking quality from pool size, and nothing had yet raised both.
    "rephrasewide": lambda config: DecomposeAgent(
        k=config.k, rephrase=True, sub_queries=4
    ),
    "rephraseshort": lambda config: DecomposeAgent(
        k=config.k, rephrase=True, rank_all=False
    ),
}


def build_agent(config: RunConfig) -> Agent:
    """The agent `config.agent` names, built for this config."""
    try:
        builder = AGENTS[config.agent]
    except KeyError:
        raise NotImplementedError(f"unknown agent {config.agent!r}") from None
    return builder(config)

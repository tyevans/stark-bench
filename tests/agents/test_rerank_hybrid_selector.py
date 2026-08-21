"""The hybrid selector: one batched `rank_texts` call for all candidates.

The relation *selector* is where the value is -- eight arbitrary neighbour
names score 0.030 BELOW showing none, eight BM25-ranked ones score 0.054
above (FINDINGS 1b). This scores them by embedding and BM25 together.

Two properties matter more than the fusion maths:

- **one call, not forty.** Per-candidate scoring would be forty round trips
  a query, and would compute each name's idf within a single document,
  where rarity is meaningless -- what makes a name distinctive is the OTHER
  candidates' names.
- **no silent fallback.** A scored mode rendered without scores would emit
  document order, which is the arm measured at 0.031 below titles-only. It
  would look like a result.
"""

from __future__ import annotations

import pytest

from stark_bench.agents.rerank import (
    PairRelevances,
    RerankAgent,
    relation_names,
    relations_by_score,
)
from stark_bench.domain import Passage, Query

DOC_A = (
    "- name: Alpha\n- type: pathway\n- relations:\n"
    "  interacts_with: {gene/protein: (RAC1, DCC, NTN1)}\n"
    "  member_of: {pathway: (Axon guidance)}\n"
)
DOC_B = (
    "- name: Beta\n- type: gene/protein\n- relations:\n"
    "  ppi: {gene/protein: (TP53, BRCA1)}\n"
)


class RankingToolset:
    """Records every `rank_texts` call so batching can be asserted."""

    def __init__(self, passages, judged, scores=None):
        self._passages = passages
        self._judged = judged
        self._scores = scores or {}
        self.rank_calls: list[tuple[str, list[str], str]] = []
        self.prompt = None

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def rank_texts(self, query, texts, *, mode="hybrid"):
        self.rank_calls.append((query, list(texts), mode))
        return [self._scores.get(t, 0.0) for t in texts]

    async def extract(self, prompt, schema):
        self.prompt = prompt
        return self._judged


def _passages():
    return [
        Passage(node_id="1", text=DOC_A, score=1.0),
        Passage(node_id="2", text=DOC_B, score=0.9),
    ]


async def _run(mode, scores=None):
    tools = RankingToolset(
        _passages(), PairRelevances(scores=[[1, 10], [2, 20]]), scores
    )
    agent = RerankAgent(k=2, fetch=2, pair_scores=True, passage_mode=mode)
    await agent.retrieve(Query(query_id=1, text="DCC and TP53"), tools)
    return tools


def test_relation_names_collects_every_name() -> None:
    assert relation_names(DOC_A) == ["RAC1", "DCC", "NTN1", "Axon guidance"]


def test_relation_names_is_empty_without_a_block() -> None:
    assert relation_names("- name: X\n") == []


async def test_one_call_covers_every_candidate() -> None:
    """Forty round trips a query is the thing this avoids."""
    tools = await _run("title_rel_hybrid")
    assert len(tools.rank_calls) == 1
    _, texts, _ = tools.rank_calls[0]
    assert {"RAC1", "DCC", "NTN1", "Axon guidance", "TP53", "BRCA1"} == set(texts)


async def test_the_batch_is_deduplicated() -> None:
    shared = Passage(node_id="3", text=DOC_A, score=0.5)
    tools = RankingToolset([*_passages(), shared], PairRelevances(scores=[[1, 1]]), {})
    agent = RerankAgent(k=3, fetch=3, pair_scores=True, passage_mode="title_rel_hybrid")
    await agent.retrieve(Query(query_id=1, text="q"), tools)
    _, texts, _ = tools.rank_calls[0]
    assert len(texts) == len(set(texts))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("title_rel_hybrid", "hybrid"),
        ("title_rel_dense", "dense"),
        ("title_rel_lexical", "lexical"),
    ],
)
async def test_the_mode_reaches_the_toolset(mode: str, expected: str) -> None:
    """Catches every scored mode silently requesting the same channel,
    which would make the channel-isolating arms measure one thing."""
    tools = await _run(mode)
    assert tools.rank_calls[0][2] == expected


async def test_the_scores_choose_the_rendered_name() -> None:
    tools = await _run("title_rel_hybrid", {"DCC": 9.0, "RAC1": 1.0, "NTN1": 0.5})
    assert "DCC" in tools.prompt
    assert "RAC1" not in tools.prompt


async def test_an_unscored_mode_makes_no_rank_call() -> None:
    """`title` and `title_rel_ranked` must not pay for a round trip."""
    tools = await _run("title")
    assert tools.rank_calls == []


async def test_rendering_a_scored_mode_without_scores_raises() -> None:
    """The silent failure: falling back to document order, which is the arm
    measured 0.031 BELOW titles-only and looks like a result."""
    agent = RerankAgent(passage_mode="title_rel_hybrid")
    with pytest.raises(ValueError, match="rank_texts"):
        agent._render_passage(DOC_A, "q", None)


def test_a_missing_score_sorts_last_rather_than_raising() -> None:
    out = relations_by_score(DOC_A, {"NTN1": 5.0})
    assert "NTN1" in out


def test_equal_scores_keep_document_order() -> None:
    """A channel that cannot separate the names must render what the
    unranked arm would, not an arbitrary reshuffle."""
    flat = dict.fromkeys(relation_names(DOC_A), 1.0)
    assert relations_by_score(DOC_A, flat) == relations_by_score(DOC_A, {})


async def test_no_relations_anywhere_skips_the_call() -> None:
    plain = [Passage(node_id="1", text="- name: X\n- type: t\n", score=1.0)]
    tools = RankingToolset(plain, PairRelevances(scores=[[1, 5]]), {})
    agent = RerankAgent(k=1, fetch=1, pair_scores=True, passage_mode="title_rel_hybrid")
    await agent.retrieve(Query(query_id=1, text="q"), tools)
    assert tools.rank_calls == []

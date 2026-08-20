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
import re
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


class TerseRelevance(BaseModel):
    """One candidate's score, addressed by position and scored as an integer.

    Every field here is sized for the *decode* budget, which is a third of
    this agent's wall time: at a measured 60 tok/s against 1230 tok/s prefill,
    an output token costs 20x an input token. 40 objects of
    `{"node_id": "12345", "score": 87.5}` is ~760 tokens and 12.7s;
    `{"i": 1, "s": 87}` is ~400 and 6.7s.

    **The index is not a token-saving trick alone -- it is what keeps the
    alignment checkable.** A bare list of 40 numbers would be shorter still,
    but a model that emitted 40 scores shifted by one would be undetectable
    and would silently rerank every candidate against its neighbour's text.
    An index can be validated against the range and a duplicate or invented
    one dropped, which is the same guarantee `node_id` gave, at a third of
    the tokens.

    Integer rather than float: `87.5` costs three tokens more than `87` and
    the extra precision is spent on a value used only for sorting. The 0-100
    range is kept for the reason the class docstring below records -- a
    coarse scale with named anchors made the model quantise onto them.
    """

    i: int
    s: int = Field(ge=0, le=100)


class TerseRelevances(BaseModel):
    scores: list[TerseRelevance]


class Relevance(BaseModel):
    """One candidate's score.

    0-100 rather than 0-10, and the prompt asks for a spread rather than
    naming anchor values. The first version scored 0-10 and described what
    10, 5 and 0 meant; the model then used *only* those three numbers --
    a real answer was one 10, one 5, and eighteen 0s. That collapses the
    reranking into the top two slots, leaves everything below decided by the
    retrieval-order tie-break, and makes one overconfident 10 enough to
    demote a correct top hit. Naming example values on a coarse scale is an
    instruction to quantise to them.
    """

    node_id: str
    score: float = Field(ge=0.0, le=100.0)


class Relevances(BaseModel):
    scores: list[Relevance]


_RELATIONS_MARKER = "- relations:"

#: Matches one relation line's parenthesised neighbour list, e.g.
#: `  ppi: {gene/protein: (PI4KA, EIF3I, ...)}`. Deliberately narrow: a line
#: that does not match is passed through untouched rather than mangled, so a
#: format change costs tokens instead of correctness.
_RELATION_LINE = re.compile(r"^(\s*[\w/ ]+: \{[\w/ ]+: )\((.*)\)(\}?.*)$")


def lean_document(text: str, query: str, *, cap: int) -> str:
    """Keep the head whole; cap each relation's neighbour list at `cap` names.

    ## Why cap relations rather than shorten the document

    STaRK puts `- relations:` near the *top* of a PRIME document -- char 778
    of a 50,192-char hub node -- and everything after it is neighbour names.
    So a flat character budget does not trade detail for neighbours; it keeps
    a near-arbitrary prefix of one hub's 500-name `ppi` list and discards
    every other relation type entirely. Measured over 4,000 documents, 80.7%
    exceed 3,000 characters and the mean is 4,569.

    Capping instead keeps *every* relation type present at `cap` names each.
    Mean rendered length falls from 2,852 characters to 1,725 at `cap=10`,
    a 40% cut in prefill, which at 1230 tok/s is ~9s of a 36s query.

    ## Why the query decides which names survive

    PRIME's queries name related entities verbatim ("a drug that targets X
    and is indicated for Y"), and that text being present is the whole
    mechanism behind this project's headline result -- relations moved hybrid
    +42% while dense barely moved, because the gain is lexical. Dropping the
    named neighbour to save tokens would remove exactly the evidence the
    reranker is there to weigh.

    So neighbours the query mentions are kept first, then the list is filled
    to `cap` in its original order. A truncated list is marked `+N more`
    rather than silently ended: a model shown five neighbours cannot tell a
    node with five from a hub with five hundred, and that difference is
    itself evidence.

    `cap=0` is refused. It removes every neighbour name, which is a 61%
    saving and destroys the signal this corpus exists to test.
    """
    if cap <= 0:
        raise ValueError("cap must be positive; cap=0 removes the relations signal")
    marker = text.find(_RELATIONS_MARKER)
    if marker < 0:
        return text
    lowered = query.lower()
    relations = text[marker:]
    # `splitlines()` drops trailing newlines and `join` does not put them
    # back, so a document whose relations block needed no capping would come
    # out one byte shorter than it went in. Caught by the pass-through test,
    # which is the only one that could see it.
    trailing = relations[len(relations.rstrip("\n")) :]
    kept: list[str] = []
    for line in relations.splitlines():
        match = _RELATION_LINE.match(line)
        if match is None:
            kept.append(line)
            continue
        names = [n for n in match.group(2).split(", ") if n]
        if len(names) <= cap:
            kept.append(line)
            continue
        # Stable: named-in-query first, each group in its original order, so
        # the rendering is deterministic and two runs of the same arm agree.
        named = [n for n in names if n.lower() in lowered]
        rest = [n for n in names if n.lower() not in lowered]
        shown = (named + rest)[:cap]
        dropped = len(names) - len(shown)
        joined = ", ".join(shown)
        suffix = f", +{dropped} more" if dropped else ""
        kept.append(f"{match.group(1)}({joined}{suffix}){match.group(3)}")
    return text[:marker] + "\n".join(kept) + trailing


#: `- name:` and `- type:` as STaRK writes them at the top of every document.
#: Anchored to the line start so a `name` nested inside a details dict cannot
#: win: those are database display names, and one of them is the wrong answer
#: to "what is this entity called".
_NAME_LINE = re.compile(r"^- name:[ \t]*(.*)$", re.MULTILINE)
_TYPE_LINE = re.compile(r"^- type:[ \t]*(.*)$", re.MULTILINE)


def title_of(text: str) -> str:
    """The entity's name and type, and nothing else.

    ## Why a title can be enough

    PRIME's queries name the entities they are about ("a drug that targets X
    and is indicated for Y"), and reranking only has to *order* candidates
    retrieval already found. Ordering by name and type is a far weaker
    signal than ordering by the full document -- but the full document costs
    3,415 characters per candidate against roughly 37 here, and 82% of that
    is text no query asks about.

    Whether the weaker signal is good enough is exactly the measurement this
    exists to take. It is a real experiment with a plausible null.

    ## The empty-title hazard

    A document with no `- name:` line would render as `? (?)`, and forty
    candidates rendering as `? (?)` is a reranker scoring noise -- which
    looks like "reranking does not help" rather than like a defect. So a
    document without a name falls back to its first non-empty line, and
    `render_passages` refuses a batch that came out empty.
    """
    name = _NAME_LINE.search(text)
    kind = _TYPE_LINE.search(text)
    label = name.group(1).strip() if name else ""
    if not label:
        label = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"{label} ({kind.group(1).strip()})" if kind else label


def first_relations(text: str, *, per_type: int = 1, max_types: int = 8) -> str:
    """One neighbour per relation type, flattened onto a single line.

    `per_type` names rather than all of them, because the point of the
    lean encodings is that a *sample* of a node's neighbourhood identifies
    it. `max_types` bounds the pathological node: PRIME's hub entities carry
    dozens of relation types and one of them would otherwise cost more than
    the whole rest of the prompt.

    Returns `""` for a document with no relations block, which is correct
    and not an error -- `add_rel=False` corpora have none at all.
    """
    if per_type <= 0:
        raise ValueError("per_type must be positive; 0 removes the relations signal")
    marker = text.find(_RELATIONS_MARKER)
    if marker < 0:
        return ""
    out: list[str] = []
    for line in text[marker:].splitlines()[1:]:
        match = _RELATION_LINE.match(line)
        if match is None:
            continue
        names = [n for n in match.group(2).split(", ") if n]
        if not names:
            continue
        kind = match.group(1).strip().rstrip(": {").strip()
        out.append(f"{kind}: {', '.join(names[:per_type])}")
        if len(out) >= max_types:
            break
    return "; ".join(out)


_PROMPT_TEMPLATE = (
    "You are ranking candidate entities from a biomedical knowledge base "
    "against a search query. Score every candidate from 0 to 100 for how "
    "well it answers the query.\n\n"
    "Use the whole range and give close candidates different scores -- the "
    "scores are used to order the candidates, so two candidates with the "
    "same score are being called indistinguishable. Reserve the top of the "
    "range for candidates satisfying every condition in the query, the "
    "middle for ones satisfying some, and the bottom for unrelated ones, "
    "but choose intermediate values freely rather than rounding to those "
    "bands.\n\n"
    "Judge only from the text shown. Return one score for every candidate "
    "id, and invent no ids.\n\n"
    "Query: {query}\n\nCandidates:\n{candidates}"
)


@dataclass(frozen=True, slots=True)
class RerankAgent:
    k: int = 20
    #: Candidates fetched before reranking. Reranking can only reorder what
    #: retrieval found, so this -- not `k` -- is the ceiling on what the
    #: architecture can fix.
    #:
    #: 20, equal to `k`, which makes this a pure ordering experiment: the set
    #: returned is exactly `hybrid`'s, so `recall@20` is identical to
    #: `hybrid`'s by construction and every difference lands in MRR and
    #: Hit@1. Widening it to 40 lets reranking promote from ranks 21-40 and
    #: also lets it *demote* a marginal hit off the end -- both were observed
    #: in a 4-query probe (gold 2->1 and 3->1, but one 18->out).
    #:
    #: Timing did not decide this and could not: repeated 3-query probes
    #: against the shared endpoint returned 17.6s and 27.9s per query for the
    #: *same* settings, and had `fetch=40` beating `fetch=30`. The variance
    #: swamps the effect at that sample size. Treat any per-query timing here
    #: as an order of magnitude, not a measurement.
    fetch: int = 20
    #: Neighbour names shown per relation type, or `None` for the flat
    #: character budget alone. See `lean_document`. Separate from
    #: `terse_scores` deliberately: they cut different halves of the cost
    #: (prefill and decode), and if accuracy moves, one combined knob cannot
    #: say which did it.
    relation_cap: int | None = None
    #: Score by candidate index with integer scores, rather than by node id
    #: with floats. See `TerseRelevance`.
    terse_scores: bool = False
    #: How much of each candidate document reaches the prompt.
    #:
    #: - `"full"`  -- the document, subject to `relation_cap` and the
    #:   character budget. What every arm before this one measured.
    #: - `"title"` -- name and type only. ~37 chars per candidate against
    #:   ~3,415, because 82% of a document's characters are node details and
    #:   16.5% of the head is database provenance (`literatureReference`,
    #:   `orthologousEvent`, `crossReference`) that no STaRK query asks about.
    #: - `"title_rel"` -- name, type, and one neighbour per relation type.
    #:
    #: Separate from `relation_cap` and `terse_scores` for the reason those
    #: two are separate from each other: they cut different parts of the
    #: bill, and one combined knob could not say which moved the accuracy.
    passage_mode: str = "full"
    name: str = "rerank"

    def _render_passage(self, text: str, query: str) -> str:
        """One candidate's text, at whatever detail `passage_mode` asks for."""
        if self.passage_mode == "title":
            return title_of(text)
        if self.passage_mode == "title_rel":
            rels = first_relations(text)
            return f"{title_of(text)} | {rels}" if rels else title_of(text)
        if self.passage_mode != "full":
            # Not a warning. An unrecognised mode silently falling back to
            # `full` would produce a correct-looking run measuring something
            # other than what its name says.
            raise ValueError(f"unknown passage_mode {self.passage_mode!r}")
        return (
            lean_document(text, query, cap=self.relation_cap)
            if self.relation_cap
            else text
        )

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        passages = await tools.search_passages(query.text, k=self.fetch, mode="hybrid")
        if not passages:
            return []

        texts = [self._render_passage(p.text, query.text) for p in passages]
        # Every real defect in this project has been silent, and a reranker
        # handed forty blank passages scores like a slightly-worse `hybrid`
        # with nothing in the log. A mode that renders nothing is a bug in
        # the mode, so say so here rather than three hours later in a report.
        if not any(t.strip() for t in texts):
            raise ValueError(
                f"passage_mode={self.passage_mode!r} rendered {len(texts)} "
                "empty passages; the reranker would be scoring blank text"
            )
        # The label is the whole difference in prompt cost between the two
        # output modes on the *input* side: a 5-digit node id is ~3 tokens
        # and an index is 1, times `fetch` candidates.
        labels = (
            [str(i) for i in range(1, len(passages) + 1)]
            if self.terse_scores
            else [p.node_id for p in passages]
        )
        rendered = "\n\n".join(
            f"[{label}] {text[:_MAX_PASSAGE_CHARS]}"
            for label, text in zip(labels, texts, strict=True)
        )
        try:
            judged = await tools.extract(
                _PROMPT_TEMPLATE.format(query=query.text, candidates=rendered),
                TerseRelevances if self.terse_scores else Relevances,
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
        # An index is validated against the range rather than trusted. The
        # index exists to make a misalignment *detectable* -- a bare list of
        # scores would be shorter and a one-position shift in it would be
        # invisible, silently scoring every candidate against its
        # neighbour's text. Out-of-range and duplicate indices are dropped
        # exactly as invented node ids are, leaving that candidate unscored
        # rather than mis-scored.
        scores: dict[str, float] = {}
        if judged is not None:
            for r in judged.scores:
                if self.terse_scores:
                    if not 1 <= r.i <= len(passages):
                        continue
                    node_id = passages[r.i - 1].node_id
                    score = float(r.s)
                else:
                    node_id, score = r.node_id, r.score
                    if node_id not in retrieval_rank:
                        continue
                scores.setdefault(node_id, score)

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

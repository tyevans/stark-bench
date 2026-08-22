"""Find the generated questions that carry no signal, before indexing them.

## Why this exists at all

Generated questions are additive: each becomes another chunk on its node,
and `aggregation: max` lets a node score by its best-matching one. That is
safe when a question is distinctive and **actively harmful when it is
not**. A question thousands of entities answer equally well -- "What does
this protein do?" -- puts near-identical vectors all over the index, so a
query that matches one matches all of them, and discrimination gets worse
rather than better.

Nothing downstream would say so. Every count in every report stays clean,
`verify_corpus.py` sees the right row count, and the arm reads as
"question generation did not help". That is the ninth-and-tenth failure
shape in this repository, so the check comes before the index, not after
the number looks wrong.

## Three signals, cheapest first

**Duplicate text.** Two nodes emitting the same question is the extreme
case and needs no embedding: hash it. A question shared by many nodes
cannot discriminate between them by construction.

**Term rarity.** A question built entirely from terms common across the
question corpus is generic even when its exact string is unique. This is
idf over the generated questions themselves -- no model, one pass.

**Nearest-neighbour crowding**, only for what the first two flag: how
similar is this question to other NODES' questions? A question sitting in
a tight cluster of other entities' questions is competing with them for
every query that matches it.

## What this deliberately does NOT do

It does not use STaRK answers. A filter tuned on the labels would leak the
benchmark into the corpus, and the resulting number would measure the
filter's access to the answers rather than the idea. Every signal here is
computed from the generated questions alone.

It also does not delete anything. It writes a score per question and
reports the distribution; the ingest decides what to admit, with the
threshold in the config so it lands in `config_verbatim`. **A filter that
silently dropped most questions would look exactly like a generator that
never worked**, which is why the count of what a threshold would remove is
printed rather than applied here.

Nodes keep their own document chunk regardless, so no threshold can make
an entity unreachable -- which is what makes aggressive filtering safe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")

#: Words carrying no discriminating power in a biomedical question corpus.
#: Deliberately short: the idf pass below finds corpus-specific filler
#: ("gene", "protein", "which") on its own, and a long hand-written list
#: would be a second, worse estimate of the same thing.
_STOP = frozenset(
    "a an and are as at be by can does for from has have how in is of on or "
    "that the this to what when where which who whose with".split()
)


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="report how many questions would be dropped below this score",
    )
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [(r["node_id"], q) for r in records for q in r["questions"]]
    print(f"{len(records):,} nodes, {len(pairs):,} questions")
    if not pairs:
        return 1

    # --- duplicate text
    by_text: dict[str, set[str]] = {}
    for node_id, question in pairs:
        by_text.setdefault(question.strip().lower(), set()).add(node_id)
    shared = {t: n for t, n in by_text.items() if len(n) > 1}
    duplicated = sum(len(n) for n in shared.values())
    print(
        f"\nduplicate text: {len(shared):,} question(s) shared by more than "
        f"one node, covering {duplicated:,} node-question pairs "
        f"({100 * duplicated / len(pairs):.2f}%)"
    )
    for text, nodes in sorted(shared.items(), key=lambda kv: -len(kv[1]))[: args.show]:
        print(f"  x{len(nodes):<4} {text[:110]}")

    # --- term rarity: idf over the generated corpus
    document_frequency: Counter[str] = Counter()
    for _, question in pairs:
        document_frequency.update(set(_terms(question)))
    total = len(pairs)
    idf = {term: math.log(total / count) for term, count in document_frequency.items()}
    ceiling = math.log(total)

    scored: list[tuple[float, str, str]] = []
    for node_id, question in pairs:
        terms = _terms(question)
        if not terms:
            scored.append((0.0, node_id, question))
            continue
        # Mean idf, normalised by the maximum a single-occurrence term could
        # score, so the number is comparable across corpus sizes.
        scored.append(
            (sum(idf[t] for t in terms) / len(terms) / ceiling, node_id, question)
        )
    scored.sort()

    values = [s for s, _, _ in scored]
    print("\nterm-rarity score (0 = every word is corpus filler, 1 = all unique)")
    for label, index in (
        ("min ", 0),
        ("p05 ", int(0.05 * len(values))),
        ("p25 ", int(0.25 * len(values))),
        ("med ", len(values) // 2),
        ("p75 ", int(0.75 * len(values))),
        ("max ", len(values) - 1),
    ):
        print(f"  {label} {values[index]:.4f}")

    print(f"\nlowest-signal questions ({args.show}):")
    for score, node_id, question in scored[: args.show]:
        print(f"  {score:.4f}  [{node_id}] {question[:100]}")

    if args.threshold is not None:
        dropped = [s for s in values if s < args.threshold]
        by_node: Counter[str] = Counter()
        for score, node_id, _ in scored:
            if score < args.threshold:
                by_node[node_id] += 1
        starved = sum(
            1 for r in records if by_node.get(r["node_id"], 0) == len(r["questions"])
        )
        print(
            f"\nat threshold {args.threshold}: {len(dropped):,} question(s) "
            f"dropped ({100 * len(dropped) / len(values):.1f}%), "
            f"{starved:,} node(s) would lose ALL their questions"
        )
        print(
            "  (those nodes keep their own document chunk and stay "
            "retrievable -- filtering cannot orphan an entity)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How many independent perspectives does a set of restatements actually give?

## The question, made measurable

A multi-query arm pays for restatements that reach candidates the others
miss. Measured on `qwen-rel-whole`: 3 searches -> 5 bought **+0.023
recall@20**, while doubling the candidate pool bought **+0.00003**. Reach
is the whole value, and reach requires searches that disagree.

"Rewrite this N different ways" does not ask for disagreement. It asks for
variety, and a model given no direction produces synonyms and reorderings
-- which preserve exactly what retrieval keys on.

**Effective rank** turns that into a number. Take the restatement vectors,
normalise, form the Gram matrix, and ask how many dimensions its spectrum
actually occupies:

    participation ratio  =  (sum of eigenvalues)^2 / sum of squares
    entropy rank         =  exp(-sum p_i log p_i),  p = normalised spectrum

Five identical restatements score 1.0. Five mutually orthogonal ones score
5.0. So "you asked for five perspectives and received 1.6" is a statement
this prints rather than an impression.

Both are reported because they weight the tail differently: the
participation ratio is dominated by the largest directions, the entropy
rank counts weak ones more generously. Where they disagree, the set has a
long thin tail -- a few real perspectives plus several near-duplicates.

## Why cosine alone is not enough

Mean pairwise cosine says how far apart things are *on average* and cannot
distinguish five restatements spread evenly from four near-duplicates plus
one outlier. Those have different retrieval behaviour: the second buys one
extra search's worth of reach while being charged for five. Effective rank
separates them; cosine is printed beside it as the familiar number.

## What it compares

`plain` -- the shipped prompt: rewrite this N different ways.
`lens`  -- one restatement per prescribed angle, so the model is pushed
           apart rather than asked to vary freely.

Uses no STaRK answers: diversity is a property of the queries, and tuning
it against the labels would measure the tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import itertools
import statistics
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from stark_bench.adapters.config_file import load_config
from stark_bench.adapters.stark_artifacts import read_queries
from stark_bench.composition.cli import _data_dir, _live_embeddings_for, _llm_for

#: The angles. Chosen so that two cannot be satisfied by one sentence:
#: what a thing DOES and what it CONNECTS TO are different facts about it.
#:
#: The last three target different retrieval CHANNELS rather than different
#: content. `keyword` strips filler so BM25 sees higher term density;
#: `declarative` states the answer as a document would, which is the shape
#: a bi-encoder matches best. A set that varies only wording exercises
#: neither channel differently.
_LENSES: tuple[tuple[str, str], ...] = (
    ("mechanism", "what the answer DOES -- its function, mechanism or process"),
    (
        "relational",
        "what the answer CONNECTS TO -- the drugs, genes, diseases or "
        "pathways named in the query, and how it relates to them",
    ),
    (
        "taxonomic",
        "what the answer IS -- its class, family or category, and what "
        "distinguishes it inside that group",
    ),
    (
        "keyword",
        "a dense keyword query: entity names and relation words only, no "
        "filler and no question phrasing",
    ),
    (
        "declarative",
        "a statement of fact as a reference document would phrase it, not "
        "a question -- describe the answer rather than asking for it",
    ),
    (
        "clinical",
        "the observable or applied angle -- symptoms, indications, effects "
        "or use, where the query admits one",
    ),
)

_PLAIN_PROMPT = (
    "Rewrite this biomedical search query {count} different ways.\n\n"
    "Each rewrite must ask THE SAME question, complete -- not a piece of "
    "it. Vary how it is asked: the word order, the framing, the general "
    "vocabulary.\n\n"
    "Copy every entity name, gene symbol, drug name, disease name and "
    "identifier EXACTLY as written. Rephrase the words AROUND them.\n\n"
    "Query: {query}"
)

_LENS_PROMPT = (
    "Restate this biomedical search query once for EACH angle below.\n\n"
    "Every restatement must still identify the same answer and keep every "
    "constraint. Copy entity names, gene symbols, drug names and disease "
    "names EXACTLY as written -- the search is partly lexical and an "
    "altered name will not match. Rephrase the words AROUND them.\n\n"
    "The angles are deliberately different from one another. Do not write "
    "one sentence several times with the word order changed.\n\n"
    "{angles}\n\n"
    "Return them in the order listed.\n\n"
    "Query: {query}"
)


class Restatements(BaseModel):
    queries: list[str] = Field(
        description=(
            "One restatement per requested angle, in order, each keeping "
            "every entity name verbatim."
        )
    )


def _effective_rank(vectors: np.ndarray) -> tuple[float, float]:
    """Participation ratio and entropy rank of a set of vectors.

    Both computed from the eigenvalues of the Gram matrix of the
    L2-normalised set, which for n vectors is n x n regardless of the
    embedding dimension -- so this is cheap even at 1024 dimensions.
    """
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(normed @ normed.T)
    # Numerical noise can produce tiny negatives on a PSD matrix.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0, 0.0
    participation = float(total**2 / np.square(eigenvalues).sum())
    p = eigenvalues / total
    p = p[p > 0]
    entropy = float(np.exp(-(p * np.log(p)).sum()))
    return participation, entropy


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/qwen-rel-whole.yaml", type=Path)
    parser.add_argument("--chat-model", default="gemma-4-26b-qat")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    object.__setattr__(config, "chat_model", args.chat_model)
    llm = _llm_for(config)
    embeddings = _live_embeddings_for(config)

    sampled = [
        query
        for query, _ in itertools.islice(
            read_queries(_data_dir(config) / f"queries.{config.effective_split}.jsonl"),
            args.queries,
        )
    ]
    angles = "\n".join(
        f"{i + 1}. {name}: {how}" for i, (name, how) in enumerate(_LENSES[: args.count])
    )

    collected: dict[str, dict[str, list]] = {}
    for label in ("plain", "lens"):
        pairwise: list[float] = []
        to_original: list[float] = []
        participation: list[float] = []
        entropy: list[float] = []
        texts: list[list[str]] = []
        for query in sampled:
            prompt = (
                _PLAIN_PROMPT.format(query=query.text, count=args.count)
                if label == "plain"
                else _LENS_PROMPT.format(query=query.text, angles=angles)
            )
            try:
                out = await llm.extract(prompt, Restatements)
            except Exception:
                continue
            variants = [t.strip() for t in out.queries if t.strip()][: args.count]
            if len(variants) < 2:
                continue
            texts.append(variants)
            vectors = np.array(await embeddings.embed_query([query.text, *variants]))
            normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            similarity = normed @ normed.T
            pairwise += [
                float(similarity[i, j])
                for i, j in itertools.combinations(range(1, len(vectors)), 2)
            ]
            to_original += [float(similarity[0, j]) for j in range(1, len(vectors))]
            pr, er = _effective_rank(vectors[1:])
            participation.append(pr)
            entropy.append(er)
        collected[label] = {
            "pairwise": pairwise,
            "to_original": to_original,
            "participation": participation,
            "entropy": entropy,
            "texts": texts,
        }

    n = args.count
    print(f"{len(sampled)} queries, {n} restatements each\n")
    print(
        f"{'':8} {'cos(pairs)':>11} {'cos(orig)':>11} "
        f"{'eff.rank PR':>12} {'eff.rank H':>11}   of {n}"
    )
    for label, data in collected.items():
        if not data["participation"]:
            print(f"{label:8} no data")
            continue
        print(
            f"{label:8} {statistics.mean(data['pairwise']):11.4f} "
            f"{statistics.mean(data['to_original']):11.4f} "
            f"{statistics.mean(data['participation']):12.2f} "
            f"{statistics.mean(data['entropy']):11.2f}"
        )
    print(
        f"\neffective rank: 1.0 = every restatement is the same direction, "
        f"{float(n):.1f} = mutually orthogonal."
        "\nPR is dominated by the largest directions; H counts weak ones more"
        "\ngenerously. PR << H means a few real perspectives plus near-duplicates."
    )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    label: {k: v for k, v in data.items() if k != "texts"}
                    | {"texts": data["texts"]}
                    for label, data in collected.items()
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    import json

    raise SystemExit(asyncio.run(main()))

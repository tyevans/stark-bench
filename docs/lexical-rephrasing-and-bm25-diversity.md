# Query rephrasing buys no semantic diversity — and that is not the problem

**Multi-query retrieval works on this benchmark. It does not work for the
reason it is usually described as working.**

The standard account of rewriting a query several times and fusing the
results is that each rewrite is a different *angle* on the question, so
each reaches part of the answer space the others miss. Measured here, the
rewrites are not different angles: four restatements of a query occupy
**1.1 independent directions** in embedding space out of a possible four.

They still pay. `rephrase` — three searches, union, rerank — reaches
**recall@20 0.545** against a single hybrid search's 0.468 on STaRK-PRIME.
The reach is real. It just comes from somewhere other than where the
explanation says.

---

## Setup

- **Corpus**: STaRK-PRIME, 129,375 entities, documents carrying a
  `- relations:` block naming each node's neighbours.
- **Queries**: `test-0.1`, natural-language and fully specified — *"Which
  gene or protein is engaged in DCC-mediated attractive signaling, can
  bind to actin filaments, and belongs to the actin-binding LIM protein
  family?"*
- **Restatements**: `gemma-4-26b-qat`, non-reasoning, instructed to keep
  every entity name verbatim.
- **Embeddings**: `qwen3-embedding-0.6b` (Q8_0 GGUF), 1024 dimensions,
  with the query-side instruction prefix the retrieval arms use.
- **Retrieval**: pgvector HNSW at `ef_search = 800` for the dense channel,
  BM25 over a terms table for the lexical channel.
- Ten queries, four restatements each. Small — see *Limits* below.

## Measuring diversity: effective rank

Mean pairwise cosine is the usual number and it cannot distinguish four
evenly-spread restatements from three near-duplicates plus one outlier.
Those behave differently in retrieval, so the measure has to see the
difference.

**Effective rank** does. Normalise the restatement vectors, form the Gram
matrix, and ask how many dimensions its eigenvalue spectrum occupies:

```
participation ratio  =  (Σλ)² / Σλ²
entropy rank         =  exp(−Σ pᵢ log pᵢ),   p = λ / Σλ
```

Four identical restatements score **1.0**. Four mutually orthogonal ones
score **4.0**.

| prompt | cos(pairs) | cos(to original) | eff. rank (PR) | eff. rank (H) |
|---|---|---|---|---|
| "rewrite this 4 different ways" | 0.9517 | 0.9593 | **1.08** | 1.21 |
| six prescribed angles | 0.9188 | 0.9202 | **1.13** | 1.34 |

The second row is the interesting one. The angles were chosen so that two
could not be satisfied by one sentence — what the answer **does**
(mechanism), what it **connects to** (relational), what it **is**
(taxonomic) — plus three that target retrieval channels rather than
content: a bag-of-keywords form, a declarative document-style statement,
and a clinical framing.

Prescribing all six moved effective rank by **0.05**.

Even the structural outlier does not escape. This restatement —

> `DCC-mediated attractive signaling actin filaments actin-binding LIM protein family`

— sits at **0.9146** mean cosine to its three prose siblings, against the
mechanism angle's 0.9159. Removing it changes effective rank from 1.13/4
to 1.11/3: proportionally nothing.

## Why the angles collapse

**This is the embedder working correctly.** A query encoder *should* map
faithful paraphrases of one question to one point — that invariance is the
training objective. Asking for semantic diversity through restatement is
asking the model to undo what it was built to do.

A second, corpus-specific effect compounds it. **STaRK queries are already
fully specified.** The example above names its mechanism, its relations
*and* its taxonomy in one sentence, so "restate from the mechanism angle"
has nothing left to select — every angle resolves to the same content with
different word order. On a corpus of terse or underspecified queries the
angles might have more to grip.

## Where the reach actually comes from

The same restatements, retrieved per channel, measured as Jaccard overlap
between the node sets each returns:

| channel | Jaccard between restatements | distinct nodes reached, vs one search |
|---|---|---|
| dense (HNSW) | 0.5934 | 1.56× |
| **lexical (BM25)** | **0.3174** | **2.31×** |
| hybrid | 0.4189 | 1.92× |

Vectors at cosine 0.92 return **59% the same nodes**. BM25 returns **32%
the same**, and the union of four restatements reaches **2.31× as many
distinct nodes** as any one of them.

So multi-query reach on this corpus is a lexical phenomenon. Restatements
that a bi-encoder considers near-identical are, to a term-matching
retriever, meaningfully different queries — because they differ in exactly
what BM25 keys on: which words are present, how often, and how rare.

This lands on top of the same corpus's headline result. Adding relational
text to documents moved the lexical channel **+22%** and the dense channel
**+2%**. Restatement widens through the lexical channel too. Same corpus,
same mechanism, opposite ends of the pipeline.

## What follows

**Design restatements for surface-form variation, not for perspective.**
Synonyms, term density, word forms, keyword-versus-prose. The angles are
unavailable; the terms are not.

**Do not expect a bigger candidate pool to substitute for more searches.**
Measured separately on the same arm: going from 3 searches to 5 moved
recall@20 by **+0.023**, while doubling the candidate pool from 40 to 80
moved it by **+0.00003**. Reach comes from searches that disagree, and
ranking quality comes from pool size — they are independent knobs.

**Genuine dense diversity requires leaving query space.** Two routes, both
untested here:

- A **hypothetical document** (HyDE) lives where the corpus lives rather
  than where the question does, so it is not bound by query-side
  invariance.
- **Graph expansion** reaches candidates by traversal rather than by text
  at all.

Both bypass the invariance that blocks restatement, and on this corpus the
dense channel has never moved — 0.18 to 0.25 mrr across every arm, model
and encoding tried.

## Limits

- **Ten queries, four restatements.** Enough for a 0.92-cosine, 1.1-rank
  effect; not enough to characterise the distribution or to rank the six
  angles against each other.
- **One embedder**, and a Q8_0 GGUF of a 0.6B model. A larger or less
  invariant encoder might spread restatements further — though "less
  invariant" is not obviously a better encoder.
- **One corpus**, whose queries are unusually complete. The angle collapse
  is partly a property of STaRK-PRIME.
- The retrieval overlap is measured over eight queries at k=20, on an
  indexed store at `ef_search = 800`.

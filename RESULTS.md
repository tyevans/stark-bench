# Results

**Seven of twelve retrieval numbers are in** (2026-08-19). The table is
generated — run `uv run python scripts/results_table.py` and paste.
Everything else on this page is the framework for reading it, written
before the numbers existed so that it is a prediction rather than a
rationalisation.

## Findings so far

| corpus | chunks/node | agent | mrr | hit@1 | hit@5 | recall@20 |
|---|---|---|---|---|---|---|
| vss-control (ada-002) | 1.000 | dense | 0.23057 | 0.1536 | 0.3107 | 0.37878 |
| vss-control (ada-002) | 1.000 | hybrid | 0.23111 | 0.1643 | 0.3214 | 0.37096 |
| native-wholedoc | 1.057 | dense | 0.21635 | 0.1357 | 0.3036 | 0.37780 |
| native-wholedoc | 1.057 | hybrid | 0.21872 | 0.1500 | 0.2964 | 0.36799 |
| redstring-native | 1.139 | dense | 0.18446 | 0.1214 | 0.2500 | 0.32397 |
| redstring-native | 1.139 | hybrid | 0.19854 | 0.1357 | 0.2643 | 0.35162 |
| native-wholedoc | 1.057 | **rerank** | **0.34075** | **0.2857** | **0.4000** | 0.36799 |

All seven on complete corpora: 280 queries, 129,375 nodes, both native arms
carrying the full 8,100,498 edges.

`rerank` is the top row by a wide margin and is a *reordering* of the
`native-wholedoc` hybrid row directly above it -- same corpus, same
candidates, one LLM call per query. See finding 4. For scale, the best
published entry on this split is GPT-4 reranking at 0.2655 mrr, though at a
different protocol; the comparison is drawn honestly in finding 4 and is not
a claim to have beaten it.

### 1. Chunk granularity is the largest effect, and finer is worse

`native-wholedoc` → `redstring-native` holds the model, the prefixes, the
table and the graph fixed and varies only the chunker. Dense retrieval
loses **15% of mrr and 14% of recall@20** going from 1.057 to 1.139
chunks/node.

Both move in the same direction, which rules out a precision/recall
trade-off and points at retrieval finding less. A mechanism that fits, and
that `native-sliding1k` at 1.94 chunks/node will test: STaRK scores
**nodes**, `aggregation: max` takes each node's best chunk, so more chunks
per node is more draws from the score distribution. A distractor needs one
spuriously high chunk to outrank the right answer, and finer chunking hands
every distractor more tickets. That predicts monotone degradation.

### 2. The embedding model costs ranking, not recall

`vss-control` → `native-wholedoc` is the model swap with chunking held
near-constant. Nemotron-3-Embed-1B loses **6% of mrr** to ada-002 while
**recall@20 is unchanged** (0.37780 against 0.37878, a quarter of one
query).

It retrieves the same documents and orders them worse. That is a reranking
problem rather than a retrieval one, and it is the shape a cross-encoder
fixes and a bigger bi-encoder may not. Note also that Nemotron is running
at Q4_K_M against ada-002 at full precision, and *with* its task prefixes —
this is the model at its best-configured, not a strawman.

### 3. Lexical fusion earns its keep only on finer chunks

| corpus | chunks/node | Δmrr from fusion | Δrecall@20 |
|---|---|---|---|
| vss-control | 1.000 | +0.0005 | −0.0078 |
| native-wholedoc | 1.057 | +0.0024 | −0.0098 |
| redstring-native | 1.139 | **+0.0141** | **+0.0277** |

On whole-document corpora BM25 fusion is worth nothing and costs recall. At
1.139 chunks/node it is worth **+7.6% mrr and +8.6% recall@20** — and is
the only intervention measured so far that improves both at once.

The two effects interact rather than adding: finer chunking hurts dense
badly, fusion recovers part of it, and the recovery is not enough.
`redstring-native`'s best (0.19854) still trails `native-wholedoc`'s worst
(0.21635).

**This is BM25, not the graph.** `dense` and `hybrid` both call
`search_chunks` and neither reads a relationship; `hybrid` is redstring's
rank fusion of the vector and lexical channels. Three findings were
initially reported here as graph results on the strength of the agent's
*name*, while the "What each comparison isolates" section below said
`dense` → `hybrid` was the lexical channel all along. `--agent lexical`
now exists so the third column is measured rather than subtracted.

### 4. Showing the LLM the document is worth more than everything else combined

`rerank` reorders `hybrid`'s top 20 with one listwise LLM call, on the same
corpus and the same candidate set:

| metric | hybrid | rerank | Δ |
|---|---|---|---|
| mrr | 0.21872 | **0.34075** | +0.12203 |
| hit@1 | 0.15000 | **0.28571** | +0.13571 |
| hit@5 | 0.29643 | **0.40000** | +0.10357 |
| recall@20 | 0.36799 | 0.36799 | ±0.00000 |

**recall@20 identical to five decimals is the control**: recall@20 is a
property of the candidate *set*, so an unchanged value proves the two arms
saw the same 20 documents and differ only in order. The metrics were also
recomputed straight from the persisted predictions without the sidecar and
agree to five decimals. The gap is 4.6 standard errors (per-query se
0.0263); the 95% CI is [0.289, 0.392].

This confirms finding 2's prediction. That finding argued the embedding
model "retrieves the same documents and orders them worse... a reranking
problem rather than a retrieval one, and the shape a cross-encoder fixes."
It was written before any reranker existed. The correction is that the fix
did not need a cross-encoder — a generative model reading the text listwise
was enough.

**What was actually missing was the text.** No agent before this one had
ever shown the LLM a document. `get_node` returns a name and a type;
`search_chunks` matched on text and then discarded it. `zero_shot` and
`deep` were reasoning over identifiers, which is why their numbers sit on
top of the retrieval they were given. `Toolset.search_passages` returns the
matched passage, and one prompt does the rest.

#### Listwise against STaRK's pointwise, and why ours gains more

STaRK's published GPT-4 reranker tops the PRIME Synthesized(10%) board at
0.2655 mrr, a gain of about +3.05 over its dense baseline. Ours gains
+12.20. **That is not "we beat GPT-4"** — it is a different protocol, and
the comparison worth drawing is mechanical rather than competitive:

| | STaRK's reranker | ours |
|---|---|---|
| shape | pointwise, one call per candidate | listwise, one call per query |
| calls/query | up to `max_k=100` | 1 |
| output | one float 0.0–1.0, `max_tokens=5` | 20 scores, 0–100 |
| candidate text | `add_rel=True` (relations included) | relation-free |
| prior | `sim_weight=0.1` retrieval-rank blend | retrieval rank as tie-break only |

Theirs sees *more* text per candidate and gains *less*. That rules out
"better documents" as the explanation and points at the comparison itself:
scoring candidates against each other in one context lets the model use
relative judgements it cannot make when each candidate is scored blind. It
is also ~100x cheaper per query.

#### Reranking is now near its ceiling, and recall is the constraint

hit@20 on this arm is **0.44643** — no reranker can promote a document
retrieval never surfaced. Against that ceiling:

- hit@1 0.28571 / 0.44643 = **64%** of reachable queries answered first
- hit@5 0.40000 / 0.44643 = **90%** reachable into the top five

So further reranking effort has little left to win. For 55% of queries the
answer is not in the candidate set at all, and that is where the next gain
is. This reverses the page's working assumption: ranking was the bottleneck
and is now largely solved; retrieval recall is the bottleneck.

### What is not measured

**The knowledge graph.** It is reached only by `deep`, through `neighbors`
and `get_relationships`, and those arms are phase C. Nothing on this page
yet says whether building a graph helps retrieval — which is the question
the benchmark exists to answer.

### Confidence

280 queries. Individual metric gaps of ~0.014 are four queries and should
not be read alone. What carries weight is direction and structure across
metrics: the granularity effect moves mrr and recall together by ~15%, and
the fusion effect changes sign between corpora. Single-metric differences
under about 0.01 are noise at this sample size.

Read `scripts/results_table.py`'s "Suspect cells" output before believing any
row. It flags all-zero metrics, unbacked cost columns, `deep` against an
edgeless corpus, and reports written against a config that has since changed.

## The benchmark

STaRK `prime`, split `test-0.1`, 280 queries against 129,375 nodes. Metrics
come from `stark_qa.evaluator.Evaluator` in a Python 3.11 sidecar — we compute
none of them ourselves, deliberately, so the numbers are comparable to
published STaRK results rather than to our own arithmetic.

## Where these numbers sit against the published board

STaRK's own leaderboard for **PRIME, Synthesized (10%)** — the same
`test-0.1` split every number on this page uses. Reported as percentages
there; converted here to match our scale.

| method | hit@1 | hit@5 | recall@20 | mrr |
|---|---|---|---|---|
| GPT-4 reranker | 0.1828 | 0.3728 | 0.3405 | **0.2655** |
| Claude-3 reranker | 0.1779 | 0.3690 | 0.3557 | 0.2627 |
| GritLM-7b | 0.1679 | 0.3429 | 0.4111 | 0.2499 |
| multi-ada-002 | 0.1536 | 0.3286 | 0.4099 | 0.2370 |
| **ada-002** | **0.1536** | **0.3107** | **0.3788** | **0.2350** |
| BM25 | 0.1393 | 0.3107 | 0.3284 | 0.2168 |
| voyage-l2-instruct | 0.1214 | 0.3142 | 0.3734 | 0.2123 |
| ColBERTv2 | 0.1500 | 0.2607 | 0.2778 | 0.1998 |
| QAGNN (roberta) | 0.0714 | 0.1714 | 0.3295 | 0.1627 |
| LLM2Vec | 0.0929 | 0.2070 | 0.2554 | 0.1500 |
| DPR (roberta) | 0.0500 | 0.2357 | 0.3050 | 0.1350 |
| ANCE (roberta) | 0.0678 | 0.1615 | 0.1707 | 0.1142 |

**Our `vss-control` reproduces the ada-002 row.** Hit@1 0.1536, hit@5
0.3107 and recall@20 0.37878 are identical to two decimal places in the
published units. That is the control doing its job: it exercises ingest,
tenant isolation, the STaRK-id mapping, the chunk store and the scoring
sidecar end to end, and a harness defect anywhere in that chain would show
up here rather than being attributed to a model.

The one gap is mrr, 0.23057 against 0.2350, and it is explained by the
single difference between the metrics: hit@1, hit@5 and recall@20 are all
decided inside rank 20, while mrr counts ranks beyond it. We return `k=20`,
so a gold answer at rank 40 earns the published run 0.025 and earns us 0.
Worth ~0.4 points here. **Every mrr on this page is therefore an mrr@20**
and reads very slightly low against the board; the other three metrics are
directly comparable.

Two things to read off the board before reading our numbers:

- **The whole field spans about 5 mrr points**, 0.2123 to 0.2655 among the
  serious retrievers. Differences of 0.01 between our arms are a fifth of
  the entire published spread, not noise.
- **The two reranker rows are the top of the board, and both have *lower*
  recall@20 than the ada-002 they rerank** (0.3405 and 0.3557 against
  0.3788). Reranking on this benchmark buys hit@1 and mrr by promoting
  right answers, and pays for it by demoting marginal ones off the end of
  the list. That is a property of the architecture rather than a defect in
  any implementation of it, and it is why recall@20 sits beside mrr in
  every table here.

## What each arm is

| config | embeddings | dim | chunker | chunks/node |
|---|---|---|---|---|
| `vss-control` | ada-002, **precomputed by STaRK** | 1536 | whole-document | 1.000 |
| `native-wholedoc` | Nemotron-3-Embed-1B | 2048 | capped-whole-5000 | 1.057 |
| `redstring-native` | Nemotron-3-Embed-1B | 2048 | boundary-preference | 1.14 |
| `native-sliding1k` | Nemotron-3-Embed-1B | 2048 | sliding 1000/500 | ~1.94 |

`vss-control` embeds nothing: it loads the vectors STaRK ships. It is the
control in the strict sense — if our harness, stores and scoring are wired
correctly, it must reproduce STaRK's own dense-retrieval number, and any
deviation is a bug in us rather than a finding about anything.

## What each comparison isolates

The three native arms share a model, a dimension, a prefix pair, a chunk
table and a corpus. **Only the chunker differs**, which is what makes the
sweep a controlled experiment rather than three unrelated runs.

- `vss-control` → `native-wholedoc` — **the embedding model**, and only that,
  since both present documents essentially whole. This is the cell that
  exists because `vss-control` → `redstring-native` varies model *and*
  chunking simultaneously and can attribute a gap to neither.
- `native-wholedoc` → `redstring-native` → `native-sliding1k` — **chunk
  granularity**, model held fixed. Three points, so the answer is a direction
  rather than an anecdote.
- `dense` → `hybrid` — the lexical channel's contribution, retrieval
  architecture held fixed.
- `dense` → `zero_shot` → `deep` — **agent architecture**, corpus held fixed.
  Read this column against the cost column, never alone: `deep` is budgeted
  at up to 8 LLM and 8 tool calls per query, so it can buy accuracy at
  10-100x the cost of `dense`, and a benchmark that reports only accuracy
  would call that a win.

## Caveats that change what a number means

**The sweep's low two points are closer than intended.** 1.057 and 1.14
chunks/node, against 1.94 for the third. The whole-document cap had to fall
from 10,000 to 5,000 characters to fit a 2048-token `--ubatch-size`. So a
null result between `native-wholedoc` and `redstring-native` is weaker
evidence than a null result between either of them and `native-sliding1k`.

**86% of this corpus is under 1000 characters** and passes through every
chunker whole. The sweep can only move the 14% long tail, which bounds how
large any chunking effect could possibly be.

**`native-sliding1k` carries ~7% redundant chunks.** `SlidingWindowChunker`
emits one fully-redundant tail chunk per document longer than its window
(B-SLIDING-REDUNDANT-1). The text is real, so this is waste rather than
corruption, but its chunk *count* overstates distinct coverage.

**The Nemotron GGUF is Q4_K_M with no importance matrix and no MTEB
evaluation.** NVIDIA's own quantised release is NVFP4 with quantisation-aware
distillation done specifically to recover long-sequence retrieval accuracy,
which is indirect evidence that naive post-training quantisation costs this
model something. Every Nemotron number here is a number for *this
quantisation*, not for the model. A Q8_0 arm would be the control that
settles it and has not been run.

**The two families are not a fair fight on parameters.** ada-002 is a
commercial API model; Nemotron-3-Embed-1B is a 1B model quantised to 4 bits
running on one consumer GPU. If the native arms win, they win having given
away nothing; if they lose, the loss is not attributable to redstring.

## Cost

Accuracy alone cannot rank these architectures, which is the whole reason the
report carries a `cost` block. `tool_calls_per_query` and
`llm_calls_per_query` are exact; `tokens_per_query` is `int | None` and
**`None` is not zero** — it means the provider did not report usage, which is
a different claim from "used no tokens".

`ingest.wall_time_s` is the one-off cost of building each corpus, and it is
not small relative to the differences it buys: the arms differ by hours of
embedding time, and a chunking strategy that wins by a point while costing
twice the ingest is a different recommendation from one that wins for free.

## Reproducing

Every report embeds its config verbatim in `config_verbatim`, including the
task prefixes. A prefix that is not recorded is a number that cannot be
reproduced, and `scripts/results_table.py` compares that field against the
config on disk to catch a report that has outlived its configuration.

    uv run python scripts/sweep.sh      # all arms, all agents, ~5h
    uv run python scripts/results_table.py

Superseded numbers live in `results/archive/` with a README explaining what
each measured and why it is not comparable. The one currently there records
what nomic scored with its task prefixes missing — plausible numbers, nothing
about them visibly wrong, which is the point.

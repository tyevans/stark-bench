# Results

**All twelve retrieval numbers are in** (2026-08-19), and the first
LLM-agent cell with them: `rerank` is now the best result on the page by a
wide margin (finding 6). The table is generated — run
`uv run python scripts/results_table.py` and paste. Everything else on this
page is the framework for reading it, written before the numbers existed so
that it is a prediction rather than a rationalisation.

Two of that framework's predictions were wrong and are marked as retracted
below, in place rather than quietly edited: chunk *granularity* turned out
not to be the variable, and lexical fusion turned out to track dense
weakness rather than chunk size.

## Results

All twelve retrieval numbers, 2026-08-19. 280 queries against 129,375 nodes;
metrics from `stark_qa.evaluator.Evaluator` in the 3.11 sidecar.

| config | agent | mrr | hit@1 | hit@5 | recall@20 | tool/q | llm/q | run s | chunks/node | ingest s |
|---|---|---|---|---|---|---|---|---|---|---|
| native-sliding1k | dense | 0.2125 | 0.1393 | 0.2964 | 0.3373 | 1.00 | 0.00 | 245.6 | 2.238 | 8577.6 |
| native-sliding1k | hybrid | 0.2211 | 0.1571 | 0.2857 | 0.3422 | 1.00 | 0.00 | 460.4 | 2.238 | 8577.6 |
| native-sliding1k | lexical | 0.1988 | 0.1464 | 0.2643 | 0.2445 | 1.00 | 0.00 | 219.3 | 2.238 | 8577.6 |
| native-wholedoc | dense | 0.2163 | 0.1357 | 0.3036 | 0.3778 | 1.00 | 0.00 | 124.0 | 1.057 | 2566.9 |
| native-wholedoc | hybrid | 0.2187 | 0.1500 | 0.2964 | 0.3680 | 1.00 | 0.00 | 225.7 | 1.057 | 2566.9 |
| native-wholedoc | lexical | 0.1944 | 0.1464 | 0.2714 | 0.2197 | 1.00 | 0.00 | 123.4 | 1.057 | 2566.9 |
| redstring-native | dense | 0.1845 | 0.1214 | 0.2500 | 0.3240 | 1.00 | 0.00 | 121.2 | 1.139 | 5066.5 |
| redstring-native | hybrid | 0.1985 | 0.1357 | 0.2643 | 0.3516 | 1.00 | 0.00 | 245.6 | 1.139 | 5066.5 |
| redstring-native | lexical | 0.2014 | 0.1429 | 0.2750 | 0.2402 | 1.00 | 0.00 | 123.1 | 1.139 | 5066.5 |
| vss-control | dense | 0.2306 | 0.1536 | 0.3107 | 0.3788 | 1.00 | 0.00 | 85.7 | 1.000 | 297.0 |
| vss-control | hybrid | 0.2311 | 0.1643 | 0.3214 | 0.3710 | 1.00 | 0.00 | 176.3 | 1.000 | 297.0 |
| vss-control | lexical | 0.1848 | 0.1429 | 0.2607 | 0.2139 | 1.00 | 0.00 | 94.4 | 1.000 | 297.0 |
| native-wholedoc | **rerank** | **0.3408** | **0.2857** | **0.4000** | 0.3680 | 1.00 | 1.00 | 4364.7 | 1.057 | 2566.9 |

Reproducibility was checked rather than assumed: `native-wholedoc/dense` was
re-run four hours after its first scoring and returned
`0.21634584166375653` both times, digit for digit. Differences below are
signal, not run-to-run variance.

### mrr, arranged so the shape is visible

| corpus | chunker | chunks/node | dense | lexical | hybrid |
|---|---|---|---|---|---|
| vss-control (ada-002) | whole document | 1.000 | 0.2306 | 0.1848 | **0.2311** |
| native-wholedoc | capped-whole-5000 | 1.057 | 0.2163 | 0.1944 | 0.2187 |
| redstring-native | boundary-preference | 1.139 | **0.1845** | 0.2014 | 0.1985 |
| native-sliding1k | sliding-1000-500 | 2.238 | 0.2125 | 0.1988 | 0.2211 |

### 1. Nothing in *retrieval* beats the published-vector control

`vss-control` -- plain dense retrieval over STaRK's own ada-002 vectors,
whole documents, no graph -- is the best of the twelve retrieval cells at
0.2311. The closest any locally-embedded arm comes is
`native-sliding1k/hybrid` at 0.2211, 4% behind.

**Scope corrected**: this finding read "nothing beats the control" until
`rerank` scored 0.3408 (finding 6). The claim holds over retrieval and only
retrieval -- which, given that no amount of retrieval work in this table
closed a 4% gap and one LLM call opened a 47% one, is the narrower and less
interesting half of the page.

### 2. The embedding model costs ranking, not recall

`vss-control` -> `native-wholedoc` holds chunking near-constant and swaps
ada-002 for Nemotron-3-Embed-1B. mrr falls 6%; **recall@20 is unchanged**
(0.3778 against 0.3788, a quarter of one query). The same documents are
retrieved and ordered worse -- the shape a reranker fixes and a bigger
bi-encoder may not. Nemotron here is Q4_K_M *with* its task prefixes, so
this is the model configured at its best rather than a strawman.

### 3. The chunker matters, and granularity does not

This retracts a claim an earlier version of this file made. With only
`native-wholedoc` (1.057) and `redstring-native` (1.139) in hand, the 15%
dense gap between them read as "finer chunking is worse", with a mechanism
to match: STaRK scores nodes, `aggregation: max` takes each node's best
chunk, so more chunks per node means more draws and every distractor gets
more lottery tickets.

`native-sliding1k` falsified it. **Twice** the granularity of
`redstring-native` and it scores 15% *better* on dense. The effect is not
monotonic in chunks/node, so it was never about granularity -- two points
that differ in both count and strategy cannot separate the two, and this
file's own caveats section had already warned the low points were closer
together than intended.

What the four points actually show is that **`boundary-preference` is an
outlier**: last of the three chunkers on dense retrieval, beaten by a
whole-document cap and by a naive sliding window at double the granularity.
It is also redstring's own default.

### 4. Dense swings with the chunker; lexical does not

The single most informative column comparison:

| | range across the four corpora |
|---|---|
| dense mrr | 0.1845 - 0.2306 (25% spread) |
| lexical mrr | 0.1848 - 0.2014 (9% spread) |

BM25 over the *same chunks* barely notices what the chunker did. So the
text is present and the terms are present -- what degrades on
`boundary-preference` is specifically the **vector representation** of
those chunks.

Two mechanisms were tested against that and **both were ruled out**:

- **Not lost content.** All three arms cover all 129,375 nodes, and
  `redstring-native` stores 3.7% *more* text than `native-wholedoc`
  (101,616,325 characters against 98,025,525).
- **Not chunk length.** `native-wholedoc` and `redstring-native` have
  nearly identical distributions (median 109 vs 129, p90 2292 vs 2751,
  57.4% vs 53.1% under 200 characters) and the largest dense gap in the
  table. `native-sliding1k`'s distribution is wildly different (median 774,
  27.1% short) and it scores like `native-wholedoc`.

So the deficit is real, reproducible, and unexplained by the obvious
candidates. The remaining one is *where* the splits fall inside long
documents, which needs a passage-level diagnostic rather than an aggregate.
It is not claimed here.

### 5. Lexical fusion tracks dense weakness

| corpus | dense | fusion gain over dense |
|---|---|---|
| vss-control | 0.2306 | +0.0005 |
| native-wholedoc | 0.2163 | +0.0024 |
| redstring-native | **0.1845** | **+0.0141** |
| native-sliding1k | 0.2125 | +0.0085 |

The largest gain lands on the corpus with the weakest dense channel and the
smallest on the strongest. On `redstring-native` fusion does not even beat
its own lexical channel (0.1985 against 0.2014): mixing in a weak vector
signal *costs* mrr.

This also retracts an earlier claim here that fusion "earns its keep on
finer chunks". It earns its keep where the embeddings are struggling, and
on this sweep those two happened to coincide.

The channels are genuinely complementary, though, which is worth separating
from the above. On `native-wholedoc`, BM25 alone **beats dense at hit@1**
(0.1464 against 0.1357) while its recall@20 is 42% lower (0.2197 against
0.3778) -- exact matching nails the obvious cases and misses paraphrase,
and the vector channel is the reverse. Fusion beats both on mrr there.

### 6. Showing the LLM the document beats every retrieval change combined

`rerank` reorders `native-wholedoc/hybrid`'s top 20 with one listwise LLM
call:

| metric | hybrid | rerank | Δ |
|---|---|---|---|
| mrr | 0.21872 | **0.34075** | +0.12203 |
| hit@1 | 0.15000 | **0.28571** | +0.13571 |
| hit@5 | 0.29643 | **0.40000** | +0.10357 |
| recall@20 | 0.36799 | 0.36799 | ±0.00000 |

**recall@20 unchanged to five decimals is the control.** recall@20 is a
property of the candidate *set*, so an identical value proves both arms saw
the same 20 documents and differ only in order. Metrics were also recomputed
from the persisted predictions without the sidecar and agree to five
decimals. The gap is 4.6 standard errors (per-query se 0.0263), 95% CI
[0.289, 0.392].

Put against findings 1-5: every retrieval change measured on this page moves
mrr within 0.1845-0.2311, a band of 0.047. One LLM call moves it 0.122 —
**two and a half times the entire spread of the retrieval work.**

This is finding 2's prediction, which said the embedding model "retrieves the
same documents and orders them worse -- the shape a reranker fixes and a
bigger bi-encoder may not." It was written before a reranker existed. The one
correction: it did not need a cross-encoder. A generative model reading the
passages listwise was enough.

**What was missing was the text.** No agent before this one had shown the LLM
a document. `get_node` returns a name and a type; `search_chunks` matched on
text and discarded it. `zero_shot` and `deep` were reasoning over
identifiers, which is why they never separated from the retrieval they were
handed. `Toolset.search_passages` returns the matched passage.

#### Listwise against STaRK's pointwise

The published GPT-4 reranker tops the PRIME Synthesized(10%) board at 0.2655
mrr, about +3.05 over its dense baseline. Ours gains +12.20. **That is not
"we beat GPT-4"** — it is a different protocol:

| | STaRK's reranker | ours |
|---|---|---|
| shape | pointwise, one call per candidate | listwise, one call per query |
| calls/query | up to `max_k=100` | 1 |
| output | one float 0.0-1.0, `max_tokens=5` | 20 scores, 0-100 |
| candidate text | `add_rel=True` | relation-free |
| prior | `sim_weight=0.1` rank blend | retrieval rank as tie-break only |

Theirs sees *more* text per candidate and gains *less*, which rules out
"better documents" and points at the comparison itself: scoring candidates
against each other in one context supports relative judgements that scoring
each blind cannot. It is also ~100x cheaper per query.

#### Reranking is near its ceiling; recall is the constraint

hit@20 on this arm is **0.44643**, and no reranker can promote a document
retrieval never surfaced. Against that ceiling, hit@1 is **64%** of what is
reachable and hit@5 is **90%**. For 55% of queries the answer is not in the
candidate set at all.

So the page's working assumption inverts. Ranking was the bottleneck and is
now largely spent; **retrieval recall is the bottleneck**, and that is where
the next gain has to come from.

### What is still not measured

**The knowledge graph.** `dense`, `lexical` and `hybrid` all call
`search_chunks` and none of them reads a relationship; `hybrid` is
redstring's rank fusion of the vector and lexical channels, not traversal.
Both native arms carry all 8,100,498 edges and **not one query has touched
them.**

The graph is reached only by `deep`, through `neighbors` and
`get_relationships`. Those arms need the embedding server at `-np 1` with a
4096-token context so the 27B chat model can be co-resident -- see
B-CORESIDENCE-1. Until they run, this page says nothing about whether
building a knowledge graph helps retrieval, which is the question the
benchmark exists to answer.

### Confidence

280 queries. A gap of 0.014 in mrr is roughly four queries and should not
be read alone. The findings above rest on direction and structure across
metrics and corpora -- the chunker effect moves mrr and recall together by
15%, the lexical band is narrow across a 2.2x granularity range, and the
fusion gain changes sign in its relationship to dense strength. Single
metric differences under about 0.01 are noise at this sample size.

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

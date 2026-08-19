# Results

**All twelve retrieval numbers are in** (2026-08-19); the four LLM-agent
cells are not. The table is generated — run
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

### 1. Nothing beats the published-vector control

`vss-control` -- plain dense retrieval over STaRK's own ada-002 vectors,
whole documents, no graph -- is still the best cell in the table at 0.2311.
The closest any locally-embedded arm comes is `native-sliding1k/hybrid` at
0.2211, 4% behind.

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

# stark-bench

Measures the `redstring` knowledge-graph library against the
[STaRK benchmark](https://stark.stanford.edu/) (NeurIPS 2024 D&B), with
**pluggable agent architectures** — the point is not one number, it is the
comparison between retrieval strategies over the same corpus.

This repo is a *consumer* of redstring. It ships no extraction of its own; it
loads STaRK's pre-built semi-structured knowledge base into redstring's stores
through redstring's own ports and adapters, then retrieves against them.

## The one-paragraph orientation

A **loader** (`skb/ingest.py`) projects STaRK nodes into `Entity` +
`StoredChunk` via redstring's `EntityWriter`/`ChunkWriter`. A **toolset**
(`tools/redstring_tools.py`) exposes retrieval as `search_chunks` /
`neighbors` / `extract`. An **agent** (`agents/`) is anything satisfying the
`Agent` protocol in `ports.py` — it gets a `Query` and a `Toolset` and returns
ranked node ids. The **runner** drives agents over the query set, and
**scoring** shells out to the official `stark_qa` evaluator in a Python 3.11
sidecar. A **config** (`config/*.yaml`) names the chunker, embedding model,
aggregation, agent and k, and is embedded verbatim in every results file.

## Environment

`uv`, Python 3.13. Everything project-scoped runs through `uv run`.

```
uv sync                     # dev group is default
uv run pre-commit install   # gate for the gates; see below
docker compose up -d        # postgres :55432, neo4j :57687 (non-standard ports)
```

**What the stores hold, as of 2026-08-20.** One chunk table exists,
`kg_chunks_qwen3_embedding_0_6b_1e22db42`, with 878,479 rows across six
tenants. Run `uv run python scripts/verify_corpus.py` for the current
picture rather than trusting this paragraph — it compares every ingest
report against `count(*)` for that arm's tenant and prints `MISSING` for
corpora that no longer exist.

Live: `qwen-rel-sliding1k` (549,697), `qwen-wholedoc` (151,232),
`qwen-rel-whole` (129,656), `vss-control` (129,375), and the three
`qwen-mini-*` arms.

**Gone**, and needing a re-ingest before they can be scored again: every
nomic and Nemotron arm. `results/*.json` for them live on disk and
survived, so their numbers are still readable — they just cannot be
reproduced without rebuilding the corpus.

`vss-control` was rebuilt on 2026-08-21 and **reproduced
`0.23057383129905376` to every digit**, which is the check that it came
back correctly. It costs minutes and no endpoint time, since it reloads
STaRK's precomputed vectors.

That exact match is now doing double duty. The rebuild ran through
redstring PR #71's rewritten `ORDER BY`, so it is also the evidence that
the rewrite is behaviour-preserving on real data to seventeen significant
figures rather than merely monotonic in argument.

A `--run` against a missing corpus now refuses up front rather than
scoring an empty store as a bad retriever.

**`redstring` is a path dependency and the path is load-bearing:**

```toml
[tool.uv.sources]
redstring = { path = "../redstring", editable = true }
```

It points at the user's own checkout, on `main`. It used to point at
`../redstring-chunkfix`, a worktree kept deliberately separate so that
checkout could sit on a feature branch without changing what this benchmark
measures — that worktree still exists, on the older
`feat/embedding-task-prefixes` branch, and is no longer what is installed.

**What is actually load-bearing is the chunker source, not the directory
name.** redstring's chunking behaviour is one of the things under
measurement and it has already changed once mid-campaign (PR #64 below), so
the check worth running before trusting a cross-arm comparison is the diff
itself:

```
diff -rq --exclude=__pycache__ \
    ../redstring/src/redstring/extraction/chunkers \
    ../redstring-chunkfix/src/redstring/extraction/chunkers
```

At the qwen re-ingest (2026-08-19) that came back identical — only `.pyc`
files differed — which is why the qwen arms are comparable to the nomic and
Nemotron rows despite the move. Had it come back different, every existing
number would have been incomparable and the fix would have been to repoint,
not to reason about it. `.pyc` noise is why `--exclude=__pycache__` is in the
command rather than left to be rediscovered.

`editable = true` is also load-bearing, for a different reason the inline
comment in `pyproject.toml` records: installed as a plain path copy, a
redstring change does not reach this venv until something forces a
reinstall, which already cost one calibration run.

Check the checkout is clean of *library* source before quoting a number —
`git -C ../redstring status --porcelain`. Session notes and `bench/*.yaml`
being dirty is normal and harmless; anything under `src/` means the arm is
measuring an uncommitted state and is not reproducible.

`pyproject.toml` and `uv.lock` are frequently modified in the working tree
for this reason; do not sweep them into an unrelated commit.

Never edit dependency tables by hand — `uv add` / `uv remove`.

## Running it

```
# ingest a config's corpus (idempotent; resumes by skipping existing chunks)
uv run python -m stark_bench.composition.cli --config config/<name>.yaml --ingest \
    --embed-concurrency 16

# score one architecture against it
uv run python -m stark_bench.composition.cli --config config/<name>.yaml --run \
    --agent hybrid

# regenerate the results table
uv run python -m stark_bench.composition.cli --summarise results/ > RESULTS.md

# what the stores actually hold, against what the reports claim
uv run python scripts/verify_corpus.py
```

`--agent` overrides the config's own `agent:`, so one config serves every
architecture. Reports land in `results/<name>.<agent>.json`.

**Seventeen agents are registered**; `uv run python -c "from
stark_bench.composition.agent_registry import AGENTS; print(sorted(AGENTS))"`
is the list, and `tests/composition/test_agent_registry.py` asserts it as an
exact set so a new architecture cannot appear without someone deciding it
should. Four families:

| family | what varies |
|---|---|
| `dense`, `lexical`, `hybrid` | retrieval only, no LLM |
| `zero_shot`, `deep` | LLM-driven; scored only on the dead Nemotron arms |
| `rerank`, `rerank40`, `rerank40lean` | rerank on full documents |
| `rerank40title*`, `rerank80*` | rerank on lean encodings — see FINDINGS 1b |

### Flags

**Ingest:** `--limit N` (throughput calibration only — never for a reported
number, and see B-RATE-UNIT-1 before extrapolating from one), `--no-resume`
(force re-embed; upserts cleanly over existing rows), `--no-cache` (force a
cold embed), `--ingest-edges` (off by default), `--embed-concurrency`,
`--embed-batch`.

**Run:** `--query-concurrency N` (queries in flight; **set it to at least
the chat peer's `-np` or the extra slots idle**, and record it — it is in
the report and in `RESULTS.md`'s `conc` column), `--chat-model ID`
(override the LLM for one run; the report and the filename carry the model
that ran, because `config_verbatim` is the config FILE's bytes and would
name the one that did not), `--split NAME` (run `test` instead of
`test-0.1` against the same store — the tenant is derived from the config
NAME, so no re-ingest).

**Never make a new config just to change the model or the split.** The
tenant is `uuid5` of the config name, so a new name points at an empty
corpus — the incident where 280 queries returned empty and the only symptom
was a `min()` error from a subprocess.

### Chunk vectors are cached across arms

Live-embedded chunks are cached in `kg_embedding_cache`, content-addressed on
`(model, document_prefix, sha256(text))`, so text embedded once is never
embedded again.

**Measured on `prime-mini` (10,000 nodes), 2026-08-20:**

| case | wall | hits | misses | speedup |
|---|---|---|---|---|
| cold, empty cache | 293.8 s | 200 | 11,781 | — |
| **same config again** | **14.1 s** | 11,981 | 0 | **21x** |
| **different chunker, same corpus** | 409.3 s | 10,860 | 13,424 | **~1.45x** |

Read the third row before planning a sweep. An earlier version of this
section estimated that a three-chunker sweep would cost "roughly one endpoint
pass rather than three", and **that was too optimistic**: the cross-chunker
hit rate is **44.7%**, not ~86%. The document-level intuition is right — most
documents are short enough to come through whole under any chunker — but a
finer chunker also emits *more* chunks per document (2.43/node here against
1.20), and every extra chunk is a guaranteed miss. Hit rate is per chunk, not
per document.

Where the cache pays enormously is re-running the *same* configuration:
recovering from a crash, re-scoring after a code change, or re-ingesting a
tenant. That went from 5 minutes to 14 seconds.

**The key carries the model and the prefix, and that is not optional.** A
corpus embedded with a prefix and the same corpus embedded without it are not
comparable vectors (ADR 0002, ADR 0043) — the same reason `_table_for` folds
both into the chunk table name. A cache keyed on text alone would serve one
arm's vectors to another, and cosine similarity between them returns a
perfectly plausible number.

**Correctness is checked, not assumed.** A warm-cache `qwen-mini-wholedoc`
scores `mrr=0.3590479185937771` — byte-identical to the figure measured
before the cache existed. Embedding is deterministic, so the cache either
returns the same vector or a different one; "close" would mean a defect. Redo
that comparison after any change to the key.

**The key carries the model and the prefix, and that is not optional.** A
corpus embedded with a prefix and the same corpus embedded without it are not
comparable vectors (ADR 0002, ADR 0043) — it is the same reason `_table_for`
folds both into the chunk table name. A cache keyed on text alone would serve
one arm's vectors to another, and cosine similarity between them returns a
perfectly plausible number.

This is **not** the resume path. Resume skips chunks already stored *in this
tenant*; the cache skips embedding text seen in *any* tenant, ever, including
text whose `chunk_id` differs because another chunker gave it a different
`start_char`.

Every report carries `cache_hits` and `cache_misses`. A sweep's second arm
over the same corpus that is **not** almost entirely hits is telling you the
key is wrong — which nothing else in the report would show. `--no-cache`
forces a cold run, for the same reason `--no-resume` exists: reusing work is
exactly the kind of optimisation that can hide a bug.

Nothing evicts. At ~1.2M distinct chunks across these corpora the table is on
the order of gigabytes; `TRUNCATE kg_embedding_cache` is the reset.

## The shared inference endpoint

`http://192.168.1.14:8080/v1/` serves both embeddings and chat, and **it is
the user's machine, used for other things**. Ask before saturating it, and
record the slot count in any result you report — it changes mid-session.

Two hazards, both hit for real:

- **Restarting the server hangs while our clients hold connections.** Kill the
  ingest first, confirm with `ss -tnp | grep 192.168.1.14:8080`, then restart.
- **Per-slot context is `--ctx-size / -np`, not `--ctx-size`.** With
  `-np 16` and `--ctx-size 32768` each slot gets 2048 tokens, and a longer
  document is rejected with `input (N tokens) is larger than the max context
  size`. For embeddings llama.cpp uses non-causal attention, so
  `--ubatch-size` must *also* be ≥ the longest sequence. Both have to clear
  it, and **`--ubatch-size` is the one that is usually forgotten** — it was
  the binding constraint on three separate occasions here while attention
  was on `--ctx-size`, and its error says *"increase the physical batch
  size"* rather than anything about context.
- **A server flag you edited may not be the one running.** llama-swap
  launches the embedding peer with its own command line, so edits to
  `llama-server-embed.service` did nothing until the peer was restarted
  through llama-swap. Twice this looked like a model limitation: a config
  reporting `n_ctx: 2048` was diagnosed as "nomic's GGUF declares 2048" and
  that diagnosis was wrong — a completely different model reported the same
  2048 an hour later. **Verify a server change against `/props` on the peer
  port before reasoning about what it means.**
- **Client concurrency must be at least the server's `-np`.** A request
  occupies one slot, so `--embed-concurrency 1` against `-np 4` leaves three
  quarters of the server idle whatever the batch size. Measured over 3000
  nodes: `1 x 128` gave 1233 nodes/min and `4 x 128` gave 1618. Slots scale
  sublinearly (1->2 is +17%, 2->4 is +12%), so there is little past four.
  The full table and the reasoning live in `skb/ingest.py`'s docstring.
- **`--embed-batch` is the second knob and is worth about 18%** at fixed
  concurrency. Both are needed: batching raises the per-slot ceiling,
  concurrency decides how many slots work at all.
- **Never tune against GPU utilisation.** `nvidia-smi` reported its highest
  figure on `1 x 128` -- the *slowest* of seven configurations -- because a
  kernel is resident whenever any slot is busy and three idle slots are
  indistinguishable from none. It answers "is the device doing something",
  not "is it doing as much as it could". Tuning against it would have picked
  a setting 24% off the best while feeling well-informed.
- **A throughput claim needs the baseline you are actually replacing.** A
  standalone probe here showed "6x from batching" and was measuring TCP and
  HTTP handshakes: it used `urllib` with no connection reuse, while the
  ingest has a pooled client and was never in that regime.
- **Do not compare two throughput numbers measured on different slices.**
  The first probes ran on the first 400 nodes of `nodes.jsonl`, which are
  unusually short — 1801 nodes/min there against 1449 on the first 1500,
  same server. That looked like a regression from a server change and was
  entirely the sample. Fix the slice before varying anything else.
- **A resumed ingest measures nothing.** Re-chunking to compute ids and
  skipping what is already stored is CPU work that writes no rows, so the
  row count is flat while the process is busy and then jumps. On
  2026-08-19 a rate taken across that phase read **611 chunks/min** and was
  reported with a 1h45m ETA; the same run's real rate was ~3,600/min and it
  finished in minutes. **Only measure an arm ingesting into an empty
  tenant**, and if you must watch a resumed one, wait for the skip phase to
  end before starting the clock.
- **Scope every progress query to the tenant.** `native-wholedoc`,
  `redstring-native` and `native-sliding1k` share one chunk table and are
  separated only by `tenant_id`, so a bare `count(*)` sums three arms. This
  made an arm at 133,919 read as 141,673 against a target of ~136,700 —
  finished and overshooting when it was neither. See B-MONITOR-TENANT-1.

### The only clean number we have

**2,046 chunks/min**, whole-arm: `redstring-native` wrote 147,329 chunks
into an **empty** tenant in 72 minutes on 2026-08-19, at
`--embed-concurrency 4 --embed-batch 64` against Nemotron-3-Embed-1B
served with `--ctx-size 16384 --batch-size 8192 --ubatch-size 8192 -np 4`.

Quote that figure, not an interval. An earlier version of this section said
**1,666 chunks/min** from two consecutive 3-minute samples early in the same
run, and it was 19% low: the run's instantaneous rate ranged from 334 to
6,666 chunks/min depending on document length and flush timing. The tail is
the slow part -- the longest documents produce the most tokens per batch,
and request rate fell from 117/min at the start to 9/min at the end while
the process was entirely healthy.

One interval read 1606 and was **not** clean: a `--run` scoring pass was
launched during it. That pass used precomputed vectors and touched no GPU, which is
why it was thought safe to run alongside — and it contends on *Postgres*
instead, where `hybrid` does lexical search over a 5.7M-row terms table
while the ingest is writing to the same database. The next interval fell to
1071/min, a 36% drop, and **recovered to 2406/min in the interval after
that pass finished**. The recovery is what makes this a diagnosis rather
than a coincidence: 1666 before, 1071 during, 2406 after, with nothing else
changed. **Nothing else may touch the database while a throughput number is
being taken**, whatever it does to the GPU.

### Sampling this workload is harder than measuring it

Two mistakes, both made on 2026-08-19, both of which produced a confident
wrong answer:

- **A short interval measures the flush, not the throughput.** Chunks
  commit in batches (`CHUNK_BATCH = 1000`), so the row count steps rather
  than climbs. Three-minute samples of one arm gave 1650, 5768, 2000, 333,
  667 and 5771 chunks/min -- a seventeen-fold spread -- while the running
  average stayed near 2100. A 333 reading was *exactly one flush*. **Quote
  the whole-arm average**: total chunks over total wall time, which is the
  only figure immune to when you happened to look.
- **Two rates from two different windows are not comparable.** A request
  rate sampled in one minute (117/min) was divided by a chunk rate from a
  different interval, implying each request carried a handful of texts
  against a configured 64 -- and nearly became a filed batching bug.
  Measured over *one* 90-second window: 91 requests, 5500 chunks, **~60
  texts per request**. Batching was working the whole time. Sample both
  sides of a ratio in the same window or do not compute it.

**Whether `--ubatch-size 8192` beat 4096 is unresolved, and no number in
this file answers it.** Every 4096 measurement was taken across a resume
skip phase, and the one prior clean figure (1792 nodes/min at
`--ubatch-size 2048`) used a *different chunker*, so its chunks differ in
both size and count. Settling it needs the same arm re-ingested at both
settings — about 40 minutes of GPU time for a throughput knob. It was
judged not worth it against ten uncollected result numbers. Do not quote a
comparison; there isn't one.

Background a long ingest with the harness's own backgrounding, **not `nohup`
or `setsid`** — those did not survive here, twice, and the second time the
launcher exited 0 while the real work never started.

## Reasoning is off, and the request wins over the server

redstring's `LangChainLlmProvider.openai_compatible` defaults to
`thinking=False` and sends
`chat_template_kwargs: {"enable_thinking": false}`. `_llm_for` does not
override it, so every LLM call from this harness is non-reasoning.

**A server configured the other way does not override it**, which is worth
knowing because the endpoint here is launched with `--reasoning on` and
`--chat-template-kwargs '{"reasoning_effort":"low"}'`. Measured against it,
same prompt, `temperature=0.0`:

| | wall | completion tokens | reasoning chars |
|---|---|---|---|
| `enable_thinking: false` (ours) | 6.14s | 375 | **0** |
| server default | 8.04s | 512 (capped) | 1503 |

Structured extraction through the port is faster still -- **0.7s** per
`extract` call, three runs, byte-identical output. Budget phase C off that
number: `deep` at up to 8 LLM calls per query is roughly 2.4h for all three
configs, not the 6-9h a reasoning model would cost. An estimate built on
reasoning-on latency will be wrong by an order of magnitude and will tempt
you into dropping arms you do not need to drop.

Do not turn thinking on to "improve" extraction. Beyond the latency, the
docstring on `NO_THINKING` records that two thinking-on runs at temperature
zero disagreed with each other about how many entities a document held,
while two thinking-off runs did not. Every accuracy number in this
repository is a difference between two runs.

## A chunk the server rejects is re-split, so the cap need not be right

When the embedding provider rejects a text for length, `stark_ingest_engine`
re-chunks that group at half the size and retries, up to
`MAX_RESPLIT_ATTEMPTS`. The path only runs on rejection, so a correct cap
costs nothing and a wrong one costs a retry rather than a run.

**Why the cap is not computed instead.** Capping by tokens with the model's
own vocabulary was considered and rejected: it needs a `tokenizers`
dependency and a downloaded vocabulary, and it would put a *second*
estimate of the model's tokenization next to the server's real one.
`all-MiniLM-L6-v2` was available locally as a stand-in and shares BERT
WordPiece — using it would have been the same "close enough" reasoning that
produced three wrong caps in a row. **The server's own 400 is the only
oracle that cannot disagree with the server.**

`/tokenize` was probed first and is not routed by llama-swap; only the
`/v1/*` surface is reachable.

## Embedding models need task prefixes, and the port cannot express them

**Read this before trusting any retrieval number.**

`nomic-embed-text-v1.5` requires `search_document: ` on corpus text and
`search_query: ` on queries. The BGE family wants an instruction on the query
side and nothing on the document side. Most modern embedders are asymmetric
this way.

**This is now solved in the library, not here.** redstring's
`EmbeddingProvider` port has two sides — `embed(texts)` is the corpus side
and `embed_query(texts)` is the query side — and
`LangChainEmbeddingProvider` takes `document_prefix` and `query_prefix`,
both defaulting to empty. See redstring ADR 0043. The port grew the
distinction because a rule that every call site must remember to prepend a
string is a rule that holds only until someone forgets.

An unprefixed run does not fail. The vectors are well-formed, they cluster
sensibly, and they score plausibly. The only symptom is a retrieval number
quietly worse than the model can produce — indistinguishable from "this model
is mediocre", which is the conclusion this project nearly drew about nomic.

What this repo does: `RunConfig.document_prefix` and
`RunConfig.query_prefix` are stated per config, so they land in
`config_verbatim` in every report — a prefix that is not recorded is a
number that cannot be reproduced. `_live_embeddings_for` passes them
through, and `_table_for` folds them into the chunk table name, because
**a corpus embedded with a prefix and the same corpus embedded without it
are not comparable vectors**. Two prefixings must never share a table.

`PrefixedEmbeddingProvider`, a wrapper drafted here before the library
grew the method, has been deleted. Do not reintroduce it.

The current model is `Nemotron-3-Embed-1B`, which wants `passage: ` and
`query: ` — with the trailing space, no newline, and a prefix on *both*
sides. Its `1_Pooling/config.json` sets `include_prompt: true`, so the
prefix tokens belong inside the mean pool, which is what prepending
client-side gives you.

## How precisely an LLM number may be quoted

**LLM arms are not reproducible run to run.** `rerank40title` on
`gemma-4-26b-qat`, same corpus, same tenant, same split,
`temperature=0.0`, thinking off, run twice within an hour:

| metric | run 1 | run 2 | delta |
|---|---|---|---|
| mrr | 0.34100392 | 0.33974803 | 0.00126 |
| hit@1 | 0.25714286 | 0.25357143 | 0.00357 |
| hit@5 | 0.43928571 | 0.45000000 | **0.01071** |
| recall@20 | 0.46431879 | 0.47205688 | 0.00774 |

Temperature zero does not make a *batched* server deterministic: with
`-np 4` and continuous batching, a request's logits depend on which other
requests share its batch, and floating-point addition is not associative. A
handful of near-tied argmaxes flip and the ranking moves.

**So: a difference below ~0.001 mrr between two LLM arms is noise, and
hit@5 should not be quoted precisely on these arms at all** — its floor is
~0.011, about 2.5% relative.

**Retrieval-only arms are NOT unaffected. This paragraph used to say they
were, and that was falsified on 2026-08-21.** Two runs of
`qwen-rel-sliding1k` dense — same code, same rows, no index in either —
disagreed on **103 of 280 queries**, moving mrr by 0.000054 and recall@20
by 0.000397.

The cause is upstream of the database: the same text embedded twice
against the endpoint, alone, back to back, returns **different vectors**
(max component delta 3.3e-03, cosine 0.99981). Calls 2, 3 and 4 agree with
each other and differ from call 1, so it is the first call after idle
rather than per-call jitter; the mechanism is not pinned down.

The noise source was never "an LLM" — it is a batched inference server,
and the query embedder is one of those too. So **a dense or hybrid mrr
difference below ~0.0002 is noise.** The LLM floor above still dominates
and is unchanged.

`vss-control` reproducing `0.23057383129905376` to every digit remains a
valid check, and this explains why: its vectors are precomputed, so it
makes no live embedding call. That is also why it is deliberately left
**unindexed** while every other arm is indexed — it is the one arm whose
exactness is the check. See B-QUERY-EMBED-NONDETERMINISM-1.

Whether `--query-concurrency 1` restores determinism is unresolved and
cheap to settle: two serial runs of one arm. If it does, the cause is
confirmed as batch composition and reproducibility can be bought at ~4x
wall time. See B-LLM-RUN-NOISE-1.

## Where the numbers are

**`RESULTS.md` is generated** — `--summarise results/` renders every scored
arm, grouped by corpus. Regenerate it rather than editing it.

**`FINDINGS.md` is the narrative**: what the campaign established, what it
falsified, what it learned about measuring itself, and what is worth doing
next. Read it before designing an arm. What follows here is the short
version.

### The headline: relational text is worth a lot, through the LEXICAL channel

`prime` / `test-0.1`, 280 queries, k=20, official evaluator. Same model
(qwen3-embedding-0.6b), same ~1.0 chunks/node, differing only in whether the
indexed document carries a `- relations:` block naming the node's neighbours:

| agent | `qwen-wholedoc` (no relations) | `qwen-rel-whole` (relations) | change |
|---|---|---|---|
| dense | 0.18274 | 0.18664 | **+2%** |
| lexical | 0.20479 | 0.24913 | **+22%** |
| hybrid | 0.19870 | **0.28214** | **+42%** |

`qwen-rel-whole` hybrid at **0.28214** was the best figure this project had
produced when this section was written — above `vss-control`'s best
(0.23111) and above STaRK's published ada-002 VSS figure of 0.2350.

**Both have since been beaten by chunking the same corpus.**
`qwen-rel-sliding1k` — the identical relational documents in
1000-character windows — reaches **0.34675** hybrid and **0.25319** dense
with no LLM at all, against `qwen-rel-whole`'s 0.27711 and 0.18269 on the
same basis (`hnsw/ef=800`). That is the best retrieval-only figure here
and above every published retriever on this split, including GritLM-7b
at 0.2499 — on an embedder that loses to ada-002 by 6.4 mrr on the dense
channel, so the margin is corpus construction and fusion rather than the
model.

**Reranking has since gone well past it.** The current best is
`qwen-rel-whole` + `rerank40` at **0.46323** (measured on exact retrieval,
before the index — see the indexed-basis section), and the best lean arm
is `rerank40titlerelmatrix` at 0.41771. `hybrid` remains the right
baseline to read the rerank arms against, because reranking can only
reorder what it found: at `fetch=20`, recall@20 is *identical* to hybrid's
to four decimals, by construction.

**The mechanism is the interesting part.** PRIME's queries name related
entities verbatim ("a drug that targets X and is indicated for Y"). BM25
matches those names directly in the relations block; a single dense vector
compresses them away. So the gain is almost entirely lexical — dense barely
moves. That also retro-explains an oddity in the older data: lexical beat
dense on 5 of 9 arms, which looked like noise and was this effect showing
through weakly on corpora that lacked the text.

### Two later findings sharpen this one

**FINDINGS 1b — relational text only helps if you choose WHICH relations.**
Eight neighbour names per candidate, identical count and token cost,
differing only in which eight: document order scores **0.030 BELOW** showing
no relations at all, BM25-ranked scores 0.054 above. The swing from
selection alone is +0.083 mrr. "More context helps the reranker" is false
here — names of entities unrelated to the query are noise it spends
attention on.

That does not contradict this section. Here the whole relation block reaches
the *retriever*, where BM25 selects at retrieval time and nothing has to
choose. On a lean encoding the choosing is explicit, and doing it badly is
worse than not doing it.

**FINDINGS 1c — chunking helps the lexical channel by 29% on this corpus**,
against the prediction written into the config before the run. Length
normalisation is the mechanism: a term matched inside a 12,000-character
document is heavily discounted, the same term in a 1,000-character chunk is
not, and `aggregation: max` lets the node take its best chunk. It also
explains why the earlier chunking sweep found so little — that ran on
`prime`, median document 103 characters, where chunking cannot rescue
documents from a penalty they were never long enough to incur.

### A hypothesis this falsified, recorded because it drove real decisions

`qwen-wholedoc` dense (0.183) against `vss-control` dense (0.231) was read
here as "ada-002 is reading a richer corpus" — the theory being that STaRK's
precomputed vectors are over `add_rel=True` documents. That theory redirected
a day of endpoint time, and **it is wrong**: giving qwen the relational text
moved dense by 0.004, leaving the gap to ada-002 essentially intact.

So on the dense channel, ada-002 really is better than qwen3-embedding-0.6b
here, and the earlier caveat stands with one addition — every local number is
for a **Q8_0 GGUF served by llama.cpp**, not for the model as published.
B-ADA002-PROVENANCE-1 records what is still unverified about STaRK's own
embeddings; it no longer changes any conclusion of ours.

The lesson worth keeping: the arm was built to test a hypothesis, the
hypothesis lost, and the arm produced the project's best result anyway — for
a reason nobody predicted.

`vss-control` is validated: after a table migration orphaned its corpus, a
re-ingest reproduced `0.23057383129905376` to every digit. That exact-match is
the standing reproducibility check for the ada-002 path — it costs no endpoint
time, since its vectors are precomputed.

The MRR gap to STaRK's published ada-002 figure (0.2350) is fully explained by
our `k=20` truncation, which can only lower MRR and cannot touch hit@1, hit@5
or recall@20 — all three of which match.

### The configs, and what varies between them

| config | chunker | model | chunks/node |
|---|---|---|---|
| `vss-control` | whole-document | ada-002 (precomputed) | 1.00 |
| `native-wholedoc` | capped-whole-7000 | nomic | ~1.03 |
| `redstring-native` | boundary-preference | nomic | 1.14 |
| `native-sliding1k` | sliding-1000-500 | nomic | ~1.94 |

The bottom three hold the model fixed and vary only chunking — that is the
sweep answering "does chunking help or hurt". `vss-control` differs from all
of them in *two* ways at once (model and chunking), so a gap to it cannot be
attributed to either without `native-wholedoc` in hand.

Two things to remember when reading the sweep: **86% of this corpus is under
1000 characters** and comes through whole under every chunker, so only the 14%
long tail can move. And `native-wholedoc` splits rather than truncates the
2.07% of documents the endpoint will not take whole, deliberately — truncating
would make that arm hold less *text* than the others, measuring content loss
and calling it granularity.

## Retrieval is indexed now, and that is a basis, not a detail

As of 2026-08-21 each live tenant has a **partial HNSW index** built by
`scripts/build_ann_indexes.py`, and the database is set to
`hnsw.ef_search = 800`. Every report records `ann_index`,
`ann_index_scans_cumulative`, `hnsw_ef_search` and `retrieval_is_exact` in
its cost block, because an approximate number and an exact one are
otherwise indistinguishable on disk.

**The right `ef_search` depends on corpus size, and a sweep on a small
corpus will mislead you.** Measured against exact scans on both:

| `ef_search` | `qwen-rel-whole` dense (129,656) | `qwen-rel-sliding1k` dense (549,697) |
|---|---|---|
| 40 (pgvector default) | −0.0032 mrr, −0.0117 recall@20 | not measured |
| 200 | −0.0040 mrr | **−0.0110 mrr** |
| 400 | not measured | −0.0028 mrr |
| 800 | +0.0002 mrr (= exact) | +0.0003 mrr (= exact) |

200 was chosen first, off the 129k corpus alone, and costs nearly three
times as much on the 550k one. **There is no speed argument for the lower
setting**: `qwen-rel-sliding1k` dense is 8.0s at 800 against ~30s at 200
and 165.8s exact, because a wider walk still beats scanning 2.25GB of
vectors. Sweep on the largest corpus you have, not the fastest one.

**Numbers taken on this basis are not comparable to numbers taken before
it.** Check `retrieval_is_exact` and `hnsw_ef_search` before comparing two
rows.

`vss-control` is deliberately **not** indexed. See the reproducibility
section above.

### Why partial, and why it took a redstring change

A single ANN index over the shared multi-tenant chunk table is the trap
redstring's own docstring describes: the planner either takes the `k`
globally nearest rows and drops other tenants' afterwards, or ignores the
index entirely. `WHERE tenant_id = '...'` makes it usable only for queries
carrying that predicate, which is redstring BACKLOG B10k's option (3) and
fits because there are six tenants, not six thousand.

That alone was not enough. redstring's `_semantic_candidates_sql` ordered
by `1 - (embedding <=> $2) / 2 DESC`, which no pgvector opclass can serve,
so the first three indexes built here went **completely unused** — 5.7GB,
`idx_scan = 0` on all three — and made dense retrieval **3.1x slower**
(165.8s -> 519.1s) by evicting the table from the page cache. redstring
PR #71 changes the ordering to the raw distance ascending. Same order,
same tie-break, servable by an index.

**Verify with `idx_scan`, never with `EXPLAIN` of a query you wrote
yourself.** `EXPLAIN` on the simplified `ORDER BY embedding <=> $1 LIMIT
20` shows a beautiful index scan and is about a query this codebase never
issues; that is how the above survived an hour.

```
select indexrelname, idx_scan from pg_stat_user_indexes
 where indexrelname like '%hnsw%';
```

### Building them

`uv run python scripts/build_ann_indexes.py <config-name>...`, roughly one
minute per 130k rows at 1024 dimensions. Three traps, all hit:

- Docker's default `/dev/shm` is 64M and defeats a parallel build. It must
  also **exceed `maintenance_work_mem`**, because a parallel build
  allocates that much *shared* memory — 12G of work mem against 2G of shm
  fails with `DiskFullError` on a host with 131G free. `docker-compose.yml`
  now sets `shm_size: 16gb`.
- The memory cliff is real but not a cliff: at 2G (under the 2.25GB of raw
  vectors) the build fell back to disk and ran at 0.9%/min; at 12G it ran
  at 2.4%/min. Better by 2.6x, not by an order of magnitude.
- **Killing the client does not cancel `CREATE INDEX`.** The backend runs
  on and finishes. Cancel it with `pg_cancel_backend`, and do not assume a
  build died because its launcher did — one index here completed after its
  process was written off, and the relaunch collided with it.

## Chunk tables are keyed on the embedding model

`_table_for` derives `kg_chunks_<embeddings-slug>`; tenant is `uuid5` of the
config `name`. So configs sharing a model share a table and are separated by
`tenant_id`, and a different model gets a different table — ADR 0002.

**Serving a new model under an existing model id silently mixes two models'
vectors in one table.** Cosine similarity between them returns a perfectly
plausible number. Give a new model a new `embeddings:` identifier.

This keying has already caused one incident: the per-model rename landed
*after* `vss-control` had been ingested and scored, orphaning 129,375 rows in
the old `kg_chunks` and leaving the new table empty. All 280 queries then
retrieved nothing, `runner.run` logged zero failures — retrieval *succeeded*
and returned empty — and the only symptom was a `ValueError: min() arg is an
empty sequence` from inside a 3.11 subprocess. `score_predictions` now rejects
empty predictions up front and names the queries and the likely cause.

## Every real bug in this project has been silent

Nine for nine. A chunker that inflated its output 3.2x; a corpus that moved
tables; 280 queries retrieving nothing with no error logged; an embedding
model running below spec for want of a string; a model swapped underneath a
still-advertised model id; a `write_report(ingest={})` that emptied the cost
column of every report ever written.

Three more on 2026-08-20, all found the same way:

- **A reranker returning `{"scores": []}`** for most queries, in 1.55s with
  no decode. Empty scores fall through to retrieval order, which scores
  *identically to `hybrid`*, while `run_queries` still logs `0 empty`. The
  cause was stripping the schema docstrings to save tokens — which removed
  the only text telling the model what to put in the array. It would have
  read as "titles are not enough", which is exactly the hypothesis the arm
  was built to test.
- **189 chunks silently deduplicated.** Chunk ids are content-addressed on
  `(source, text)` with no `start_char`, so sliding windows over repetitive
  text collapse on upsert. The reported count says 549,886 and the table
  holds 549,697. Found by `scripts/verify_corpus.py`, not by anything
  failing.
- **`seconds_total` quietly stopped being wall time** when the runner became
  concurrent — the calls overlap, so the sum counts the same seconds up to N
  times. A run taking ~480s reported 1933s under a column headed `seconds`.

**None raised an exception at the point of failure.** Each surfaced only
because a number looked wrong, and each took real time to trace back.

The habit that follows: **assert on the data at each step, not on exit codes.**
A stage that "succeeded" is not evidence it did anything. Check chunks per
node, minimum chunk length, row counts per tenant, and non-empty predictions —
and make the check fail loudly rather than log a warning nobody reads.

Two of those checks now exist as commands, and both found something the
first time they ran:

```
uv run python scripts/verify_corpus.py   # reports vs rows, per tenant
```

and a `--run` refuses up front when the arm's tenant holds no chunks.

Corollary, learned the same way: **a zero, an empty, and a perfect score are
the results most in need of suspicion.** "0 failed" and "0 collected" are the
same exit status.

### Two of the six were "the helper works, nobody calls it"

Worth its own heading because it happened **twice in one session**, hours
apart, in unrelated code, and both times the tests written specifically to
prevent it passed:

- `_live_embeddings_for` was reverted by hand to drop both `*_prefix=`
  arguments — the original prefix defect, restored deliberately. All 39
  harness tests passed, including four new ones covering the prefix
  machinery. A table name reacting to a prefix proves the config field is
  read *somewhere*, not that a byte of it reaches the server.
- `write_report(...)` was reverted to `ingest={}`. All 45 harness tests
  passed, including four new ones covering `_ingest_stats` from every
  angle.

The shape: exhaustive tests of a helper, and nothing asserting the call
site uses it. No test of a helper can see this, because the helper is
correct. **When you add a helper that one place is supposed to call, add
the test that the place calls it** — an AST check on the call site is
legitimate and cheap when running the caller needs Postgres, Neo4j and an
endpoint. See `tests/composition/test_ingest_stats_reach_the_report.py` for the
pattern.

The general habit this project keeps relearning: **break the implementation
on purpose and watch the suite go red before believing it.** Every one of
the defects above was found that way and none by reading.

## The redstring bugs found from here

Finding these is a legitimate output of this project — it is the first serious
external consumer of these code paths.

1. **`BoundaryPreferenceChunker` degenerated to 1-character advances**
   (redstring PR #64, merged `7df622f`). The boundary search only required a
   candidate lie after the current window's `start`, not after the previous
   chunk's `end`, so once overlap rewound `start` past a spent boundary the
   same boundary was re-found forever. `prime:62382` produced 1,018 chunks
   averaging 147 chars — 3.2× the document's own text. Post-fix: 19 chunks.
2. **`SlidingWindowChunker` emits one fully-redundant tail chunk** for every
   document longer than the window, whenever overlap > 0 (BACKLOG
   B-SLIDING-REDUNDANT-1). Zero overlap is unaffected — which localises it to
   the advance, and makes it plausibly the same off-by-one family as #1.
3. **`EmbeddingProvider` cannot express task prefixes** (above). Design gap,
   not a defect.

Both chunker bugs were found by *looking at the data* — "does any document in
this dataset justify that many chunks?" — not by reading the code.

## Testing

`uv run pytest -p no:randomly` for a focused file; the suite is fast. The
`integration` marker (stark-qa sidecar or a live database) is deselected by
`addopts`; select it explicitly when you mean it.

`redstring`'s `CLAUDE.md` carries a long, hard-won section on **test strength**
— inputs that make two candidate implementations agree, and therefore prove
nothing. **It applies here in full; read it.** The short version, and the
standing instruction:

> **Before trusting a test, break the implementation on purpose and watch it
> fail.** A test that stays green under a deliberate defect is worse than no
> test, because its existence stops anyone writing the one that would have
> worked.

This has already paid out in this repo three times: a toolset aggregation test
whose fixture had one chunk per node (making `max`, `mean` and `sum` agree), a
deep-agent context bound that let a 72,000-character observation through
untouched, and a report-path test that could not distinguish per-agent
filenames from the old shared one.

`tests/test_pre_commit_hook_is_installed.py` is the gate for the gates.

## Architecture contract

`lint-imports` enforces one rule, and it is the important one:

```
stark_bench.agents  may not import  harness | skb | sidecar
```

An agent sees `ports.py` and nothing else. That is what makes the agent seam
real rather than decorative — an architecture that could reach into the
harness would be measuring the harness. A new agent that needs something from
`harness` is telling you the *toolset* is missing a capability; add it to
`Toolset`, not an import.

The sidecar is the other hard boundary: `sidecar/*.py` runs under Python 3.11
via `uv run --no-project --python 3.11 --with stark-qa --with "numpy<2"` and
**must import nothing from `stark_bench`**. It is invoked by file path, not
`-m`, because `--no-project` makes module invocation impossible. `numpy<2` is
required: `stark_qa` pulls `ogb` → `rdkit`, which crashes on NumPy 2.x.

`Evaluator` requires `candidate_ids` — it computes `max(self.candidate_ids)`
and cannot be constructed without them.

## Quality gates run on commit

`pre-commit` runs whitespace/EOF/YAML/TOML checks, `ruff check --fix`,
`ruff format`, and `lint-imports` on every `git commit`. **Do not run them
separately first** — that duplicates the hook's work. Write, then commit; the
hook fixes much of it in place (re-`git add` and commit again when it does).

Tests are *not* in the hook. Run them yourself as you work.

Prefer many small commits. Stage **explicit paths, never `git add -A`** — the
working tree routinely carries the `pyproject.toml`/`uv.lock` redstring path
switch, and a parallel agent's in-flight edits.

## Deferred work goes in BACKLOG.md — always

Same rule as redstring, same reasoning. Anything noticed and not fixed lands
in `BACKLOG.md` **in the commit that passes it by** — not a TODO, not a PR
comment, not a sentence in chat. Name the file and line, say what is actually
wrong, and say what you learned that made deferring right. Delete the entry in
the commit that fixes it.

Nineteen entries open at this handoff; `grep '^## ' BACKLOG.md` for the
list rather than reading a copy of it here, which goes stale the moment
anyone closes one.

The ones that shape what you can do next: **`B-EMBED-RETRY-1`** and
**`B-EMBED-COLDSTART-1`** both need a forced model swap on the shared GPU
to diagnose, and both warn that adding retry without measuring would be a
no-op that looks like a fix — the openai client already retries every
`>=500`. **`B-DEEP-EDGES-1`** gates any `deep` number.
**`B-LLM-RUN-NOISE-1`** governs how precisely any LLM number may be
quoted.

## Before running the deep agent

`B-DEEP-EDGES-1`. An ingest without `--ingest-edges` leaves `neighbors` and
`relationships` returning empty, and a low deep-agent score against that
corpus is a data finding wearing an architecture finding's clothes.

Loading them afterwards **skips the embedding, not the edge load**, and the
reason is worth knowing rather than rediscovering: `skb/ingest.py:162`'s
resume path returns `None` for the vectors and nothing else — the entity is
still built, still batched, still added to `known`. So `--ingest
--ingest-edges` with resume at its default re-upserts entities without
embedding a character, then loads the edges. The entities being present is
also what keeps `upsert_relationships` from raising `MissingEntityError` on
the first edge.

**The edge load itself is not minutes.** PRIME is 8,100,498 relationships and
takes **~28 minutes** into a fresh graph — measured on 2026-08-19, where it
was 28.5 of the qwen-wholedoc arm's 100.5-minute total. An earlier version of
this section said "minutes, not a re-ingest"; that was true of the phase it
was written about (a resume whose entities already existed) and badly wrong
as a general claim. Rate is roughly 200-280k relationships/min and eases as
the graph grows, since Neo4j maintains indexes against a larger set.

Budget for it, and watch the right line. The edge loop reports
`edge progress: N relationships, R/s` every 30 seconds and `edges done`
when it finishes, and the node phase now ends with `ingest **nodes** done`
rather than `ingest done`.

That wording is load-bearing. The phase used to log nothing at all after a
line reading `ingest done`, and ~28 minutes of silence following a claim
that the work had finished produced a confident wrong diagnosis of a hang
(B-EDGE-PROGRESS-1, now closed). Every signal checked at the time was a
misread of a healthy run -- flat chunk count, unchanged client CPU, a
`/slots` snapshot. **A snapshot cannot distinguish stalled from busy; only
a window can**, which is why the edge line carries a rate.

Check two things afterwards, neither of which the ingest will volunteer:
`edges` matches `edges.jsonl`'s line count minus `self_loops_dropped`, and
the Neo4j relationship count matches it too — **scoped to your arm**, since
every config's edges share one graph and a bare `count(r)` sums all of them.

`self_loops_dropped` is expected to be **0** on the current PRIME export.
This section previously said a zero more likely means the loader stopped
looking; that was checked on 2026-08-19 and is wrong for this data —
`data/prime/edges.jsonl` contains no self-loops at all across all 8,100,498
lines. A zero is the correct answer here, and `edges` matching the line count
exactly is the check that actually has teeth.

Note also that `runner.run` holds **one agent for the whole query set**. A
`DeepAgent` carrying a single `Budget` would spend the entire allowance on
query 1 and return nothing for the other 279 — near-zero score, no error
anywhere. `PerQueryDeepAgent` in `harness/agents.py` rebuilds the budget per
`retrieve`; keep it that way.

## Still to do

1. ~~**Run the four built-but-unrun arms.**~~ **Done 2026-08-21.**
   `rerank40titlerelmatrix` 0.41771 is the best lean arm; `rerank80titlerelranked`
   0.40486; `rerank40titlereldense` 0.37999 and `rerank40titlerelhybrid`
   0.37984 both lose to BM25-ranked selection and are within 0.0002 of each
   other, answering FINDINGS 1b: an embedding selector does not beat BM25.
   See FINDINGS §10.
2. ~~**Score `qwen-rel-sliding1k` on `dense` and `hybrid`.**~~ **Done
   2026-08-21**, and the config's prediction held: dense 0.25319 against
   `qwen-rel-whole`'s 0.18269, hybrid 0.34675 against 0.27711. Chunking the
   relational corpus helps both channels, not just the lexical one.
3. **`zero_shot` and `deep` have never been scored on a LIVE corpus.** This
   entry used to say they had never been scored at all, which is wrong:
   five results exist, on the three Nemotron arms, whose ingests carried
   all 8,100,498 edges -- so `deep` had its `neighbors` tool and
   B-DEEP-EDGES-1 was satisfied for those runs. They cannot be reproduced,
   because those corpora are gone.

   What they measured is worth keeping, because it is not encouraging for
   the agentic architectures:

   | corpus | dense | hybrid | zero_shot | deep |
   |---|---|---|---|---|
   | `native-wholedoc` | 0.2163 | 0.2187 | 0.2139 | 0.1851 |
   | `native-sliding1k` | 0.2125 | 0.2211 | 0.2199 | 0.2015 |
   | `redstring-native` | 0.1845 | 0.1985 | 0.1998 | — |

   `zero_shot` costs one LLM call per query and matches or slightly
   trails `hybrid`, which costs none. `deep` costs ~7.46 LLM calls per
   query -- against a cap of 8, so most queries ran to exhaustion -- and
   is the **worst** arm on both corpora where it ran, below even
   `lexical` on one. Reranking, which is not agentic, beats everything
   here by a wide margin (`native-wholedoc` `rerank` = 0.3408).

   So the open question is not "do they work" but "do they work on the
   rel corpora", where retrieval is far stronger and `deep`'s
   `neighbors` tool has relational text to walk. That needs
   `--ingest-edges` on a live tenant first: the qwen arms were ingested
   without edges.

4. Whole-branch review, then `superpowers:finishing-a-development-branch`.

When reading the chunking sweep, hold one caveat: its three points are
**1.06, 1.14 and 1.94 chunks/node**, and the first two are closer together
than intended — the whole-document cap had to drop to 5000 characters to fit
a 2048-token ubatch. A null result between `native-wholedoc` and
`redstring-native` is therefore weaker evidence than a null result between
either and `native-sliding1k`.

And hold one about the model: the Nemotron GGUF is **Q4_K_M with no
importance matrix and no MTEB evaluation**, and NVIDIA's own quantised
release is NVFP4 with quantisation-aware distillation done specifically to
recover long-sequence retrieval accuracy. Every Nemotron number here is a
number for this quantisation, not for the model.

Deferred by explicit decision: the **extraction track** — ingesting raw
documents through redstring's own extraction pipeline rather than loading
STaRK's pre-built SKB, on a ~2k-node subset with all configs re-scored against
the same restricted candidate set.

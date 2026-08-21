# Backlog

Deferred work, one entry per item. Delete an entry in the commit that fixes it.

## B-BUDGET-CAPS-1: the per-query budget caps are constants, not config

`MAX_TOOL_CALLS`, `MAX_LLM_CALLS` and `MAX_SECONDS` in
`src/stark_bench/composition/agent_registry.py` are module constants
(8/8/60s). They are the single biggest lever on what a `deep` number means
and they are not in `RunConfig`, so they are not in `config_verbatim`.

**Narrowed 2026-08-20, not closed.** The reports now record the caps that
actually ran, read off the agent (`budget_max_tool_calls` and friends in
`cost`), alongside `exhausted_queries`. That closes the half that mattered
most: a cut-off count without its cap beside it is half a fact, and the
artefacts carried the numerator only.

What remains is that the caps cannot be SET per config -- only observed.
Still deliberate: a config field nobody sets is its own kind of noise, and
the caps have never been tuned. The moment anyone wants two `deep` arms at
different budgets in one sweep, they belong in `RunConfig`, and the
recording added today is what will make those two arms readable.

## B-DEEP-EDGES-1: `deep` against an edgeless corpus measures nothing useful

`--ingest-edges` defaults off, and both existing ingests
(`results/*.ingest.json`) were run without it. The `deep` agent's `neighbors`
and `relationships` actions go to the graph store, so running it now yields
an agent whose traversal always returns empty -- a low number that looks like
an architecture finding and is a data finding. Re-ingest with
`--ingest-edges` before reporting any `deep` number, or state clearly that
the number is traversal-free.

**It is not a re-ingest, and that is worth knowing before anyone budgets
one.** Read `skb/ingest.py:162`: the resume path returns `None` for the
vectors and nothing else -- the entity is still built, still appended to
`batch`, still added to `known`. So `--ingest --ingest-edges` over an
already-ingested corpus, with resume left at its default, re-upserts
entities without embedding a single character and then loads the edges.
Minutes, not the two hours the word "re-ingest" implies. The entities being
present is also what keeps `upsert_relationships` from raising
`MissingEntityError` on the first edge.

Two things to check when it runs, neither of which the ingest itself will
tell you: that `self_loops_dropped` is non-zero (STaRK's PRIME has them, so
a zero is more likely a loader that stopped looking than a clean corpus),
and that `edges` matches the line count of `edges.jsonl` minus those drops.

## B-SLIDING-REDUNDANT-1 — SlidingWindowChunker emits one fully-redundant tail chunk

`redstring/extraction/chunkers/sliding_window_chunker.py`. For every document
longer than the window, the last chunk is entirely contained in the one before
it. The penultimate chunk already reaches end-of-text; the loop advances one
more stride and emits a subset.

Measured, window=1000 overlap=500, always exactly one redundant chunk:

```
len   chunks  last three spans
1001    3     (0,1000)    (500,1001)  (501,1001)
2600    6     (1500,2500) (2000,2600) (2100,2600)
5000   10     (3500,4500) (4000,5000) (4500,5000)
```

Reproduce:
```python
from redstring.extraction.chunkers.sliding_window_chunker import SlidingWindowChunker
c = SlidingWindowChunker(default_chunk_size=1000, default_overlap=500)
[(x.start_char, x.start_char + len(x.text)) for x in c.chunk("x" * 2600).chunks]
```

Not fixed here because this is a redstring defect, not a stark-bench one, and
the `native-sliding1k` ingest was already sized and queued against the current
behaviour. Cost in this benchmark: ~18.6k of ~250.7k chunks (7%) are wasted
embeddings. It is waste rather than corruption -- the text is genuine -- so it
does not invalidate the sweep, but note that chunk COUNT for that config
overstates distinct coverage by ~7%.

Worth checking whether the loop-termination condition here is the same shape
as the boundary-preference bug fixed in redstring PR #64 (that one compared a
candidate against the current window's `start` rather than the previous
chunk's `end`). Same file family, same class of off-by-one on the advance.

Fix belongs upstream in redstring, with a test asserting no chunk's span is
contained in another's.

## B-MODEL-IDENTITY-1 — the served model is taken on the endpoint's word

`src/stark_bench/composition/cli.py:_table_for` and every `model:` string stored
next to a vector are derived from `config.embeddings`, which is a name *we*
write in a YAML file. Nothing checks it against what the server actually
loaded.

This is not hypothetical. On 2026-08-19 the endpoint was swapped from
`nomic-embed-text-v1.5` to `Nemotron-3-Embed-1B` while llama-swap continued
to advertise the model id `nomic-embed-text`; `/v1/models` reported the old
name and only `/props` on the peer port (8082, not the 8080 llama-swap
front) revealed `nemotron-3-embed-1b-q4_k_m.gguf`. Had the two models shared
a dimension, an entire corpus would have been embedded by one model, stored
in a table named for another, and labelled with the wrong provenance -- and
nothing would have raised.

What saved it was accidental: redstring's `LangChainEmbeddingProvider`
checks the returned width against the declared `dimension` on the first
`embed` and raises `EmbeddingProviderError` naming both, and 768 != 2048.
That guard is real and worth relying on, but it is a *dimension* check
standing in for a *model identity* check, and it fails open for any two
models of equal width -- which, at 768 and 1024, is most of them.

Why this was deferred rather than fixed: there is no portable route to the
truth. `/props` carries `model_path` but only on the llama.cpp peer itself;
llama-swap exposes neither `/props` nor `/upstream/{model}/props` on its
front port, so a preflight would have to be told the peer's address, which
is a second piece of endpoint topology in the config for a check that only
works against this one deployment. A stronger and portable option exists
and is the one to build: embed a fixed canary string at ingest, store the
vector alongside the run report, and refuse to query a store whose canary
does not reproduce within cosine 0.99 (per redstring's port docs, vectors
reproduce in direction, not bit-for-bit). That catches a model swap, a
quantisation change, and a pooling-flag change, none of which any name
comparison can see.

## B-CORESIDENCE-1 — RESOLVED: both models are resident together

**Resolved 2026-08-19.** The embedding endpoint now runs concurrently with
the chat model — confirmed operationally, not inferred — so there is no
swapping and the LLM arms are runnable. The analysis below is kept because
it is why the endpoint is configured the way it is, and because the
prediction it makes about the fix held.

Two things downstream of this were wrong while it was open and have been
corrected: `composition/cli.py` carried a comment claiming one model is
resident at a time, and run queues were ordering `rerank` last to avoid a
thrash that cannot happen. Neither is true.

The 36.9s cold-start latencies seen during ingest are a *first load* after
the model has been evicted for idleness, not swap churn between models.

### Original entry

`zero_shot` and `deep` cannot run against the current endpoint. Both search
with text the LLM produced moments earlier -- `zero_shot` rewrites the query
(`agents/zero_shot.py:42-51`) and `deep` searches on `step.argument` chosen
per round (`agents/deep.py:101,127`) -- so the embedding model has to answer
*during* the agent loop, interleaved with chat completions.

Nemotron-3-Embed-1B does not fit in VRAM beside `qwen3.8-27b-mtp`, so
llama-swap unloads one to serve the other. `deep` is budgeted at 8 LLM calls
and 8 tool calls per query, which is up to 16 alternating swaps per query,
280 queries, three configs. That is not a slow run, it is a non-starter.

The obvious workaround does not work, and it is worth writing down so nobody
spends an afternoon on it: precomputing the 280 query vectors while the
embedder is loaded and serving them from `PrecomputedEmbeddingProvider`
covers `dense` and `hybrid` exactly, and covers neither LLM agent at all,
because neither one ever embeds the original query text.

**Resolved in approach, not yet run.** The embedding model is not what
fills the VRAM -- it is ~700MB at Q4_K_M. The KV cache is: 32 slots at 4096
tokens each. Dropping the embedding server to `-np 1` with a 4096-token
context shrinks that cache by 32x and both models fit, with no swapping, no
weaker chat model, and no loss of comparability against the `dense` and
`hybrid` arms.

It costs nothing for this workload, which is the part worth noticing: the
agent loop embeds one short query at a time and waits for it, so 31 of the
32 slots were never going to be used during an LLM run. High concurrency is
an *ingest* setting. The two phases want opposite server configurations, and
the plan is now to run them as two phases:

  - ingest and the `dense`/`hybrid` scoring at `-np 32`, embeddings alone;
  - the `zero_shot`/`deep` scoring at `-np 1`, both models resident.

Rejected, and worth recording so they are not retried: precomputing the 280
query vectors covers neither LLM agent, because neither embeds the original
query text; a smaller chat model would have made "deep beat dense" mean
something different from what it says; and accepting the swap cost on a
reduced subset would have produced a number comparable to nothing else here.

Nothing here is blocked on it: the control plus three arms times
`dense`/`hybrid` is seven of the numbers, and none of them make an LLM call.

## B-EMBED-RETRY-1 — a transient 503 from the embedding server kills a whole ingest

`src/stark_bench/skb/ingest.py` makes embedding calls through redstring's
`EmbeddingProvider` and lets `EmbeddingProviderError` propagate. Restarting
llama.cpp mid-run therefore ends the ingest, because the server answers
`503 {"message": "Loading model"}` for the tens of seconds it takes to load
weights onto the GPU.

Observed 2026-08-19: a run at 95,277 of 136,772 chunks died on exactly this
while the server was restarted to change `--ubatch-size`. The log holds
seven `503`s, so it was retried at the HTTP layer by the OpenAI client and
still gave up well inside the load window.

Resuming makes this survivable rather than harmless, and that is why it has
not been fixed: nothing is lost, the run just has to be relaunched by hand,
and `scripts/sweep.sh` already retries an arm up to `MAX_ATTEMPTS=5`. The
gap is the *manual* path — a bare `--ingest` invocation has no retry at all,
which is the path used for every throughput probe.

What to do:

- retry `503` and connection errors with a backoff long enough to cover a
  model load (tens of seconds, not the OpenAI client's default), and
  distinguish them from a `400`, which means the batch is too big and
  retrying it forever is the wrong answer;
- decide where it belongs. Wrapping it here keeps redstring's adapter
  honest about what the server said; putting it in redstring means every
  consumer gets it. The adapter already raises a typed
  `EmbeddingProviderError`, so a caller *can* distinguish these — it just
  has to parse the message, which argues for a status code on the error
  rather than a retry loop in this repo.

## B-EMBED-COLDSTART-1 — one embedding timeout kills a two-hour ingest

`src/stark_bench/adapters/stark_ingest_engine.py:251` gathers a wave of embed
calls and lets any `EmbeddingProviderError` propagate straight out of
`ingest_corpus`. There is no retry anywhere on the path, so a single failed
request discards the run.

The failure is not hypothetical and it is not the endpoint being down, which is
what it was misdiagnosed as twice. The inference host is llama-swap, which
unloads the embedding model when a chat model is used. The first request after
a swap pays the model load: measured **36.2s cold, then 0.74s and 0.14s**.
At `--embed-concurrency 32` the whole wave queues behind that cold load and
exceeds the proxy's 60s header timeout — which is why it died ~3 minutes in
rather than at request one, and why isolated batch-size probes (`batch=32` in
6.8s) all looked healthy and sent the diagnosis to the wrong place.

Worked around by warming the model with a single embed before launching and
dropping to `--embed-concurrency 12`. That is a smaller queue, not a fix: any
mid-run swap re-arms the same failure.

Fix is retry with backoff around the wave, treating a timeout as retryable.
Note it belongs in `redstring`'s `llm/adapters/langchain_embedding.py` rather
than here if the retry should benefit every caller — decide which before
writing it.

**Do not "add retry" here without measuring first.** The `openai` client under
`OpenAIEmbeddings` already retries 408, 409, 429 and every >=500, and defaults
to `max_retries=2`. Our failure was a 500, so retries were almost certainly
already firing and a hand-written retry loop would be a no-op that looks like a
fix. `redstring`'s adapter argues this deliberately: "the caller constructs the
LangChain object, so a deployment's own retry, callback and tracing
configuration is not something this class must mirror."

The likelier mechanism is that all three attempts expire *inside* the cold
load: 36s to load, a 60s proxy header timeout, no backoff long enough to
outlast it, and a wave of concurrent requests keeping the queue saturated. If
so the lever is client `timeout` and backoff, exposed through
`openai_compatible`, not retry count.

Deferred rather than guessed because distinguishing the two needs a forced
model swap on the shared GPU, which was busy. Reproduce by issuing a chat
request to evict the embedding model, then firing N concurrent embeds and
logging per-attempt latency.

Ingest is resumable (`loaded N existing chunk ids for tenant`), so the cost of
a crash is the wave in flight, not the run. That is what makes this a backlog
item and not a blocker.

## B-EPHEMERAL-STORES-1 — the stores had no volumes, and an ingest was lost

`docker-compose.yml` declared no `volumes:` for either service, so Postgres
and Neo4j wrote to anonymous volumes that are destroyed with their container.
A `docker compose down`-shaped event at 2026-08-19T22:08:51Z removed both
containers and the `stark-bench_default` network, taking 589,790 embedded
chunks across four tenants with them.

Fixed here by adding named volumes (`stark-pgdata`, `stark-neo4jdata`). What
is still open is the detection gap, which is the part that cost time:

- **Nothing announced the loss.** The next run failed with
  `ConnectionRefusedError` on 55432, which reads as "the container is down",
  not as "the corpus is gone". Those need different responses and looked
  identical.
- **A surviving container with an empty store would have been worse.** The
  connection error at least failed loudly; had the stack been restarted first,
  the queue's ingest gate would have passed on a fresh empty corpus and the
  arms would have scored low-but-plausible numbers. That is the same silent
  degradation shape as the stale model id and the three-valued rerank scores.

So the fix worth adding is a preflight that asserts the configured tenant's
chunk count is non-zero (or that ingest is being asked for), rather than
letting an empty store look like a bad retriever.

Note what did NOT need recovering: `results/*.json` and the persisted
predictions are files in the repo, so every scored number survived intact.
Keep expensive-to-recompute artifacts out of the containers.

## B-NOMIC-CONFOUND-1 — nomic vs Nemotron varies two things at once

`nomic-wholedoc` was built to isolate the embedding model against
`native-wholedoc`, and it does not. nomic's context ceiling is 2048 tokens
against Nemotron's 4096, so it cannot run `capped-whole-5000` -- the server
rejects chunks over the limit -- and runs `capped-whole-2400` instead.

The chunker is the second-largest retrieval effect in RESULTS.md (finding 4:
a 25% spread in dense mrr across four corpora), so the comparison now varies
the model and the chunking together and a gap cannot be attributed to either.

This entry originally argued the penalty was *negative and monotonic in
granularity*, citing a RESULTS.md finding that has since been retracted --
`native-sliding1k` has twice the granularity of `redstring-native` and scores
15% better. The correct statement is that the direction of the 5000 -> 4000
change is unknown. That is a weaker claim and still sufficient: an unknown
effect of unknown sign sitting on top of the model swap is exactly what makes
the comparison unattributable.

To make it clean, re-run Nemotron at `capped-whole-2400` and compare that to
`nomic-wholedoc`. Not done here because Nemotron embeds at 18 texts/s against
nomic's 60, so the PRIME corpus is ~2.3h against ~40min, and the swap to
nomic was already justified on the ada-002 comparison
(0.2163 vs 0.2306 mrr) which is unaffected by this.

What is NOT confounded, and is the reason the MAG run exists: PRIME against
MAG, both on nomic at `capped-whole-2400`. That comparison holds the model
and the chunker fixed and varies only the corpus.

## B-TOKEN-CAP-1 — RESOLVED by catch-and-re-split

**Resolved 2026-08-20.** The cap is no longer required to be right. When the
provider rejects a text for length, `stark_ingest_engine` re-chunks that
group at half the size and retries, up to `MAX_RESPLIT_ATTEMPTS`. A correct
cap costs nothing, because the path only runs on rejection.

Option 1 from the original entry (cap by tokens with nomic's vocabulary) was
NOT taken, and the reason is worth keeping: it needs a `tokenizers`
dependency and a downloaded vocabulary, and it would put a *second* estimate
of the model's tokenization next to the server's real one. `all-MiniLM-L6-v2`
is in the local HF cache and shares BERT WordPiece, so it was available as a
stand-in -- and using it would have been the same "close enough" reasoning
that produced the three wrong caps. The server's own 400 is the only oracle
that cannot disagree with the server.

`/tokenize` was probed first and is not routed by llama-swap; only the
`/v1/*` surface is reachable.

Still worth doing eventually: the cap now sits at 2400 characters, which is
67% of the ceiling at the measured worst ratio, so every document pays a
granularity cost for a tail of a few hundred. Token-exact chunking would
recover that. It is an optimisation now rather than a correctness fix.

### Original entry

`CHUNKERS["capped-whole-2400"]` exists because nomic-embed-text rejects
anything over 2048 tokens and the chunker measures characters. The conversion
is an estimate, and it was wrong three times running: 5000 chars (from 4.0
chars/token, measured on chat prompts through a different tokenizer), then
4000 (from 2.4, the ratio the first failure implied), then 2400 (from 1.754,
measured over the 250 densest of 607,292 documents).

Each estimate was defensible and each was too high, because the worst case
lives in a tail that sampling keeps missing. 2400 has a large enough margin
to survive ratios down to 1.17 chars/token, which is why it is expected to
hold -- but it is still a guess with a bigger cushion, not a fix.

Two real fixes, either of which ends it:

1. **Cap by tokens.** Load nomic's WordPiece vocabulary with `tokenizers`
   and split on token count. Exact, and it lets the cap sit near 2048 rather
   than at 67% of it, which recovers the chunks/node the margin costs.
2. **Catch and split.** The server returns a specific, machine-readable 400
   (`exceed_context_size_error`, with `n_prompt_tokens`). Catching it and
   re-splitting just that chunk makes any cap safe.

(1) is better: it keeps failures out of the hot path and makes chunks/node
predictable. (2) is a smaller change and would also have saved the three
ingests lost to this.

Cost of not doing it: each wrong cap costs a full re-ingest, ~30 min for
PRIME at 218 chunks/s and ~1h for MAG.

Note the cap also widens B-NOMIC-CONFOUND-1: nomic now runs at 2400
characters against Nemotron's 5000, so the two arms differ more in chunking
than the original swap intended.

## B-QWEN-UNCAPPED-1: qwen's whole-document arm is capped for a reason that is not qwen's

`config/qwen-wholedoc.yaml` runs `capped-whole-2400`. qwen3-embedding-0.6b
does not need a cap at all -- it is served with `n_ctx 32768` and took a
40,000-character input whole when probed against the live endpoint, and
PRIME's longest document is 52,260 characters with a mean of 870. Every
document but a handful would come through in one chunk.

The cap is nomic's, kept so that `qwen-wholedoc` minus `nomic-wholedoc` is
the embedding model and nothing else. That is the cleanest single-variable
model comparison this repo has had, and it is worth the cost -- but the cost
is real: no arm here measures qwen at the granularity it can actually run.

The follow-up is a `qwen-uncapped` cell at `whole-document` (or a
10,000-character cap for the 52k outlier), scored against `qwen-wholedoc`.
That difference is granularity at a fixed model, which is the same question
the chunking sweep asks and would extend it to a fourth point at 1.00
chunks/node. Deferred because the four arms already queued are ~6h of
endpoint time on a single-slot server and this one adds a fifth without
answering anything the sweep does not already ask.

## B-PROXY-LIMITS-1: the embedding batch ceiling is the proxy's, not the model's

`--embed-batch 512 --embed-concurrency 2` against whole `prime-rel`
documents dies with:

```
openai.InternalServerError: peer proxy error:
net/http: timeout awaiting response headers
```

and `--embed-batch 128 --embed-concurrency 4` or `--embed-batch 256` return
`502 Bad Gateway`. Measured 2026-08-19 against llama-swap in front of
qwen3-embedding-0.6b. Safe settings for whole documents are **batch 32-128 at
concurrency 2**; the same 512 was fine on the capped 2,400-char corpus, so the
ceiling is bytes-per-request, not texts-per-request.

Two things make this worth an entry rather than a note:

**The engine's re-split does not catch it.** `MAX_RESPLIT_ATTEMPTS` fires only
when `_is_oversize(error)` matches -- an input longer than the context. A
proxy timeout and a 502 are not oversize errors, so they propagate and kill
the ingest. The re-split was built for B-TOKEN-CAP-1 and correctly does not
guess at transport failures, but the result is that the one obvious safety net
does not cover the failure mode most likely to be hit on a large-document
corpus. A bounded retry on 502/timeout, halving the batch, would.

**It is a first-batch failure, which is the good case.** Zero rows had landed,
so there was nothing partial to reason about. A proxy that failed 80% of the
way in would leave a corpus that resume treats as complete for everything
already written -- and no stage would report anything wrong.

## B-RATE-UNIT-1: extrapolate ingest time by characters, not documents

A `--limit 3000` calibration on `prime-rel` read 333 nodes/min and implied a
6.5-hour arm. The real figure is ~2 hours. Both numbers are correct; the
extrapolation was not.

Two compounding reasons, and the second is the general one:

**The head of the file is not the corpus.** The first 1,024 documents of
`prime-rel` average **5,278 characters** against the corpus mean of **1,761** --
they are gene/protein records with long summaries. `native-rel-whole.yaml`
already records a sample of the first 7,548 nodes implying 3.4x when the truth
was 1.47x, and this is the same trap in a different measurement. `--limit N`
always samples the head, so it can calibrate a *rate* but never a *total*.

**The rate is token-bound, so documents are the wrong unit.** Batch 32 and
batch 128 give 368 and 366 docs/min on the identical slice -- a 4x change in
requests moves throughput by 0.5%. The stable figure is **~1.94M chars/min**,
and `227.9M / 1.94M = 117 min` predicted the arm correctly where docs/min was
out by 3.3x.

So: measure chars/min on whatever slice is convenient, then divide the
corpus's total characters by it. Quoting nodes/min from a `--limit` run and
multiplying by the node count is wrong twice over.

**The ingest's own ETA has this bug.** `_report` in
`src/stark_bench/adapters/stark_ingest_engine.py` extrapolates from nodes
done against `total_nodes`, so on a corpus with a long document tail it is
wrong in the alarming direction and gets worse as the run proceeds. Measured
live on `qwen-rel-whole`, 2026-08-19:

| | engine ETA | measured |
|---|---|---|
| 16 min in | 152 min | -- |
| 22 min in | 180 min | -- |
| 25 min in | **198 min** | **~66 min** |

At that last point the arm was 11% through its *nodes* and **28.9% through
its characters**, because the chunks then being embedded averaged 14,241
characters against the corpus mean of 1,761. The rate was a healthy 2.37M
chars/min throughout; only the unit was wrong.

An ETA that climbs while the run is healthy trains its reader to ignore it,
and this one nearly caused a second false hang diagnosis in the same session
as B-EDGE-PROGRESS-1. The fix is to accumulate `sum(len(text))` alongside the
chunk counter and extrapolate against the corpus's total characters -- one
extra pass over `nodes.jsonl` at startup, the same place `_count_lines`
already reads it.


## B-ADA002-PROVENANCE-1: we do not know what STaRK's precomputed vectors were built from

`vss-control` scores `data/prime` text against `doc_emb.npz`, STaRK's
published ada-002 embeddings. **Nothing here establishes what text those
embeddings were computed over**, and the answer changes what every comparison
to that row means.

Checked in `stark_qa` (installed in the 3.11 sidecar) on 2026-08-19:

- `models/multi_vss.py:90` -- `get_doc_info(node_id, add_rel=True, compact=True)`
- `models/llm_reranker.py:93` -- `get_doc_info(node_id, add_rel=True)`
- `models/bm25.py:29` -- `get_doc_info(idx)`, i.e. the `add_rel=False` default
- `models/vss.py` -- loads `candidates_emb_dir` and generates nothing

So the two baselines that build text at run time disagree with each other,
and the one we replicate does not build text at all. The generation script is
not in the pip package.

Why it matters: `qwen-wholedoc` dense (0.183) was read against `vss-control`
dense (0.231) as evidence that qwen3-embedding-0.6b is a weak model, and that
reading is sound only if both indexed comparable text. If STaRK used
`add_rel=True`, the gap is largely corpus and the honest comparison is
against `qwen-rel-whole`. If `add_rel=False`, ada-002 really is better here
and the model conclusion stands.

Two ways to settle it, in order of cost:

1. Read STaRK's `emb_generate.py` in the GitHub repo (not shipped to PyPI).
2. Measure it from our side, which is already running: if `qwen-rel-whole`
   moves substantially above `qwen-wholedoc`, relational text is worth a lot
   on this benchmark whatever STaRK did -- and if it does not, the question
   stops mattering for our conclusions.

Until then `RESULTS.md` labels the `vss-control` rows a reference point of
uncertain provenance, which is the honest framing and not a placeholder for
one.


## B-QUADRATIC-DOCS-1: whole-document embedding cost is superlinear in document length

`qwen-rel-whole` (129,375 documents, 227.9M chars, `whole-document`) is
running at roughly **0.32% of corpus characters per minute sustained**, or
~720k chars/min. A probe on a 1,024-document slice averaging 5,278 chars
measured **1.94M chars/min** -- 2.7x faster per character on shorter
documents.

Longer documents are more efficient per *request* and much less efficient per
*character*, because transformer attention is quadratic in sequence length. A
30k-token document is not 10x a 3k-token one, it is closer to 100x. `prime-rel`
has 6,162 documents over 8,000 characters (4.8%) and a maximum of 133,778, and
that tail dominates the arm's wall time.

Consequences worth acting on:

- **A chars/min figure is only valid for the length distribution it was
  measured on.** B-RATE-UNIT-1 says extrapolate by characters rather than
  documents, which is right and still not sufficient -- the rate itself moves
  with document length. Measure on a representative slice, not the head
  (dense) and not a random 1k (short).
- **`capped-whole-8000` would likely cost a fraction of `whole-document` on
  this corpus** while splitting only the 4.8% over that length. Whether that
  is a good trade depends on where the `- relations:` block falls, since it
  sits at the END of the document and a cap severs the neighbour names first.
  That is a real experiment, not an obvious win.
- The **embedding cache** (shipped today) does not help a first arm at all.
  It helps the second and third, which is where the sweep cost lives.

Not filed as a defect -- nothing is wrong -- but the cost model this project
has been reasoning with ("chars/min is a constant") is wrong, and it produced
two ETAs today that were out by 3x in opposite directions.

## B-QUERY-LATENCY-SPLIT-1 — ANSWERED: dense 0.281s, lexical 0.575s

"How much of a query's 0.28s was the embed?" -- answered from the reports
rather than by instrumenting anything, since `seconds_total` on a
retrieval-only arm IS the retrieval time.

| arm | per query | what it contains |
|---|---|---|
| `dense` | 0.281s | one query embed + pgvector search |
| `lexical` | 0.575s | BM25 over the 5.7M-row terms table, no embed |
| `hybrid` | 0.890s | both, and 0.575 + 0.281 = 0.856 |

Two things worth keeping:

- **Lexical is twice dense.** BM25 over that terms table costs more than a
  vector search plus an embedding round trip. The intuition that the
  network call dominates is wrong here.
- **The channels are additive**, so `hybrid` is not sharing work between
  them. Whether it could is a separate question nobody has asked.

The embed's own share is now smaller than the 0.281s suggests: those
figures predate `PrewarmedQueryEmbeddings`, which batches all 280 query
embeddings into 3 requests before the run and reports
`query_embed_live_calls: 0`. A dense arm re-run today would be nearly all
pgvector. That re-run needs the endpoint and has not happened.

## B-RERANK-RETRIEVAL-FLOOR-1 — CORRECTED: retrieval is ~0.9s, not ~8.8s

**The original claim in this entry was wrong and is worth keeping as an
example of how.** It said: prefill 13,133 tok / 10.56s plus decode 627 tok
/ 9.27s is 19.8s of LLM against 28.6s observed per query, so "~8.8s is
retrieval".

Measured directly from the reports' own `seconds_total`, which is the sum
of tool-call durations and for a retrieval-only arm is exactly retrieval:

| arm | retrieval, per query |
|---|---|
| `dense` (pgvector + query embed) | 0.281s |
| `lexical` (BM25 over the 5.7M-row terms table) | 0.575s |
| **`hybrid` (what the reranker fetches with)** | **0.890s** |

Almost exactly additive: 0.575 + 0.281 = 0.856 against 0.890 measured. The
reranker fetches 40 rather than 20, so its retrieval is somewhat more than
0.890s -- and nowhere near 8.8s.

### The mistake

A residual was computed by subtracting ONE sampled query's LLM time from
the AVERAGE per-query wall time. Those are not the same population.
Candidate documents on this corpus run from 357 to 133,778 characters, so
prompt size varies enormously between queries; the sampled response had
13,133 prompt tokens and was cheaper than the mean. The residual absorbed
that difference and got attributed to retrieval.

This is the mistake CLAUDE.md already records under "two rates from two
different windows are not comparable", in a new costume. **Sample both
sides of a subtraction from the same population, or do not subtract.**

### What still stands

Retrieval IS now a meaningful share: at `rerank40title`'s 1.70s/query with
concurrency 4, roughly 0.9-1.4s of retrieval per query is most of it. The
conclusion "further prompt work has a floor at retrieval" survives -- it was
right for the wrong reason, and the floor is lower than claimed.

## B-RERANK-OUTPUT-ENCODING-1

`agents/rerank.py:TerseRelevance`.

Decode is now the larger half of the LLM cost and is 15.7 tok/candidate to
convey one integer. Measured alternatives, from the observed 627-token
pretty-printed response at fetch=40:

| encoding | tokens | decode |
|---|---|---|
| pretty `{"i": 1, "s": 12}` (current) | 627 | 9.3s |
| JSON pairs `[[1,12],...]` | 208 | 3.1s |
| space table `1 12\n` | ~120 | ~1.8s |

Not done. The space table needs a `complete(prompt) -> str` on `Toolset`
-- neither it nor redstring's `LlmProvider` has a raw-completion method --
and that trades away grammar-constrained decoding for ~0.6s over JSON
pairs. Judged not worth it while retrieval sits at 8.8s
(B-RERANK-RETRIEVAL-FLOOR-1). Revisit if that floor drops.

## B-QUERY-CONCURRENCY-1 — RESOLVED: 2.02x once the slots were real

`--query-concurrency` (68f8d55) runs N queries in flight. This entry
originally recorded that it **bought nothing**, measured at 6.2s/query
serial against 6.43s at concurrency 4 -- 3.7% *worse*, four clients queuing
on one slot.

That was true and is no longer. The chat peer moved to `-np 4` and the same
knob, measured in 90-second windows on `rerank40title`:

| | s/query |
|---|---|
| serial (gemma, `-np 4` server) | ~4.5 |
| `--query-concurrency 4` | **2.22** |

**2.02x, not 4x.** Sublinear, as expected when four decodes share memory
bandwidth -- and exactly the measurement this entry insisted on taking
before anyone claimed a speedup.

Both traps it warned about were real:

- **The flag you edited may not be the one running.** The first attempt was
  against a peer still at `-np 1`; `/props` is not reachable through the
  llama-swap proxy on :8080, so it took the operator checking the peer
  directly to establish it.
- **`-np 4` might not have helped anyway.** It did here, but only 2x, and
  the entry's reasoning about bandwidth-bound decode is what the shortfall
  looks like.

Default stays 1, so no previously-recorded arm's timing moves.

## B-LLM-RUN-NOISE-1

**LLM arms are not reproducible run to run, and CLAUDE.md's standing claim
that "every accuracy number in this repository is a difference between two
runs" needs a noise floor beside it.**

`rerank40title` on `gemma-4-26b-qat`, same corpus, same tenant, same split,
`temperature=0.0`, `enable_thinking: false`, run twice within an hour:

| metric | run 1 | run 2 | delta |
|---|---|---|---|
| mrr | 0.34100392200052737 | 0.3397480349721653 | 0.00126 |
| hit@1 | 0.25714285714285712 | 0.25357142857142856 | 0.00357 |
| hit@5 | 0.43928571428571428 | 0.45 | **0.01071** |
| recall@20 | 0.46431878718686570 | 0.4720568822829851 | 0.00774 |

Temperature zero does not make a batched server deterministic: with `-np 4`
and continuous batching, a request's logits depend on which other requests
share its batch, and floating-point addition is not associative. A handful
of near-tied argmaxes flip and the ranking moves.

**Consequences.** A difference below ~0.001 mrr between two LLM arms is
noise. hit@5 is worse, at ~0.011 -- roughly 2.5% relative -- so it should
not be quoted as a precise figure at all on these arms.

Retrieval-only arms (`dense`, `lexical`, `hybrid`, `vss-control`) are
unaffected: no LLM, and `vss-control` reproducing 0.23057383129905376 to
every digit after a re-ingest remains a valid check.

**Unresolved and cheap to settle:** whether `--query-concurrency 1` restores
determinism. If it does, the cause is confirmed as batch composition and a
reported number can be made reproducible by paying ~4x wall time for it.
Two serial runs of the same arm would answer it.

## B-RERANK-SCORES-DISCARDED-1

`agents/rerank.py:860` -- `_Ranked(node_id=p.node_id, score=1.0 / (1 + rank))`.

The reranker returns a reciprocal-rank placeholder, so **the model's actual
judgements never reach disk**. `write_predictions` persists the placeholder,
and every prediction file therefore shows twenty distinct scores with no
ties, whatever the model did.

Found by measuring quantisation from the prediction files and getting
"20.0 distinct scores, largest tie group 5%" for three arms that should
differ -- a result contradicted by the one raw response we have, where 5
appeared nine times and 8 six times across 40 candidates.

**What this costs.** Every question about the model's scoring behaviour
needs a fresh run and a packet capture: how hard it quantises, whether the
matrix arm's dimensions are actually orthogonal (`matrix_degenerate_rows`
is logged per query but not persisted), whether scores are calibrated, how
often a gold answer was scored highly but still lost. These are cheap
questions about data we already paid for and threw away.

**Why the placeholder is not simply wrong.** It guarantees a strict total
order downstream, and the sidecar sorts by score -- writing raw scores would
make tied candidates order arbitrarily inside the evaluator, changing
results for a reason unrelated to retrieval.

So the fix is to persist the judgements ALONGSIDE, not to change `score`.
That needs a diagnostics channel from agent to harness, which does not
exist: `ToolCall` carries cost and nothing carries per-query observations.
Deferred for that reason -- it is a real design addition, not a one-line
change, and inventing the channel casually is how the agent seam stops
being a seam.

## B-CHUNK-COUNT-OVERSTATED-1 — reported chunks count writes, not rows

`scripts/verify_corpus.py` compares each ingest report's `chunks + skipped`
against `count(*)` for the arm's tenant. Every whole-document and
boundary arm agrees exactly. Both sliding-window arms do not:

| arm | claimed | actual | gap |
|---|---|---|---|
| `qwen-rel-sliding1k` | 549,886 | 549,697 | **189** |
| `qwen-mini-sliding1k` | 24,284 | 24,274 | 10 |

**Mechanism, reproduced exactly.** `stark_ingest_engine.py:507` builds
`id=chunk_id(source_id, piece.text)` -- source and TEXT, with no
`start_char`. A sliding window over repetitive text produces windows whose
text is byte-identical, those share an id, and the upsert merges them.

Re-chunking `prime-rel` offline: 38,964 documents over 1000 characters
produce 459,475 chunks, of which **189 collapse across 78 documents** --
the observed gap, to the unit.

**The dedup is right; the reporting is wrong.** Two identical chunk texts
embed to the same vector and `aggregation: max` takes the best, so a second
copy adds nothing to retrieval and costs a row. Adding `start_char` to the
id would "fix" the count by storing redundant duplicates, which is worse.

So the defect is that `chunks` counts writes ATTEMPTED and is reported as
if it counted rows. `chunks/node` for `qwen-rel-sliding1k` is 4.250 as
reported and 4.249 in fact -- immaterial here, and only immaterial because
the collision rate is 0.034%. A chunker that produced many duplicates would
lose a lot of them silently, and nothing would say so.

Fix: report rows actually present alongside writes attempted, the same
split `seconds_total` and `seconds_wall` now make. Deferred because it
needs the ingest to read back its own tenant count, and the standalone
script already answers the question for anyone who asks it.

Related: B-SLIDING-REDUNDANT-1 is a DIFFERENT defect with a similar smell.
Its redundant tail chunk has a distinct `start_char` but identical text to
part of its predecessor -- not byte-identical to a whole chunk, so it does
NOT collide, and it is still written.

# Backlog

Deferred work, one entry per item. Delete an entry in the commit that fixes it.

## B-SUMMARISE-1: `--summarise` is not implemented

Task 14 step 4 of the plan
(`docs/superpowers/plans/2026-08-18-stark-benchmark-harness.md`) ends with:

```
uv run python -m stark_bench.composition.cli --summarise results/ > RESULTS.md
```

`src/stark_bench/composition/cli.py` has no `--summarise` flag. The four
per-architecture report files now exist (`{config}.{agent}.json`), so the
input to it is in place and the work left is the reading side: load every
`*.json` under a directory that is not `*.ingest.json`, and emit one row per
file with `metrics` and `cost` side by side. Cost belongs in the same table as
accuracy, per the plan -- a number without its cost beside it is not
actionable.

## B-BUDGET-REPORT-1: budget exhaustion is not in the report

`PerQueryDeepAgent.exhausted_queries` (`src/stark_bench/harness/agents.py`)
counts how many queries ended at the cap, and nothing reads it.
`_do_run` in `cli.py` calls `summarise_cost(tools.calls, ...)`, which counts
calls but cannot distinguish "the agent decided it was finished" from "the
agent was cut off". Those are different findings: a deep run where 90% of
queries hit the cap is a run whose accuracy number is about
`MAX_TOOL_CALLS`, not about the architecture.

The fix is small and was left out to keep the wiring commit narrow: thread
the agent back out of `run(...)` (or read it off the built agent, which
`_do_run` still holds) and pass `exhausted_queries` into `write_report`.
`write_report` takes a fixed keyword set, so it needs one more parameter --
that is the only reason this is not a one-liner.

## B-BUDGET-CAPS-1: the per-query budget caps are constants, not config

`MAX_TOOL_CALLS`, `MAX_LLM_CALLS` and `MAX_SECONDS` in
`src/stark_bench/harness/agents.py` are module constants (8/8/60s). They are
the single biggest lever on what a `deep` number means, and they are not in
`RunConfig`, so they are not in `config_verbatim` either -- a deep result
file does not record the budget it was run under. Changing them changes every
past number's comparability with no trace in the artefacts.

`chat_model` was added to `RunConfig` in the same commit and these were not,
because the caps have never been tuned and a config field nobody sets is its
own kind of noise. The moment anyone changes one, they belong in the config.

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

## B-ADA002-TABLE-1: the `vss-control` corpus is orphaned in the pre-rename `kg_chunks` table

`_table_for` in `src/stark_bench/composition/cli.py` derives a per-embedding-model
chunk table (`kg_chunks_precomputed_ada002` for `vss-control`). It landed in
e6411e3 (17:14). The `vss-control` ingest ran at 17:05 and the dense run at
17:08, both against the hardcoded `kg_chunks` of the time -- so
`results/vss-control.dense.json` (mrr 0.23057) was measured against a table
the harness no longer reads.

Today: `kg_chunks` holds 129,375 rows for tenant
`9ef286ae-92c2-5655-8d1a-47a9ff4d0892`, which is exactly
`_tenant_for(vss-control)`. `kg_chunks_precomputed_ada002` holds **zero** rows;
`ensure_schema` creates it empty on every run. Every one of the 280 queries
therefore retrieves nothing, in *all three* retrieval modes -- this is not a
hybrid-vs-dense difference, and any future `vss-control` number is a data
finding until the corpus is back.

Two ways out, both a decision rather than a fix:

- rename the data across (`kg_chunks` also holds 200 rows for a second tenant,
  `5a7ba3cf-...`, and there is a companion `kg_chunks_terms`), or
- re-ingest `vss-control`, which is cheap because its embeddings are
  precomputed -- no endpoint time.

Until then `results/vss-control.dense.json` is not comparable with anything
produced after e6411e3 and should not be quoted beside a post-rename number.

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

## B-RESUME-COMPLETE-1 — nothing records that an ingest *finished*

`scripts/resume_is_safe.py` compares the recorded config to the config on
disk, byte for byte, and refuses on any difference. What it cannot see is
whether the run that wrote that report ever completed.

`IngestOutcome` has no `complete` field, and the report is written once at
the end -- so a killed run leaves either no report at all (resume refused,
correct) or the report of some *earlier* run (resume permitted against a
corpus that earlier run did not finish).

Observed 2026-08-19: tenant `c507d57b` (`native-sliding1k`) holds 7,754
chunks from a run killed hours earlier, against ~129,375 nodes.

This is currently harmless and the reason is worth writing down, because it
is what makes the fix low priority rather than urgent. Nothing in this
codebase deletes chunk rows, and chunk ids are content-addressed over
`(source, text)`, so a later run with the *same* config upserts over the
partial rows and converges on the right corpus. The hazard is only a
partial corpus plus a *changed* chunker: then the old ids are not rewritten,
stay live, and answer queries alongside the new ones.

That is the same silent-mixture failure `resume_is_safe.py` exists to
prevent -- it just arrives by a route the guard does not check.

What to do: add `complete: bool` to `IngestOutcome`, write the report once
before the load with `complete=False` and again after with `complete=True`,
and have `resume_is_safe` require it. Writing it twice is the point; a
single write at the end cannot distinguish "did not finish" from "never
started".

Also worth a count assertion: the report records `nodes`, so a resumed run
can compare the chunk rows actually present for its tenant against what the
report claims and refuse on a mismatch.

## B-SIDECAR-RESOLVE-1 -- scoring resolves from PyPI on every run

`adapters/stark_scorer.py:83` shells out to
`uv run --no-project --python 3.11 --with stark-qa --with numpy<2`, which
re-resolves 166 packages every time anything is scored. With a warm uv cache
that is 114ms and invisible. With a cold one, or with PyPI degraded, it is a
hard failure *after* all retrieval has been paid for.

That is not hypothetical: on 2026-08-19 `redstring-native/deep` finished 280
queries in 46 minutes of shared GPU and then died on

    502 Bad Gateway ... anthropic-0.124.0-py3-none-any.whl.metadata

`anthropic` is a transitive dependency of `stark-qa` that this sidecar never
imports, so the run was lost to a package it does not use.

`write_predictions` (this commit) stops the loss being total -- retrieval now
lands on disk before anything scores it, and `scripts/rescore.py` finishes the
job later. The resolve itself is still a network dependency in the middle of
every run.

What to do: build the 3.11 environment once, into a checked-in path
(`uv venv --python 3.11 .sidecar-venv` plus `uv pip install`), and invoke that
interpreter directly instead of `uv run --with`. Then a scoring run touches no
network at all. Do not simply add `--offline`: it makes the *first* run on any
machine fail instead, which trades a rare failure for a certain one.

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

## B-EDGE-PROGRESS-1: the edge phase reports nothing, after announcing it is done

`ingest_corpus` in `src/stark_bench/adapters/stark_ingest_engine.py` calls
`_report(final=True)` at line 465 -- which logs `ingest done: 129,375/129,375
nodes (100%) ... 72m elapsed` -- and only *then* enters the edge loop at line
471, which has no logging of any kind until the process exits.

On PRIME that is 8,100,498 relationships and, measured, **~33 minutes of
total silence following a line that says the ingest is done**. The node loop
already has `_report`; this phase needs the equivalent.

Cost of not having it, paid on 2026-08-19: the qwen-wholedoc ingest was
diagnosed as hung. The reasoning looked sound at every step -- the log had
stopped, chunk count was flat over 30s, client CPU time was unchanged over
20s, and a `/slots` snapshot showed `is_processing: true` with
`n_prompt_tokens_processed: 0`. Every one of those was a misread of a healthy
run: chunks commit in `CHUNK_BATCH` steps, an I/O-bound client accrues under
a second of CPU in 20s, and `processed: 0` is what a slot reports the instant
it picks up a task. What actually settled it was `id_task` climbing ~13/s
across three `/slots` samples, and a 3-minute Postgres window showing +5,010
chunks. **A snapshot cannot distinguish stalled from busy; only a window
can.**

Two things would each have prevented the detour, and both are worth doing:

  - log progress inside the edge loop, at the same cadence as nodes;
  - do not print `ingest done` until the ingest is done. Either move
    `_report(final=True)` after the edge loop, or word the node-phase line as
    the phase it actually ends.

The second matters more than the first. A message that says the work is
finished, ~33 minutes before it is, is not a missing feature -- it is the log
actively asserting something false, and it is the reason the run nearly got
killed.

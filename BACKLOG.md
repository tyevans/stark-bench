# Backlog

Deferred work, one entry per item. Delete an entry in the commit that fixes it.

## B-SUMMARISE-1: `--summarise` is not implemented

Task 14 step 4 of the plan
(`docs/superpowers/plans/2026-08-18-stark-benchmark-harness.md`) ends with:

```
uv run python -m stark_bench.harness.cli --summarise results/ > RESULTS.md
```

`src/stark_bench/harness/cli.py` has no `--summarise` flag. The four
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

## B-ADA002-TABLE-1: the `vss-control` corpus is orphaned in the pre-rename `kg_chunks` table

`_table_for` in `src/stark_bench/harness/cli.py` derives a per-embedding-model
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

`src/stark_bench/harness/cli.py:_table_for` and every `model:` string stored
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

## B-CORESIDENCE-1 — the LLM arms need two models resident at once

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

Three real options, in the order they should be tried:

1. **A smaller chat model.** `chat_model:` is already per-config, so an arm
   can name a model that co-resides with a 700MB Q4 1B embedder. This costs
   comparability against the `dense`/`hybrid` arms only in the LLM's
   quality, which is the variable those arms do not have anyway -- but it
   must be stated in `RESULTS.md`, because "deep beat dense" and "deep beat
   dense while using a weaker LLM" are different claims.
2. **A second endpoint** serving embeddings, so the chat peer never unloads.
3. **Accepting the swap cost** for a reduced query subset, which changes the
   split and makes the number incomparable to every other row. Least good.

Nothing here is blocked on it: the control plus three arms times
`dense`/`hybrid` is seven of the numbers, and none of them make an LLM call.

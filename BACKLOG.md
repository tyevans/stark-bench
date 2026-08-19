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

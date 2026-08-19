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

**The stores are currently EMPTY** — the volumes were dropped
(`docker compose down -v`) at the last handoff, deliberately, because every
corpus needed re-ingesting for the embedding prefixes anyway. Nothing was
lost that cost endpoint time to rebuild:

- `vss-control` reloads from STaRK's precomputed vectors in minutes, and
  reproducing `0.23057383129905376` exactly is the check that it came back
  correctly.
- The nomic arms had to be re-embedded regardless.

`results/*.json` live on disk and survived.

**`redstring` is a path dependency and the path is load-bearing:**

```toml
[tool.uv.sources]
redstring = { path = "../redstring-chunkfix" }
```

That is a git worktree of redstring's `main`, deliberately separate so the
user's own redstring checkout can sit on another branch without changing what
this benchmark measures. **If you repoint it, every existing number becomes
incomparable** — redstring's chunker behaviour is one of the things under
measurement, and it has already changed once mid-campaign (see PR #64 below).
`pyproject.toml` and `uv.lock` are frequently modified in the working tree for
this reason; do not sweep them into an unrelated commit.

Never edit dependency tables by hand — `uv add` / `uv remove`.

## Running it

```
# ingest a config's corpus (idempotent; resumes by skipping existing chunks)
uv run python -m stark_bench.harness.cli --config config/<name>.yaml --ingest \
    --embed-concurrency 16

# score one architecture against it
uv run python -m stark_bench.harness.cli --config config/<name>.yaml --run \
    --agent {dense,hybrid,zero_shot,deep}
```

`--agent` overrides the config's own `agent:`, so one config serves all four.
Reports land in `results/<name>.<agent>.json`.

Useful flags: `--limit N` (throughput calibration only — never for a reported
number), `--no-resume` (force re-embed; upserts cleanly over existing rows),
`--ingest-edges` (off by default, see below).

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
- **More slots is not more throughput, and here it was less.** Measured over
  the same 1500 nodes with the same chunker: `-np 32` with the chat model
  unloaded gave 994 / 1421 / 1449 nodes/min at client concurrency 32 / 64 /
  128, while `-np 1` with the 27B chat model *also resident* gave 1745 /
  1792 at concurrency 16 / 64. One slot batching a deep client queue beats
  32 slots scheduled against each other; splitting the KV cache 32 ways
  bought nothing and cost VRAM. Client-side concurrency is the knob that
  matters, and it saturates around 64.
- **Do not compare two throughput numbers measured on different slices.**
  The first probes ran on the first 400 nodes of `nodes.jsonl`, which are
  unusually short — 1801 nodes/min there against 1449 on the first 1500,
  same server. That looked like a regression from a server change and was
  entirely the sample. Fix the slice before varying anything else.

Background a long ingest with the harness's own backgrounding, **not `nohup`
or `setsid`** — those did not survive here, twice, and the second time the
launcher exited 0 while the real work never started.

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

## Where the numbers are

`prime` / `test-0.1` (280 queries), k=20, metrics from the official evaluator:

| config | agent | mrr | hit@1 | hit@5 | recall@20 |
|---|---|---|---|---|---|
| `vss-control` (ada-002, whole-doc) | dense | 0.23057 | 0.15357 | 0.31071 | 0.37878 |
| `redstring-native` (nomic, chunked) | dense | 0.19481 | 0.13214 | 0.26429 | 0.31112 |
| `redstring-native` (nomic, chunked) | hybrid | 0.21961 | 0.14286 | 0.30000 | 0.31512 |

**The nomic rows are a floor, not a measurement** — no task prefixes. Do not
quote them against `vss-control` until they are re-run.

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

Six for six. A chunker that inflated its output 3.2x; a corpus that moved
tables; 280 queries retrieving nothing with no error logged; an embedding
model running below spec for want of a string; a model swapped underneath a
still-advertised model id; a `write_report(ingest={})` that emptied the cost
column of every report ever written. **None raised an exception at the point
of failure.** Each surfaced only because a number looked wrong, and each took
real time to trace back.

The habit that follows: **assert on the data at each step, not on exit codes.**
A stage that "succeeded" is not evidence it did anything. Check chunks per
node, minimum chunk length, row counts per tenant, and non-empty predictions —
and make the check fail loudly rather than log a warning nobody reads.

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
endpoint. See `tests/harness/test_ingest_stats_reach_the_report.py` for the
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

Open at last handoff: `B-SUMMARISE-1` (no `--summarise` to build `RESULTS.md`),
`B-BUDGET-REPORT-1` (deep agent's `exhausted_queries` is counted and nothing
reads it — a run where most queries hit the cap is a finding about the cap,
not the architecture), `B-BUDGET-CAPS-1`, `B-DEEP-EDGES-1`,
`B-ADA002-TABLE-1`, `B-SLIDING-REDUNDANT-1`.

## Before running the deep agent

`B-DEEP-EDGES-1`. An ingest without `--ingest-edges` leaves `neighbors` and
`relationships` returning empty, and a low deep-agent score against that
corpus is a data finding wearing an architecture finding's clothes.

Loading them afterwards is **minutes, not a re-ingest**, and the reason is
worth knowing rather than rediscovering: `skb/ingest.py:162`'s resume path
returns `None` for the vectors and nothing else — the entity is still built,
still batched, still added to `known`. So `--ingest --ingest-edges` with
resume at its default re-upserts entities without embedding a character,
then loads the edges. The entities being present is also what keeps
`upsert_relationships` from raising `MissingEntityError` on the first edge.

Check two things afterwards, neither of which the ingest will volunteer:
`self_loops_dropped` is non-zero (PRIME has them; a zero more likely means
the loader stopped looking), and `edges` matches `edges.jsonl`'s line count
minus those drops.

Note also that `runner.run` holds **one agent for the whole query set**. A
`DeepAgent` carrying a single `Budget` would spend the entire allowance on
query 1 and return nothing for the other 279 — near-zero score, no error
anywhere. `PerQueryDeepAgent` in `harness/agents.py` rebuilds the budget per
`retrieve`; keep it that way.

## Still to do

1. Fill the agent x config matrix. `dense` and `hybrid` are cheap; `zero_shot`
   and `deep` are LLM-bound and had never been scored as of this writing.
2. `RESULTS.md` with accuracy **and cost** per architecture — the cost side is
   why `ToolCall.tokens` exists (`int | None`, where `None` != 0), and the
   ingest half now reaches the report via `_ingest_stats`.
3. Whole-branch review, then `superpowers:finishing-a-development-branch`.

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

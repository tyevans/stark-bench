# stark-bench

Measures [redstring](https://github.com/tyevans/redstring) against the
[STaRK](https://stark.stanford.edu/) retrieval benchmark, with the retrieval
strategy as a swappable agent.

The question it exists to answer: does building a knowledge graph beat plain
vector search on the same corpus, and how much of any gap is the graph versus
the embedding model versus the chunking.

## Layout

```
domain/       values -- run config, corpus identity, budget, ingest outcome
ports/        what the use cases need: chunk index, ingest engine, agent
application/  use cases -- ingest a corpus, run queries
agents/       the subjects of the benchmark: dense, lexical, hybrid,
              zero_shot, deep
adapters/     anything with a driver, a file or a subprocess behind it
composition/  the CLI and the agent registry -- the only layer that may
              know every other one
sidecar/      runs under Python 3.11 for stark-qa; shares no code
```

Layering is enforced by `import-linter`, not documented and hoped for. Two
contracts matter beyond the ordering:

- **`agents/` sees only `ports` and `domain`.** The runner holds each query's
  `answer_ids`, so an agent that can import it is one attribute access from a
  perfect score.
- **The sidecar imports nothing first-party.** It runs under a different
  Python as a subprocess, so anything shared would not be importable there.

## Setup

```
uv sync --all-extras && uv run pre-commit install
docker compose up -d
```

Needs an OpenAI-compatible embedding endpoint, and a chat endpoint for the LLM
agents. Both are configured in `composition/cli.py`.

## Running

```
uv run python -m stark_bench.composition.cli --config config/native-wholedoc.yaml --ingest --ingest-edges
uv run python -m stark_bench.composition.cli --config config/native-wholedoc.yaml --run
```

Ingest is resumable and safe to interrupt: chunk ids are content-addressed
over `(source, text)`, so a re-run skips what it already wrote.

**That also makes a chunker change corrupting rather than merely stale** -- new
ids get written and the old rows stay live and searchable, leaving a silent
mixture of two chunkings. `scripts/resume_is_safe.py` refuses to resume
unless the recorded config is byte-identical *and* the run that wrote it
finished; the ingest report is written twice for that reason.

Useful flags: `--query-concurrency N` (set it to at least the chat peer's
`-np`), `--chat-model ID` and `--split NAME`. The last two override a config
without replacing it, which matters because the tenant is derived from the
config *name* -- a new config file for a different model would point at an
empty corpus.

### Checking a run rather than trusting it

```
uv run python -m stark_bench.composition.cli --summarise results/ > RESULTS.md
uv run python scripts/verify_corpus.py    # reports vs actual rows, per tenant
uv run bash scripts/sweep.sh              # every arm
```

`verify_corpus.py` compares each ingest report's claimed chunk count against
what the tenant holds. It found a 189-chunk discrepancy the first time it
ran. A `--run` against an empty store now refuses up front, rather than
scoring nothing retrieved as a bad retriever.

`scripts/results_table.py --check` is the other half: it exits non-zero when
a cell looks like a *broken* run rather than a bad one -- all-zero metrics,
an unbacked cost column, a `deep` arm on an edgeless corpus, or a report
written against a different config than the one now on disk. `--summarise`
renders; this one judges.

## Arms

Seventeen configs live in `config/`. The two carrying the current results:

| config | corpus | chunking | embeddings |
|---|---|---|---|
| `qwen-rel-whole` | `prime-rel` (documents name their neighbours) | whole document | qwen3-embedding-0.6b |
| `qwen-rel-sliding1k` | `prime-rel` | sliding 1000/500 | qwen3-embedding-0.6b |

The original control set -- `vss-control`, `native-wholedoc`,
`redstring-native`, `native-sliding1k` -- was built so the model change and
the chunking change could be attributed separately. Their corpora have since
been dropped; the results survive in `results/`, but reproducing one needs a
re-ingest. `vss-control` is the cheap one, since its vectors are precomputed.

**Agents** are listed by
`uv run python -c "from stark_bench.composition.agent_registry import AGENTS; print(sorted(AGENTS))"`.
Four families: retrieval only (`dense`, `lexical`, `hybrid`), LLM-driven
(`zero_shot`, `deep`), reranking on full documents (`rerank*`), and reranking
on lean encodings (`rerank40title*`), which reach comparable accuracy at
roughly a twentieth of the cost.

## Scoring

`mrr`, `hit@1`, `hit@5`, `recall@20`, computed by STaRK's own `Evaluator` in
the 3.11 sidecar rather than reimplemented here.

`scripts/build_sidecar_env.sh` builds that environment once, so scoring
touches no network. Optional -- without it the sidecar resolves from PyPI
per run, which has cost a completed run before.

**Numbers from LLM agents carry a noise floor.** Two identical runs differ
by ~0.001 mrr and ~0.011 hit@5: temperature zero does not make a batched
server deterministic. Retrieval-only arms are reproducible to every digit.
See CLAUDE.md.

## Testing

```
uv run pytest -q
```

Quality gates run on commit via pre-commit. Read `CLAUDE.md` before writing
tests -- it carries the list of ways a test here has passed while proving
nothing, which is longer than you would like.

Deferred work goes in `BACKLOG.md`, in the commit that passes it by.

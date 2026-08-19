# stark-bench

Measures [redstring](https://github.com/tyevans/redstring) against the
[STaRK](https://stark.stanford.edu/) retrieval benchmark, with the retrieval
strategy as a swappable agent.

The question it exists to answer: does building a knowledge graph beat plain
vector search on the same corpus, and how much of any gap is the graph versus
the embedding model versus the chunking.

## Layout

```
domain/       values -- corpus identity, ingest outcome, cost
ports/        what the use cases need: chunk index, ingest engine, agent
application/  use cases
agents/       the subjects of the benchmark
adapters/     Postgres, and anything else with a driver in it
harness/      CLI, config, scoring, reporting (being dissolved into the above)
skb/          STaRK's knowledge base -> redstring's stores
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
agents. Both are configured in `harness/cli.py`.

## Running

```
uv run python -m stark_bench.harness.cli --config config/native-wholedoc.yaml --ingest --ingest-edges
uv run python -m stark_bench.harness.cli --config config/native-wholedoc.yaml --run
```

Ingest is resumable and safe to interrupt: chunk ids are content-addressed
over `(source, text)`, so a re-run skips what it already wrote.

**That also makes a chunker change corrupting rather than merely stale** -- new
ids get written and the old rows stay live and searchable, leaving a silent
mixture of two chunkings. `scripts/resume_is_safe.py` refuses to resume unless
the recorded config is byte-identical.

`scripts/sweep.sh` runs every arm. `scripts/results_table.py` renders
`RESULTS.md` and flags results that look like a broken run rather than a bad
one -- all-zero metrics, an empty ingest block, an edgeless corpus under a
graph agent.

## Arms

| config | chunking | embeddings | agent |
|---|---|---|---|
| `vss-control` | whole document | precomputed ada-002 | dense |
| `native-wholedoc` | whole document, capped | Nemotron-3-Embed-1B | dense |
| `redstring-native` | boundary preference | Nemotron-3-Embed-1B | hybrid |
| `native-sliding1k` | sliding 1000/500 | Nemotron-3-Embed-1B | dense |

`vss-control` is the floor: no graph, no live embedding, published vectors.
`native-wholedoc` exists so the model change and the chunking change can be
attributed separately -- `vss-control` minus it is the model, it minus
`redstring-native` is the chunking.

Agents: `dense`, `hybrid`, `zero_shot`, `deep`.

## Scoring

`mrr`, `hit@1`, `hit@5`, `recall@20`, computed by STaRK's own `Evaluator` in
the 3.11 sidecar rather than reimplemented here.

## Testing

```
uv run pytest -q
```

Quality gates run on commit via pre-commit. Read `CLAUDE.md` before writing
tests -- it carries the list of ways a test here has passed while proving
nothing, which is longer than you would like.

Deferred work goes in `BACKLOG.md`, in the commit that passes it by.

# STaRK benchmark harness for redstring

**Date:** 2026-08-18
**Status:** approved design, not yet implemented

## What this is

A harness that measures [redstring](https://github.com/tyevans/redstring) against
[STaRK](https://stark.stanford.edu/) (NeurIPS 2024 Datasets & Benchmarks), with
retrieval agent architectures as a plug-in point so that "which agent" and "which
retrieval stack" are separately variable.

It lives in its own repository. It depends on redstring as a library and on
`stark-qa` for data and scoring. It is not part of redstring, and redstring is
not modified to accommodate it.

## What is under test

**Retrieval only.** STaRK ships a pre-built semi-structured knowledge base (SKB):
typed nodes, relations between them, and a text document per node. The harness
loads that KB into redstring's stores as-is and measures retrieval. It does not
run extraction.

This is a deliberate narrowing. Re-extracting entities from STaRK's node
documents would measure extraction and retrieval at once, and a disappointing
number would have two explanations and no way to separate them. It would also
mean redstring's entity ids no longer correspond to STaRK's node ids, so
ground truth would need an alignment layer that is itself a source of error.

Extraction remains measurable — redstring's own `bench/` already does that — and
a full-pipeline track can be added later as a separate, separately-labelled
experiment.

## Ingest: the loader is a projection

The loader reads STaRK's SKB and writes through redstring's ports. It invents
nothing and fetches nothing: the caller supplies every byte, a projection does
every write. This is consistent with redstring's stated rules.

Confirmed against source: entities, relationships, chunks and vectors can all be
written directly, with no events and no extraction machinery.

| Concern | Port and method |
|---|---|
| Entities | `EntityWriter.upsert_entities(entities)` |
| Relationships | `RelationshipStore.upsert_relationships(relationships)` (atomic) |
| Chunks | `ChunkWriter.upsert_many(chunks)` (batch only; no singular form) |
| Vectors | `VectorWriter.upsert_many(items)` |

### Identity

`EntityId = uuid5(NAMESPACE_STARK, f"{dataset}:{node_id}")`.

Deterministic, so ingest is idempotent and resumable, and the forward map needs
no stored side-table that could drift from the data.

The reverse direction rides `Entity.external_ids`, a first-class `dict[str, str]`
field on the domain type: `{"stark_node_id": "<id>"}`. Retrieval returns whole
`Entity` objects, so mapping a result back to a STaRK node id is a field read on
an object we already hold — never a store query. This matters because no reader
port offers lookup-by-external-id, and none is needed.

### Required fields

- `Entity`: `id`, `tenant_id`, `name`, `normalized_name`, `entity_type`, `provenance`.
- `Provenance`: `observed_at`, `extraction_method`, `confidence`.
  `extraction_method` is `ExtractionMethod.MANUAL` — the honest value, because
  we did not extract this KB.
- `Relationship`: `id`, `tenant_id`, `source_entity_id`, `target_entity_id`,
  `relationship_type`, `confidence`.
- `StoredChunk`: `id`, `tenant_id`, `source_id`, `text`, `chunk_index`,
  `start_char`, `end_char`; plus optional `entity_ids` and `metadata`.

### Two ordering constraints

- `upsert_relationship` raises `MissingEntityError` if either endpoint is absent,
  so every batch writes entities before the relationships that reference them.
- Self-loops are rejected by validation. STaRK relations that are self-loops are
  dropped, and **the dropped count is recorded in the ingest report**. A silent
  drop would make a recall ceiling look like a retrieval failure.

## Chunking, embeddings, and the control

Redstring's `Chunker` is a plain protocol (`chunker_type`, `chunk(...)`), and
`chunker_type` is recorded on results, so a chunking configuration labels itself
in the output.

`EmbeddingProvider` is likewise a runtime-checkable protocol (`model`,
`dimension`, batch `embed` whose ordering is the contract). STaRK's precomputed
`text-embedding-ada-002` vectors therefore enter as an ordinary provider backed
by a lookup table.

This yields **one code path and two configurations**:

| Config | Chunker | Provider | Store |
|---|---|---|---|
| `vss-control` | whole-document (one chunk per node) | precomputed ada-002, dim 1536 | control store |
| `redstring-native` | `BoundaryPreferenceChunker` | `nomic-embed-text`, dim 768 | native store |

Ingest, retrieval, aggregation and scoring code are identical either side; only
config differs. That is what makes a discrepancy between the two rows
attributable.

The control's job is diagnostic. If `vss-control` lands far from STaRK's
published VSS row, the fault is in our ingest or scoring rather than in
redstring — without it, a disappointing native number has two explanations.
It costs almost nothing, because the embeddings are a download.

### Two hard rules

- **A precomputed-vector lookup miss raises.** The table is keyed by the exact
  document text STaRK embedded. A miss must never return a zero vector and must
  never fall back to live embedding: a silent fallback turns the control into a
  second native run wearing the control's label, corrupting every comparison
  downstream while looking fine.
- **Separate stores per embedding model.** Redstring's ADR 0002 records that a
  different embedding model means a new store, and dimensions differ anyway
  (1536 vs 768). The dimension guard catches width mismatches but *not*
  model-identity mismatches, so two same-width models sharing a store would
  corrupt silently. One store per config, one tenant per config.

## Store backing

Postgres/pgvector for vectors, the postgres chunk store for the corpus, and
Neo4j for the graph, brought up by a `docker-compose.yml` in this repository.

Ingest is paid once and persisted, so agent runs re-use it without re-embedding
129k nodes' worth of chunks. pgvector's index also keeps an 11k-query run cheap,
which brute-force in-memory search would not at this size, and would not at all
for `mag` or `amazon` later.

In-memory stores remain the backing for the fixture SKB in unit tests, where
their zero setup cost is the point.

## Retrieval surface

Verified against source, and this changed the design:

**The entity-side lexical channel is a name matcher, not passage search.** It
draws candidates from `find_by_blocking_keys` (prefix/soundex) and scores them
with Jaro-Winkler. A natural-language STaRK query shares no blocking key with a
node named `PTGS2`, so that channel returns essentially nothing for this
workload.

**The chunk side is a real term-weighted ranker** — `domain/bm25.py`
(`BM25_K1 = 1.2`, `BM25_B = 0.75`) reached through `rank_chunks`.

Therefore **`ChunkRetriever` is the primary retrieval surface**, and `Retriever`
is retained only as a clearly-labelled entity-level baseline. An entity-side
hybrid row would measure a name matcher against natural language and report a
bad number about the wrong thing.

`RetrievalMode` is `SEMANTIC | LEXICAL | HYBRID` on both retrievers, so the
dense-only and hybrid baselines come from the same code path with one argument
changed.

Unit mismatch to respect: `ScoredChunk.lexical` is **unbounded** BM25 while
`ScoredEntity.lexical` is `0..1`. Two different units under one field name.
Nothing may average them. Fusion is by rank, which is unit-free, and that is
why.

### RRF is not a knob

`RRF_K = 60` is a module constant. Its docstring says exposing it "would invite
tuning against a benchmark this library does not have."

We are now that benchmark. **We do not tune it and we do not expose it.** Every
headline number is reported at the documented constant. If sensitivity becomes
an interesting question it is a separate, differently-labelled experiment, never
the number we quote. A benchmark whose purpose is to evaluate its own library
must not acquire a knob that lets it flatter itself.

## Chunk-to-node aggregation

Retrieval returns scored chunks; STaRK scores nodes. `StoredChunk.source_id`
carries the document a chunk came from, which for this corpus is the node — so
aggregation rides a real domain field rather than metadata we invent.

Default: **max over a node's chunks.** The aggregation function is a named,
recorded config value; alternatives (mean, sum, RRF over chunk ranks) are
selectable, and every results file states which produced it. An aggregation
function that is an unrecorded tuning knob turns a benchmark into a search for
its best accident.

On `vss-control` there is one chunk per node, so aggregation degenerates to
identity — the control exercises the aggregation code without aggregation being
able to change its answer.

## The agent seam

```python
class Agent(Protocol):
    async def retrieve(self, query: Query, tools: Toolset) -> Sequence[Ranked]: ...
```

`Ranked` is `(node_id, score)` — exactly `pred_dict`'s shape, so nothing is
reshaped between an agent and the official evaluator.

`Toolset` wraps **reader ports only**: `vector_search`, `lexical_search`,
`get_node`, `get_chunks`, `neighbors`, `get_relationships`. Reader-only is a
type-level guarantee, the same argument redstring's own `Retriever` makes for
holding `VectorReader` rather than `VectorStore`: an agent that cannot reach a
writer cannot poison the KB mid-run.

**Traversal is a separate handle.** `Retriever` holds `EntityReader` only and has
no traversal at all; multi-hop comes from `RelationshipStore.neighbors(...)` and
`get_relationships_for(...)`. `neighbors` returns entities with no edge type and
no hop distance, so an agent needing to know *how* two nodes connect must also
call `get_relationships`.

Every tool call is instrumented: count, latency, and tokens where applicable.
**Cost is a reported metric, not a footnote.** A deep agent that buys +4 Hit@1
for 40x the tokens is a different finding depending on which number you needed,
and Hit@1 alone cannot express it.

Budgets — max tool calls, max LLM calls, wall-clock — are enforced by the
harness. Budget exhaustion is a **recorded outcome, not an exception that voids
the run**: an agent returns its best-so-far at the cap and is scored on it.
Any loop over adapter-supplied data is hard-bounded, because a hung query in an
11k-query run reads as infrastructure trouble and gets retried rather than
investigated.

### The four architectures

| Agent | Behaviour | Purpose |
|---|---|---|
| `dense` | one `vector_search`, return it | the control; no LLM |
| `hybrid` | vector + BM25, RRF-fused (`ChunkRetriever`, `HYBRID`) | does redstring's fusion beat dense; no LLM |
| `zero_shot` | one LLM call to form tool arguments, one retrieval round, optional rerank | fixed cost per query |
| `deep` | plan/act/observe loop against the same tools, budget-bounded | stresses the seam |

The two LLM-free baselines run the full query set cheaply and often, which is
what lets us tell whether a moved agent number reflects the agent or the KB
underneath it.

## Scoring

`stark_qa.evaluator.Evaluator`, unmodified, over `pred_dict`. **We compute no
metric ourselves.** Per redstring's own testing rules, an expected value produced
by the code under test measures determinism rather than correctness, and a
reimplemented MRR is exactly that.

Reported: `mrr`, `hit@1`, `hit@5`, `recall@20` — the leaderboard's set — plus
per-query cost and budget-exhaustion counts.

Iteration runs on the official `test-0.1` split (a 10% sample). Full `test` is
for numbers we publish.

## The stark-qa sidecar

`stark-qa` declares `requires-python >=3.8` with no upper bound, but its
dependency closure (`colbert-ai`, `gritlm`, `llm2vec`, `PyTDC`, `ogb`,
`torch_geometric`, `voyageai`, `openai`, `anthropic`) serves baselines we do not
run, and several will not resolve on 3.13.

So `stark-qa` is confined to an isolated 3.11 environment invoked via
`uv run --with`, doing exactly two jobs:

1. **Export** — `load_skb` / `load_qa` to neutral artifacts (parquet/jsonl for
   nodes, edges, documents and queries; `.npy` for precomputed embeddings).
2. **Score** — the official `Evaluator` over a `pred_dict` artifact.

The harness itself is Python 3.13 with a small dependency set, and consumes only
the neutral artifacts. Scoring stays authoritative; the lockfile stays sane.

## Repository shape

```
stark_bench/
  ports.py          # Agent, Toolset, Query, Ranked  -- the only harness module agents may import
  skb/
    ids.py          # node_id <-> EntityId
    load.py         # neutral artifacts -> Entity / Relationship / StoredChunk streams
    ingest.py       # drives the writers; resumable; emits an ingest report
  agents/           # dense, hybrid, zero_shot, deep
  tools/            # the instrumented reader-only tool surface
  harness/          # runner, budgets, aggregation, scoring, report
  sidecar/          # stark-qa export + score, run under 3.11
  config/           # one YAML per run
```

Two rules carried over from redstring, because they earned their place there:

- **`agents/` may import `ports.py` and nothing else from the harness.** An agent
  with a path to the runner has a path to ground truth, and a retrieval agent
  that can see `answer_ids` is one accidental import from a perfect score.
  Enforced with `lint-imports`, not convention.
- **The resolved config is embedded verbatim in every results file**, which is
  what makes a number re-runnable.

Redstring's production adapters (`PgVectorStore`, `Neo4jGraphStore`, the postgres
chunk store) are not in `redstring.__all__`. The harness imports them by dotted
path against a pinned redstring version, and carries a test asserting those
import paths still resolve, so a version bump fails loudly rather than at ingest.

## Testing

- **A tiny fixture SKB** — roughly a dozen nodes with known answers, so ingest,
  aggregation and scoring are testable without downloading 129k nodes. Every
  agent runs against it in unit tests.
- **A deliberate-break check on scoring.** An agent returning ground truth must
  score 1.0; an agent returning garbage must score near 0. A scoring path that
  cannot be made to fail on purpose is not yet measuring anything.
- **The agent isolation rule as a test**, not a convention.
- **Bounded loops** everywhere adapter-supplied data decides an exit.

Per redstring's testing notes, tests avoid inputs that make two candidate
implementations agree: aggregation is tested with a node whose chunks have
*different* scores, ranking with ties broken deliberately, and the id map with
ids that collide on one component of a composite key.

## Slice order

Each slice produces a number. Nothing waits on the deep agent to prove the
harness works.

1. Fixture SKB, scoring, `dense` — end to end on toy data, no downloads.
2. Sidecar export of stark-prime; ingest; `vss-control`; evaluate on `test-0.1`.
3. `hybrid` on `redstring-native`. The first number that means something.
4. `zero_shot`, then `deep`, with budgets.

## Deferred deliberately

- Datasets `amazon` and `mag`. The harness is written dataset-parametric, but
  only `prime` is exercised first: it is the most graph-heavy of the three and
  the cheapest to embed, so it is where a graph story should show up at all.
- The full-pipeline (re-extraction) track.
- Chunk aggregation strategies beyond `max`, which are selectable but unstudied.

"""Bring one config to life: ingest, or run and score.

Two subcommands, `--ingest` and `--run`, over one `RunConfig`. Ingest and
retrieval are separate invocations deliberately -- a run this large is worth
being able to retry independently of the other.

`--ingest-edges` defaults OFF. `dense`, `hybrid` and `zero_shot` retrieve
through `ChunkRetriever`, which holds a `ChunkStore` and never touches the
graph. `deep` does traverse -- its `neighbors` and `relationships` actions go
to the graph store -- so a `deep` run against a corpus ingested without edges
measures an agent that has been given no edges to walk. Whichever way ingest
was run is recorded in the report, so a later reader is not left guessing
whether a number came from a graph-less store.
"""

from __future__ import annotations

import argparse
import asyncio
from time import perf_counter
import itertools
import json
import logging
from functools import partial
from hashlib import blake2b
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid5

from redstring import TenantId
from redstring.chunks.adapters.postgres import PostgresChunkStore
from redstring.extraction.chunkers.boundary_preference_chunker import (
    BoundaryPreferenceChunker,
)
from redstring.extraction.chunkers.sliding_window_chunker import SlidingWindowChunker
from redstring.graph.adapters.neo4j import Neo4jGraphStore
from redstring.llm.adapters.langchain import NO_THINKING, LangChainLlmProvider
from redstring.llm.adapters.langchain_embedding import LangChainEmbeddingProvider

from stark_bench.composition.agent_registry import AGENTS, build_agent
from stark_bench.adapters.config_file import load_config
from stark_bench.adapters.postgres_embedding_cache import PostgresEmbeddingCache
from stark_bench.domain.run_config import RunConfig
from stark_bench.adapters.prewarmed_query_embeddings import (
    PrewarmedQueryEmbeddings,
)
from stark_bench.adapters.precomputed_embeddings import (
    PrecomputedEmbeddingProvider,
    node_vector_lookup,
)
from stark_bench.adapters.model_preflight import require_chat_model
from stark_bench.adapters.report_file import (
    summarise_cost,
    write_predictions,
    write_report,
)
from stark_bench.application.run_queries import run
from stark_bench.adapters.stark_scorer import score_predictions
from stark_bench.adapters.stark_artifacts import (
    read_doc_embeddings,
    read_edges,
    read_nodes,
    read_queries,
    read_query_embeddings,
)
from stark_bench.adapters.chunkers import WholeDocumentChunker
from stark_bench.domain.stark_ids import NAMESPACE_STARK
from stark_bench.application.ingest_corpus import ingest_corpus
from stark_bench.application.summarise import summarise
from stark_bench.adapters.stark_ingest_engine import ingest
from stark_bench.adapters.postgres_chunk_index import PostgresChunkIdIndex
from stark_bench.adapters.redstring_toolset import RedstringToolset

if TYPE_CHECKING:
    from redstring import EmbeddingProvider, LlmProvider

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "results"

POSTGRES_DSN = "postgresql://stark:stark@localhost:55432/stark"
NEO4J_URI = "bolt://localhost:57687"
NEO4J_AUTH = ("neo4j", "starkbench")

#: llama-swap, which fronts **both** the embedding model and the chat model
#: on one port and loads whichever a request names.
#:
#: Named for the endpoint rather than for one of its models, because it was
#: `EMBED_BASE_URL` while also being what `_llm_for` passed as the chat
#: provider's `base_url` -- correct, and reading like a bug. The old comment
#: said it backed `nomic-embed-text`, which stopped being true when the
#: embedder became Nemotron-3-Embed-1B.
#:
#: Shared infrastructure: ask before saturating it, and see
#: `--embed-concurrency` on the CLI.
#:
#: One port, both models resident together: the embedding server runs
#: concurrently with the chat model, so an agent that interleaves embedding
#: and chat does NOT cause a swap. This comment previously claimed the
#: opposite and it was wrong -- see B-CORESIDENCE-1, which is resolved.
#:
#: Cold-start latency is still real (~37s) but it is an idle eviction and
#: first load, not churn between two models.
INFERENCE_BASE_URL = "http://192.168.1.14:8080/v1/"

#: The chat model behind `zero_shot` and `deep`, on the same endpoint as
#: `INFERENCE_BASE_URL` (llama-swap serves both, embeddings on a separate peer).
#: Overridable per config via `chat_model:`; a config that omits it gets this.
#: Text-only, 16k context, `-np 1`. The variant matters for a reason that
#: has nothing to do with quality: the multimodal `qwen3.8-27b-mtp` at
#: `--ctx-size 65536 -np 4` left 373 MiB free on the card beside the
#: embedder, and the embedding peer died of `CUDA error: out of memory` four
#: times -- twice at client concurrency 64 and twice at 16, which is how we
#: learned the variable was headroom rather than request rate.
#:
#: Nothing is given up. `runner.run` is a bare `for query in queries` loop,
#: so exactly one chat request is ever in flight and three of those four
#: slots could never be used; STaRK is text, so the F16 projector was
#: resident for nothing; and 16384 is the per-slot window the previous
#: configuration actually gave (65536 / 4), so `deep.py`'s context
#: assumptions are unchanged.
#:
#: Raised to a 64k window on 2026-08-19, for the reranker: it puts 40
#: candidate documents in one prompt, and at 16k each had to be cut to 600
#: characters -- which on the relations corpus truncates every document
#: *before* the `- relations:` block, the exact text that arm exists to test.
#:
#: **The model id carries the window, so raising it renames the model.** The
#: 16k id stopped existing the moment the 32k one appeared, and a stale
#: constant here does not degrade gracefully: every chat call 404s, the
#: agents that swallow LLM errors fall back to plain retrieval, and the run
#: still produces a full set of plausible numbers. `redstring-native/deep`
#: died mid-run at the changeover with 143 of 280 queries empty, which is
#: the loud version of the same event -- the quiet version scores like
#: `hybrid` and says nothing.
DEFAULT_CHAT_MODEL = "qwen3.8-27b-64k-txt"

#: Chunking strategies, as zero-argument builders.
#:
#: `sliding-1000-500` is the deliberately over-chunked end of the sweep: a
#: 1000-character window advancing 500, so every position in a long document
#: appears in two chunks. It exists to answer whether chunking helps or hurts
#: at all, by bracketing `boundary-preference` (1.14 chunks/node) between it
#: and `whole-document` (1.00) with the embedding model held fixed.
#:
#: Note what it can and cannot move: 86% of this corpus is under 1000
#: characters and comes through whole regardless, so the three configs differ
#: only on the 14% long tail.
CHUNKERS = {
    "whole-document": WholeDocumentChunker,
    "boundary-preference": BoundaryPreferenceChunker,
    "sliding-1000-500": partial(
        SlidingWindowChunker, default_chunk_size=1000, default_overlap=500
    ),
    #: Whole documents, split only where the embedding server cannot take
    #: them. Nemotron-3-Embed-1B is served with a 4096-token per-slot context
    #: (--ctx-size / -np), and embeddings are non-causal so a sequence cannot
    #: span two physical batches -- the real ceiling is
    #: min(ctx-per-slot, ubatch-size), which is why the flag that mattered
    #: turned out to be --ubatch-size and not --ctx-size.
    #:
    #: 5000 characters is measured, not guessed. Sampling the longest and
    #: the least-whitespace documents in this corpus against the live
    #: tokenizer gives a worst case of 2.55 characters per token -- against a
    #: 4.31 median, because PRIME carries SMILES strings and other dense
    #: identifiers that tokenise nothing like prose. 5000 / 2.55 = 1961
    #: tokens, inside the 2048-token ubatch with the `passage: ` prefix and a
    #: BOS on top. Sizing this off the median would have put the densest
    #: documents 1100 tokens over the limit.
    #:
    #: The binding constraint is `--ubatch-size`, not `--ctx-size`, and the
    #: server is deliberately configured small: `-np 1 --ctx-size 4096
    #: --ubatch-size 2048` is what leaves room for the 27B chat model to stay
    #: resident beside the embedder. Measured, that configuration also
    #: ingests FASTER than `-np 32` with the chat model unloaded -- 1792
    #: against 1449 nodes/min over the same 1500 nodes -- so the small
    #: setting costs throughput nothing and the only price is this cap.
    #:
    #: A 10000-character cap would give 1.016 chunks/node against this
    #: 1.057. Both are unambiguously the low end of a sweep whose other
    #: points are 1.14 and 1.94, and the larger cap is not worth a second
    #: server configuration and a phase split to reach.
    #:
    #: Zero overlap, so this splits rather than truncates. Truncating would
    #: have matched what ada-002 does to its own overlong inputs, but ada-002
    #: truncates 26 documents at its 8191-token ceiling where we would
    #: truncate 1,826 at 10000 characters -- so the whole-doc arm would
    #: silently hold less text than the chunked arms, and the sweep would be
    #: measuring content loss as well as granularity. Splitting keeps the
    #: text identical across all three configs and leaves granularity the
    #: only variable, which is the entire point of the cell.
    #:
    #: 1,826 of 129,375 documents (1.41%) exceed it, against 2.57% at the old
    #: 7000-character cap -- so this arm is now ~1.02 chunks/node.
    #:
    #: Zero overlap also sidesteps B-SLIDING-REDUNDANT-1: the redundant tail
    #: chunk appears only when the window advances by less than its width.
    "capped-whole-5000": partial(
        SlidingWindowChunker, default_chunk_size=5000, default_overlap=0
    ),
    #: For nomic-embed-text, whose hard ceiling is **2048 tokens** -- the
    #: model's trained context. The server enforces it per request:
    #:
    #:     400 exceed_context_size_error
    #:     input (2083 tokens) is larger than the max context size (2048)
    #:
    #: That failure is why this entry exists. `capped-whole-5000` was reused
    #: for nomic on the strength of an estimate of 4.0 characters per token,
    #: which was measured on CHAT prompts through Qwen's tokenizer -- a
    #: different tokenizer over different text. Through nomic's WordPiece the
    #: densest PRIME documents run about **2.4** characters per token, so a
    #: 5000-character chunk is ~2083 tokens and overruns by 35.
    #:
    #: **This cap was wrong three times, each time from a sample that missed
    #: the dense tail.** The sequence is worth keeping, because every estimate
    #: was defensible and every one was too high:
    #:
    #:   5000 chars -- from 4.0 chars/token, measured on CHAT prompts through
    #:                 Qwen's tokenizer. Failed at 2083 tokens.
    #:   4000 chars -- from 2.4, the ratio implied by that failure. A 25-doc
    #:                 sample said 3.42 and would have justified 6300; it was
    #:                 ignored in favour of the failure, and the failure was
    #:                 still not the worst case. Failed at 2088 tokens.
    #:   2400 chars -- from 1.754, measured by ranking all 607,292 documents
    #:                 over 800 characters in BOTH corpora by token density
    #:                 and probing the 250 densest individually. The worst is
    #:                 a MAG physics title.
    #:
    #: 2400 is deliberately below the 2873 that 1.754 permits at a 20% margin.
    #: At the measured worst it is 1368 tokens, 67% of the ceiling, and it
    #: survives any ratio down to 1.17. Two consecutive underestimates are
    #: enough evidence that the tail is longer than sampling finds.
    #:
    #: The principled fix is to cap by TOKENS, which needs nomic's WordPiece
    #: vocabulary locally, or to catch the 400 and split the offending chunk.
    #: Either ends this guessing permanently -- see B-TOKEN-CAP-1.
    #:
    #: This is NOT a free swap of one cap for another. RESULTS.md finding 4
    #: measures a 25% spread in dense mrr across chunkers, so the chunker is
    #: the second-largest retrieval effect on the page and changing the cap
    #: changes it. A nomic-against-Nemotron comparison therefore varies the
    #: model AND the chunking and cannot attribute a gap to either.
    #:
    #: Note what this does NOT claim. An earlier version of this comment said
    #: finer chunking is monotonically worse, citing a finding RESULTS.md has
    #: since **retracted**: `native-sliding1k` has twice `redstring-native`'s
    #: granularity and scores 15% better, so the effect is not monotonic in
    #: chunks/node. The direction of the 5000 -> 4000 change is therefore
    #: unknown, not merely unmeasured -- which is a weaker claim than the
    #: confound needs, and enough for it. Corpus-against-corpus on nomic
    #: (`nomic-wholedoc` against `mag-wholedoc`) stays clean, which is the
    #: comparison the MAG run exists for.
    "capped-whole-2400": partial(
        SlidingWindowChunker, default_chunk_size=2400, default_overlap=0
    ),
}
#: Models `_live_embeddings_for` will build a provider for. Membership is the
#: only gate -- dimension and task prefixes come from the config, so adding a
#: model here is adding a name, not a code path.
#:
#: Nemotron-3-Embed-1B is kept alongside `nomic-embed-text` because the
#: Nemotron arms in `results/` are still the comparison for the model swap:
#: it scored WORSE than precomputed ada-002 (0.2163 vs 0.2306 MRR on
#: `dense`, recall@20 a tie) at 2048 dimensions against 768, which is what
#: made the swap worth doing. Removing it would make those numbers
#: unreproducible.
#: `qwen3-embedding-0.6b` is the current endpoint model (Q8_0 GGUF, 1024
#: dimensions, served with `n_ctx 32768` and a physical batch large enough to
#: take a 40,000-character input in one piece -- probed, not assumed). It is
#: the first model here whose context is not the binding constraint on the
#: chunker, so its whole-document arm caps at 5000 characters for parity with
#: the Nemotron sweep rather than because the server would refuse more.
LIVE_EMBEDDINGS = {"Nemotron-3-Embed-1B", "nomic-embed-text", "qwen3-embedding-0.6b"}


def _tenant_for(config: RunConfig) -> TenantId:
    """One deterministic tenant per config name -- stable across re-runs."""
    return TenantId(uuid5(NAMESPACE_STARK, f"tenant:{config.name}"))


def _count_lines(path: Path) -> int | None:
    """Node count for the ingest progress ETA, or None if it cannot be had.

    Read as bytes and counted with `bytes.count`, which is one pass in C --
    on MAG's 916 MB nodes.jsonl a Python-level loop is seconds of wall time
    before the first embedding request goes out.

    Returns None rather than raising: this exists to make a log line say
    "40%" instead of "?", and an ingest must not fail because a progress
    figure was unavailable.
    """
    try:
        with path.open("rb") as handle:
            return sum(
                block.count(b"\n") for block in iter(lambda: handle.read(1 << 22), b"")
            )
    except OSError:
        return None


def _data_dir(config: RunConfig) -> Path:
    return DATA_ROOT / config.dataset


def _table_for(config: RunConfig) -> str:
    """One chunk-store table per embedding model, never shared.

    ADR 0002 records that a different embedding model means a new store.
    Dimension differs here anyway (1536 vs 768), but width alone would not
    protect against a second same-width model landing in the same table
    later -- the dimension guard cannot see a model-identity mismatch, so
    the table name is keyed on the embeddings identifier by construction.
    """
    # Lowercased and non-alphanumerics collapsed, because a model id is a
    # vendor's string and a Postgres identifier is not. `Nemotron-3-Embed-1B`
    # has capitals; redstring's chunk store rejects anything but a bare
    # lowercase identifier, which is how that was found rather than by
    # anyone reading this line.
    slug = "".join(c if c.isalnum() else "_" for c in config.embeddings).lower()
    if not config.document_prefix and not config.query_prefix:
        return f"kg_chunks_{slug}"
    # The task prefix is part of the model's identity, not a formatting
    # detail: text embedded behind `search_document: ` and the same text
    # embedded bare land in different regions of the space, and cosine
    # between them is meaningless. redstring ADR 0043 says so explicitly,
    # extending ADR 0002's "new model means a new store".
    #
    # A digest rather than the prefixes themselves because a prefix is
    # free-form text and a table name is not. It is appended only when a
    # prefix is set, so the unprefixed tables keep the names the existing
    # results were written against -- renaming those would orphan every row
    # ingested before today, which is precisely the failure that cost this
    # project a full ada-002 run.
    identity = (
        f"{config.embeddings}\x00{config.document_prefix}\x00{config.query_prefix}"
    )
    digest = blake2b(identity.encode("utf-8"), digest_size=4).hexdigest()
    return f"kg_chunks_{slug}_{digest}"


def _live_embeddings_for(config: RunConfig) -> EmbeddingProvider:
    if config.embeddings in LIVE_EMBEDDINGS:
        return LangChainEmbeddingProvider.openai_compatible(
            base_url=INFERENCE_BASE_URL,
            model=config.embeddings,
            dimension=config.dimension,
            document_prefix=config.document_prefix,
            query_prefix=config.query_prefix,
        )
    raise NotImplementedError(f"no live embedding provider for {config.embeddings!r}")


def _llm_for(config: RunConfig) -> LlmProvider:
    """The chat provider `zero_shot` and `deep` extract through.

    Built unconditionally for a `--run`, including for `dense` and `hybrid`
    which never call it: construction is pure configuration and does no I/O,
    and the alternative -- deciding per agent whether the toolset gets an
    LLM -- is a second place for the agent name to be interpreted.
    """
    model = config.effective_chat_model or DEFAULT_CHAT_MODEL
    require_chat_model(INFERENCE_BASE_URL, model)

    # Built here rather than through `openai_compatible`, which is
    # deliberately not a passthrough for extra `ChatOpenAI` kwargs and says
    # so: "a caller needing anything else still builds the chat model
    # itself". We need one such kwarg.
    #
    # ## Why `cache_prompt`
    #
    # `rerank`'s prompt opens with 818 characters -- ~221 tokens -- that are
    # byte-identical on every query: the scoring instructions and the output
    # instruction, both before `Query:`. Everything after varies, and cannot
    # be shared: 98.2% of query pairs retrieve zero candidates in common.
    #
    # llama.cpp will reuse a slot's KV for a matching prefix, and a measured
    # response showed `cache_n: 0` -- the request never asked. It matters
    # more under concurrency, not less: prefill throughput per request falls
    # to ~500 tok/s with four slots busy, so those 221 tokens are ~0.44s of
    # every request rather than the ~0.09s a solo-request rate suggests.
    #
    # ## Why `extra_body` is spelled out rather than extended
    #
    # `openai_compatible` puts `NO_THINKING` in `extra_body`, and that is
    # load-bearing: this harness is non-reasoning by deliberate measurement
    # (CLAUDE.md -- two thinking-on runs at temperature zero disagreed with
    # each other about how many entities a document held). Passing an
    # `extra_body` that omits it would turn reasoning back on silently,
    # costing latency and reproducibility at once. So it is merged in, and
    # a test asserts it survives.
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        base_url=INFERENCE_BASE_URL,
        api_key="not-needed",
        temperature=0.0,
        extra_body={**dict(NO_THINKING), "cache_prompt": True},
    )
    return LangChainLlmProvider(chat, model=f"openai-compatible/{model}")


def toolset_for(
    *,
    chunks: object,
    graph: object,
    embeddings: EmbeddingProvider,
    config: RunConfig,
    tenant_id: TenantId,
) -> RedstringToolset:
    """The agent-facing toolset, LLM included.

    A separate function because the LLM is the part that is easy to leave
    out: a toolset built without one is fully functional for `dense` and
    `hybrid` and raises only when an LLM agent first calls `extract`, which
    is several thousand lines of run into a benchmark.
    """
    return RedstringToolset(
        chunks=chunks,
        graph=graph,
        embeddings=embeddings,
        tenant_id=tenant_id,
        dataset=config.dataset,
        llm=_llm_for(config),
        aggregation=config.aggregation,
    )


def ingest_report_path(config: RunConfig) -> Path:
    """Where `--ingest` leaves its stats, for `--run` to pick up.

    Keyed on the config name and not on the agent, because ingest happens
    once and all four agents read the corpus it produced.
    """
    return RESULTS_ROOT / f"{config.name}.ingest.json"


def _ingest_stats(config: RunConfig) -> dict[str, object]:
    """The ingest half of the cost column, read back from disk.

    `--ingest` and `--run` are separate processes -- deliberately, since an
    ingest is hours and a run is minutes -- so the run cannot observe what
    the ingest cost. It reads the file the ingest wrote.

    Returns `{}` when there is no file, rather than raising: a report with
    an empty ingest block is what every report had until now, and refusing
    to score a corpus someone ingested by hand would be a worse trade than
    an incomplete cost column. The absence is visible in the report either
    way, which is the property that matters.
    """
    path = ingest_report_path(config)
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _chat_model_tag(config: RunConfig) -> str:
    """A filename segment naming an overridden chat model, or nothing.

    Same argument as `_split_tag`: a run against a different model must not
    overwrite the number it should be compared against. Slashes and colons
    appear in model ids and are not filename characters.
    """
    if not config.chat_model_override:
        return ""
    safe = config.chat_model_override.replace("/", "-").replace(":", "-")
    return f"{safe}."


def _split_tag(config: RunConfig) -> str:
    """`"test."` on an overridden run, `""` otherwise -- and the asymmetry matters.

    Every result file written before `--split` existed is named
    `<config>.<agent>.json`. Tagging unconditionally would rename all of them,
    orphaning `RESULTS.md` and every path quoted in `FINDINGS.md`, to record
    something that did not vary. Tagging only an override keeps the default
    split's filenames stable and stops a 2,801-query run from silently
    overwriting the 280-query number it should be compared against.
    """
    return f"{config.split_override}." if config.split_override else ""


def predictions_path(config: RunConfig) -> Path:
    """Where this run's raw rankings land, written before anything scores them.

    Retrieval is the expensive half -- `deep` spends ~50 minutes of shared GPU
    on 280 queries -- and scoring is a subprocess that resolves `stark-qa` from
    PyPI on every invocation. A PyPI 502 therefore used to discard a completed
    run at the last step, with nothing on disk to score later. It has happened
    once, to `redstring-native/deep`. Persist first, score second, and
    `scripts/rescore.py` turns the survivor back into a report.
    """
    return (
        RESULTS_ROOT / f"{config.name}.{_split_tag(config)}{_chat_model_tag(config)}"
        f"{config.agent}.predictions.json"
    )


def report_path(config: RunConfig) -> Path:
    """Where this config-and-agent's numbers land.

    The agent is in the filename, not only in the file. One config serves all
    four architectures via `--agent`, so a path keyed on `config.name` alone
    would have each run overwrite the last -- and the survivor would carry
    the correct `config_verbatim` for whichever ran last, so nothing in the
    file would reveal the loss.
    """
    return RESULTS_ROOT / (
        f"{config.name}.{_split_tag(config)}{_chat_model_tag(config)}"
        f"{config.agent}.json"
    )


async def _do_ingest(
    config: RunConfig,
    *,
    ingest_edges: bool,
    embed_concurrency: int = 4,
    embed_batch: int = 64,
    limit: int | None = None,
    resume: bool = True,
    use_cache: bool = True,
) -> dict[str, object]:
    if config.embeddings != "precomputed-ada002" and config.embeddings not in (
        LIVE_EMBEDDINGS
    ):
        raise NotImplementedError(f"no ingest wiring for {config.embeddings!r}")
    if config.chunker not in CHUNKERS:
        raise NotImplementedError(f"unknown chunker {config.chunker!r}")

    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    vector_for = None
    embeddings = None
    if config.embeddings == "precomputed-ada002":
        logger.info(
            "loading precomputed doc embeddings from %s", data_dir / "doc_emb.npz"
        )
        doc_embeddings = read_doc_embeddings(data_dir / "doc_emb.npz")
        vector_for = node_vector_lookup(doc_embeddings)
    else:
        logger.info(
            "embedding live against %r (concurrency=%d)",
            config.embeddings,
            embed_concurrency,
        )
        embeddings = _live_embeddings_for(config)

    table = _table_for(config)
    chunks = await PostgresChunkStore.connect(
        POSTGRES_DSN, table=table, dimension=config.dimension
    )
    graph = Neo4jGraphStore.connect(NEO4J_URI, auth=NEO4J_AUTH)
    await chunks.ensure_schema()
    await graph.ensure_schema()

    # Only for live embedding. A precomputed-vector arm never calls a
    # provider, so a cache in front of one would be a table that is written
    # to and never read.
    embedding_cache = None
    if embeddings is not None and use_cache:
        embedding_cache = await PostgresEmbeddingCache.connect(POSTGRES_DSN)
        await embedding_cache.ensure_schema()
    try:
        nodes = read_nodes(data_dir / "nodes.jsonl")
        if limit is not None:
            nodes = itertools.islice(nodes, limit)
        edges = read_edges(data_dir / "edges.jsonl") if ingest_edges else iter(())

        # Resuming and holding an index are one decision, so the use case
        # takes one argument. `None` is "write everything".
        outcome = await ingest_corpus(
            engine=ingest,
            nodes=nodes,
            edges=edges,
            tenant_id=tenant_id,
            chunk_index=PostgresChunkIdIndex(POSTGRES_DSN, table) if resume else None,
            edges_ingested=ingest_edges,
            config_verbatim=config.raw,
            dataset=config.dataset,
            graph=graph,
            chunks=chunks,
            chunker=CHUNKERS[config.chunker](),
            vector_for=vector_for,
            embeddings=embeddings,
            concurrency=embed_concurrency if embeddings is not None else 1,
            embed_batch=embed_batch,
            total_nodes=_count_lines(data_dir / "nodes.jsonl"),
            embedding_cache=embedding_cache,
            # The key's other two fields. `config.embeddings` and
            # `config.document_prefix` are what `_table_for` already folds
            # into the table name (ADR 0002, ADR 0043) -- the cache has to
            # separate exactly the same things, or it serves one arm's
            # vectors to another.
            cache_model=config.embeddings,
            cache_document_prefix=config.document_prefix,
        )
    finally:
        await chunks.close()
        await graph.close()
        if embedding_cache is not None:
            await embedding_cache.close()

    return outcome.as_dict()


async def _do_run(config: RunConfig) -> None:
    if config.embeddings != "precomputed-ada002" and config.embeddings not in (
        LIVE_EMBEDDINGS
    ):
        raise NotImplementedError(f"no run wiring for {config.embeddings!r}")
    data_dir = _data_dir(config)
    tenant_id = _tenant_for(config)

    pairs = list(read_queries(data_dir / f"queries.{config.effective_split}.jsonl"))
    queries = [q for q, _ in pairs]
    answers = {q.query_id: a for q, a in pairs}

    if config.embeddings == "precomputed-ada002":
        logger.info(
            "loading precomputed query embeddings from %s", data_dir / "query_emb.npz"
        )
        query_vectors_by_id = read_query_embeddings(data_dir / "query_emb.npz")
        vectors_by_text = {
            q.text: query_vectors_by_id[str(q.query_id)]
            for q in queries
            if str(q.query_id) in query_vectors_by_id
        }
        embeddings = PrecomputedEmbeddingProvider(
            vectors_by_text, dimension=config.dimension
        )
    else:
        logger.info("embedding queries live against %r", config.embeddings)
        # Wrapped, then prewarmed *before* any store is opened. The live
        # path embedded one query per HTTP round-trip inside the retrieval
        # call, serialised by `run`'s bare loop; `dense` spent 78.7s on 280
        # queries that way, almost none of it compute. See
        # `PrewarmedQueryEmbeddings` for why this is a wrapper and not
        # concurrency in the runner.
        #
        # The precomputed branch is deliberately left alone: it is already a
        # dict lookup and never reaches an endpoint, so wrapping it would add
        # a second layer of counters over a path that makes no requests.
        embeddings = PrewarmedQueryEmbeddings(_live_embeddings_for(config))
        # Before `connect`, not after, so a slow prewarm does not sit on a
        # Postgres connection -- an ingest may be writing to the same
        # database, and CLAUDE.md records a scoring pass costing an in-flight
        # ingest 36% of its rate by contending there.
        await embeddings.prewarm([q.text for q in queries])

    chunks = await PostgresChunkStore.connect(
        POSTGRES_DSN, table=_table_for(config), dimension=config.dimension
    )
    graph = Neo4jGraphStore.connect(NEO4J_URI, auth=NEO4J_AUTH)
    await chunks.ensure_schema()
    await graph.ensure_schema()
    try:
        tools = toolset_for(
            chunks=chunks,
            graph=graph,
            embeddings=embeddings,
            config=config,
            tenant_id=tenant_id,
        )
        agent = build_agent(config)

        preds_path = predictions_path(config)
        # Wall time is measured around the whole query set rather than
        # derived from the calls, which overlap under concurrency.
        run_started = perf_counter()
        predictions = await run(
            agent,
            queries,
            tools,
            k=config.k,
            concurrency=config.query_concurrency,
            checkpoint=partial(write_predictions, preds_path),
        )
        run_wall_s = perf_counter() - run_started
        write_predictions(preds_path, predictions)

        candidates_path = data_dir / "candidates.json"
        candidate_ids = [int(c) for c in json.loads(candidates_path.read_text())]
        metrics = score_predictions(predictions, answers, candidate_ids=candidate_ids)
        cost = dict(
            summarise_cost(
                tools.calls,
                queries=len(queries),
                wall_s=run_wall_s,
                concurrency=config.query_concurrency,
            )
        )
        if isinstance(embeddings, PrewarmedQueryEmbeddings):
            # In the report because "the helper works, nobody calls it" has
            # happened twice in this repo, hours apart, with green tests both
            # times. `query_embed_live_calls: 0` on a dense run is the only
            # artifact that proves a byte of this reached the wire.
            cost.update(embeddings.stats())
    finally:
        await chunks.close()
        await graph.close()

    write_report(
        report_path(config),
        config=config,
        metrics=metrics,
        cost=cost,
        ingest=_ingest_stats(config),
        queries=len(queries),
    )
    print(metrics)  # noqa: T201


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, type=Path)
    parser.add_argument(
        "--summarise",
        nargs="?",
        const=Path("results"),
        type=Path,
        default=None,
        help="Render every scored arm under a directory as one markdown "
        "table on stdout and exit, without --config. Grouped by DATASET, "
        "which is load-bearing: vss-control embeds STaRK's add_rel=True "
        "documents and the prime arms do not, so a gap across that line is "
        "the corpus and not the model -- a conclusion this project drew "
        "wrongly once already.",
    )
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--agent",
        default=None,
        choices=sorted(AGENTS),
        help="Override the config's `agent:` for this invocation, so one "
        "config file serves all four architectures. The report filename "
        "carries the agent, so the four runs do not overwrite each other.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override the config's `split:` for this invocation, e.g. "
        "`--split test` for PRIME's full 2,801 queries instead of the 280 "
        "of `test-0.1`. The corpus is unaffected -- the tenant is a uuid5 "
        "of the config NAME, so both splits read the same ingested store "
        "and no re-ingest is needed. Only the query set changes. An "
        "overridden run tags its report filename with the split so it "
        "cannot overwrite the number it should be compared against, and "
        "writes the effective split into the report, because "
        "`config_verbatim` is the FILE's bytes and would otherwise name "
        "the split that did not run.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help=(
            "Chat model id, overriding the config's `chat_model:`. The "
            "report records the model that RAN and its filename is tagged "
            "with it, because `config_verbatim` is the config FILE's bytes "
            "and would name the model that did not. Uses the same corpus: "
            "the tenant is derived from the config NAME, so this does not "
            "re-ingest anything."
        ),
    )
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=1,
        help=(
            "queries in flight at once. A request occupies one server slot, "
            "so this should be at least the chat model's -np or the extra "
            "slots sit idle. Does not change accuracy -- queries are "
            "independent and predictions are keyed by query_id -- but it "
            "does change what contends on Postgres. Record it with any "
            "timing you report; the slot count changes mid-session."
        ),
    )
    parser.add_argument(
        "--ingest-edges",
        action="store_true",
        default=False,
        help="Also load edges into the graph store. Off by default: dense, "
        "hybrid and zero_shot retrieve through ChunkRetriever, which never "
        "touches the graph, so this costs ~16k transactions for no benefit "
        "to them. The deep agent does traverse -- run it against a corpus "
        "ingested with this flag, or its traversal actions find nothing.",
    )
    parser.add_argument(
        "--embed-concurrency",
        type=int,
        default=4,
        help="Embedding requests in flight at once. Set it to at least the "
        "server's -np: one request occupies one slot, so concurrency 1 "
        "against -np 4 leaves three quarters of the server idle. That was "
        "the SLOWEST of seven settings measured (1233 nodes/min even at "
        "batch 128) and simultaneously the one showing the highest GPU "
        "utilisation, because a kernel is resident whenever any slot is "
        "busy and three idle slots look like none. 4 against -np 4 gives "
        "1618. The endpoint is shared -- confirm spare capacity before "
        "raising it. Ignored for precomputed-embeddings configs.",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Chunk texts per embedding request. Worth about 18%% end to end "
        "on this corpus -- 1368 nodes/min at 1 text per request against 1612 "
        "at 64, both at concurrency 8. A standalone probe on ONE SERIAL "
        "CONNECTION showed 298 against 1850 over the same range, which is a "
        "misleading comparison and is recorded here so nobody repeats it: "
        "the ingest is not serial, and pipelining had already recovered most "
        "of that gap before a single text was batched.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N nodes. For calibrating throughput "
        "against a live endpoint before committing to a full run -- never "
        "use it for a config whose numbers get reported.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ingest is resumable by default: a node whose chunk already "
        "exists in the store (same source and text -- an unchanged node) is "
        "skipped rather than re-embedded, so an interrupted ingest restarts "
        "cheaply instead of re-embedding from zero. Pass --no-resume to "
        "force a full re-ingest of every node regardless of what the store "
        "already holds -- the escape hatch, because skipping work is "
        "exactly the kind of optimisation that can hide a bug.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Re-embed every chunk instead of reusing vectors this benchmark "
        "has already computed for the same (model, document_prefix, text). "
        "The cache is on by default and is what makes a chunking sweep cost "
        "roughly one endpoint pass rather than three -- ~80%% of these "
        "corpora are short enough that every chunker emits identical text. "
        "Pass this to force a cold run, for the same reason --no-resume "
        "exists: reusing work is exactly the kind of optimisation that can "
        "hide a bug.",
    )
    args = parser.parse_args()

    if args.summarise is not None:
        print(summarise(args.summarise))  # noqa: T201
        return

    if args.config is None:
        parser.error("--config is required unless --summarise is given")

    config = load_config(args.config)
    if args.agent is not None:
        config = replace(config, agent=args.agent)
    if args.chat_model is not None and args.chat_model != config.chat_model:
        config = replace(config, chat_model_override=args.chat_model)
    if args.query_concurrency != config.query_concurrency:
        config = replace(config, query_concurrency=args.query_concurrency)
    if args.split is not None and args.split != config.split:
        # Only when it differs. `--split test-0.1` on a config that already
        # says `test-0.1` must not tag the filename, or the same run acquires
        # two names depending on how it was invoked.
        config = replace(config, split_override=args.split)

    # Neither flag means neither phase runs, and the process exits 0 having
    # done nothing. That is not hypothetical: a run queue passed
    # `--agent dense` without `--run`, and four arms "completed" in one
    # second each with rc=0 and empty logs. The queue's own gate is on the
    # return code, so it accepted all four and moved on.
    #
    # `--agent` in particular reads like an instruction to run something. It
    # is only an override of the config's agent, and on its own it is inert.
    if not args.ingest and not args.run:
        parser.error(
            "nothing to do: pass --ingest, --run, or both. "
            "--agent only overrides the configured agent and does not run it."
        )

    if args.ingest:
        report = asyncio.run(
            _do_ingest(
                config,
                ingest_edges=args.ingest_edges,
                embed_concurrency=args.embed_concurrency,
                embed_batch=args.embed_batch,
                limit=args.limit,
                resume=not args.no_resume,
                use_cache=not args.no_cache,
            )
        )
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        ingest_report_path(config).write_text(json.dumps(report, indent=2))
        print(report)  # noqa: T201

    if args.run:
        asyncio.run(_do_run(config))


if __name__ == "__main__":
    main()

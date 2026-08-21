"""Turn a directory of result files into one table, grouped by corpus.

Closes B-SUMMARISE-1, but with a constraint that entry did not know about.

## Why grouping by corpus is the whole point

Arms in this directory do not all index the same text, and nothing in the
file format says so. `data/prime` documents stop at the node's own details;
`data/prime-rel` documents append a `- relations:` block naming every
neighbour. PRIME's queries name related entities explicitly ("a drug that
targets X and is indicated for Y"), so those names appear in the answer's own
document in one corpus and not the other. A gap across that line is the
corpus, not the model.

So this renders **one section per dataset**, and a cross-corpus comparison
requires deliberately reading two sections rather than two adjacent lines.

**A caveat this grouping does NOT cover, stated because it bit already.**
`vss-control` declares `dataset: prime` and therefore lands in the `prime`
section, but its vectors are STaRK's precomputed ada-002 embeddings rather
than anything derived from `data/prime/nodes.jsonl`. Whether STaRK generated
those with `add_rel=True` is **unverified**: `stark_qa`'s `multi_vss.py` and
`llm_reranker.py` do pass `add_rel=True`, but `VSS` itself only loads a
precomputed directory and `bm25.py` uses the `add_rel=False` default. The
generation script is not in the pip package.

That matters because `qwen-wholedoc` at 0.183 dense was read against
`vss-control` at 0.231 as evidence qwen3-embedding is a weak model, and the
reading is only sound if both indexed comparable text. `qwen-rel-whole`
settles it from our side: if relational text is the gap it should move
substantially, and if it does not then ada-002 is simply the better model on
this corpus. Until that number exists, treat the `vss-control` rows as a
reference point of uncertain provenance rather than a controlled comparison.

That is a weaker guarantee than a hard error, and deliberately so: the arms
are not wrong, and there are real questions that span corpora -- "does the
reranking gain survive a different corpus" is the reason `mag-wholedoc`
exists. What must not happen is the comparison being made *by accident*.

## Two seconds columns, because they stopped being the same number

`seconds_total` sums per-call durations; `seconds_wall` measures elapsed
time around the query set. They were identical while `run_queries.run` was
serial. Under `--query-concurrency N` the calls overlap and the sum counts
the same seconds up to N times -- one run reported 1933s while taking ~480s.

Both are kept because they answer different questions. Compute consumed is
what an arm costs the shared endpoint, and it stays comparable across
concurrencies precisely because it does not shrink when slots are added.
Elapsed time is how long before you have the number, and it is NOT
comparable between arms run at different concurrencies -- so `conc` is
rendered beside it. A wall-seconds column without it invites exactly the
comparison it cannot support.

Arms scored before this existed render `--`, which is honest: their wall
time was not recorded and, being serial, equalled their gpu seconds anyway.

## `cut off` is not a cost, it is a caveat

`exhausted_queries` counts queries that ended at the budget cap rather than
because the agent decided it was finished. A `deep` arm where most queries
are cut off has an accuracy number about `MAX_TOOL_CALLS` and not about the
architecture, and nothing else in the row would say so.

`--` means the agent has no budget to exhaust -- `dense` cannot be cut off,
and rendering 0 there would claim it ran to completion under a cap it does
not have.

## Cost sits beside accuracy

The plan this repo was built from asks for accuracy **and** cost per
architecture, because a number without its cost is not actionable: `deep` at
eight LLM calls per query and `dense` at one are not interchangeable at equal
mrr. `tokens_per_query` is `int | None`, where `None` means "not measured"
and is rendered as `--` rather than `0` -- the distinction `ToolCall.tokens`
exists to preserve.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: Ingest reports, not scoring reports. `--ingest` writes
#: `<name>.ingest.json` beside the per-agent files and it has no `metrics`.
_INGEST_SUFFIX = ".ingest.json"

#: Raw rankings, written before scoring so a sidecar failure cannot discard a
#: completed run. Also not a scoring report.
_PREDICTIONS_SUFFIX = ".predictions.json"

_METRICS = ("mrr", "hit@1", "hit@5", "recall@20")


@dataclass(frozen=True, slots=True)
class Row:
    """One scored arm: the config, the agent, and what it cost to get there."""

    config: str
    agent: str
    dataset: str
    chunker: str
    embeddings: str
    chat_model: str
    metrics: dict[str, float]
    cost: dict[str, object]
    ingest: dict[str, object]

    @property
    def chunks_per_node(self) -> float | None:
        nodes = self.ingest.get("nodes")
        chunks = self.ingest.get("chunks")
        skipped = self.ingest.get("skipped", 0)
        if not isinstance(nodes, int) or not isinstance(chunks, int) or not nodes:
            return None
        # `chunks` counts writes and `skipped` counts ids already present, so
        # a resumed arm reports far too few unless both are counted. An arm
        # once rendered 0.381 into RESULTS.md for a corpus whose real
        # granularity was 1.058 -- below the 1.0 every chunker here must
        # exceed, and it still rendered without comment.
        return (chunks + (skipped if isinstance(skipped, int) else 0)) / nodes


def _config_field(verbatim: str, field: str) -> str:
    """Read one scalar out of the embedded config, without a YAML parser.

    `config_verbatim` is the config file's own bytes, so this is the arm's
    ground truth rather than a filename convention -- two configs could name
    the same dataset and a filename cannot say so. Comment lines are skipped
    because these configs carry long ones that mention the same keys in prose.
    """
    for line in verbatim.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        if key.strip() == field:
            return value.strip().strip('"').strip("'")
    return ""


def read_rows(directory: Path) -> list[Row]:
    """Every scored arm under `directory`, newest-agnostic and order-stable.

    A file without `metrics` is skipped rather than raising: the directory
    also holds ingest reports and raw predictions, and a summariser that
    refused to run because a sibling file had a different shape would be
    useless exactly when the directory is busiest.
    """
    rows: list[Row] = []
    for path in sorted(directory.glob("*.json")):
        name = path.name
        if name.endswith(_INGEST_SUFFIX) or name.endswith(_PREDICTIONS_SUFFIX):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = report.get("metrics")
        if not isinstance(metrics, dict) or "mrr" not in metrics:
            continue

        stem = name[: -len(".json")]
        config_name = report.get("config_name") or stem.rsplit(".", 1)[0]
        # The agent lives in the filename and nowhere in the file. That is
        # load-bearing: one config serves every agent via `--agent`, so the
        # config's own `agent:` is the default and not what ran.
        agent = stem.rsplit(".", 1)[1] if "." in stem else "?"
        verbatim = report.get("config_verbatim", "")
        rows.append(
            Row(
                config=str(config_name),
                agent=agent,
                dataset=_config_field(verbatim, "dataset") or "unknown",
                chunker=_config_field(verbatim, "chunker") or "?",
                embeddings=_config_field(verbatim, "embeddings") or "?",
                # The chat model that RAN, recorded by the report rather
                # than read from `config_verbatim` -- `--chat-model`
                # overrides the file, and two arms differing only by it
                # were otherwise indistinguishable in this table.
                chat_model=str(
                    report.get("chat_model")
                    or _config_field(verbatim, "chat_model")
                    or "--"
                ),
                metrics={
                    k: v for k, v in metrics.items() if isinstance(v, int | float)
                },
                cost=report.get("cost") or {},
                ingest=report.get("ingest") or {},
            )
        )
    return rows


def _fmt(value: object, places: int = 5) -> str:
    if value is None:
        # Not zero. `tokens_per_query` is `int | None` precisely so that
        # "not measured" and "measured as none" stay distinguishable.
        return "--"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render(rows: Iterable[Row]) -> str:
    """One markdown document, one section per dataset."""
    rows = list(rows)
    out: list[str] = [
        "# Results",
        "",
        "Generated by `--summarise`. One section per **corpus**, and that "
        "grouping is load-bearing rather than cosmetic.",
        "",
        "Arms on `prime` index documents that stop at the node's own "
        "details; arms on `prime-rel` index documents that also name every "
        "neighbour. PRIME's queries name related entities explicitly, so a "
        "gap across that line is the **corpus**, not the model. Compare "
        "within a section freely; across sections, only deliberately.",
        "",
        "**`vss-control` is a reference point, not a controlled comparison.** "
        "It declares `dataset: prime` and so appears in that section, but its "
        "vectors are STaRK's precomputed ada-002 embeddings, and whether "
        "STaRK generated them over `add_rel=True` documents is unverified -- "
        "`multi_vss` and `llm_reranker` pass `add_rel=True`, while `VSS` "
        "itself only loads a precomputed directory and `bm25` uses the "
        "`add_rel=False` default. Read a gap to it with that in mind.",
        "",
    ]
    if not rows:
        out += ["_No scored arms found._", ""]
        return "\n".join(out)

    for dataset in sorted({row.dataset for row in rows}):
        section = [row for row in rows if row.dataset == dataset]
        out += [f"## `{dataset}`", ""]
        out += [
            "| config | agent | embed model | chat model | chunker "
            "| chunks/node | mrr | hit@1 "
            "| hit@5 | recall@20 | llm calls/query | tokens/query "
            "| gpu seconds | wall seconds | conc | cut off |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(
            section, key=lambda r: (-r.metrics.get("mrr", 0.0), r.config, r.agent)
        ):
            cpn = row.chunks_per_node
            out.append(
                f"| `{row.config}` | {row.agent} | {row.embeddings} "
                f"| {row.chat_model} | {row.chunker} | {_fmt(cpn, 3)} | "
                + " | ".join(_fmt(row.metrics.get(m)) for m in _METRICS)
                + f" | {_fmt(row.cost.get('llm_calls_per_query'), 2)} "
                f"| {_fmt(row.cost.get('tokens_per_query'))} "
                f"| {_fmt(row.cost.get('seconds_total'), 1)} "
                f"| {_fmt(row.cost.get('seconds_wall'), 1)} "
                f"| {_fmt(row.cost.get('query_concurrency'))} "
                f"| {_fmt(row.cost.get('exhausted_queries'))} |"
            )
        out.append("")
    return "\n".join(out)


def summarise(directory: Path) -> str:
    return render(read_rows(directory))

"""Generate the questions each node's document answers, once, to a file.

## The hypothesis

Every dense number in this project is weak -- 0.18 to 0.25 mrr -- and
adding relational text moved dense by **+2%** while moving lexical by
**+22%**. The standing reading was that `qwen3-embedding-0.6b` is a
mediocre model. There is a better explanation available: PRIME documents
are **records**, not prose (`- name:`, `- type:`, `- details:`), and STaRK
queries are natural-language questions. A bi-encoder is being asked to put
a question and a database record in one neighbourhood, and no task prefix
fixes that asymmetry.

Generating the questions a document answers makes the match symmetric:
question against question. That is the mechanism, and it is why this should
move the channel relational text could not.

It should help the **lexical** channel too, which is where this project's
gains have actually come from: generated questions put query-shaped
vocabulary into the terms table. doc2query found the same on MS MARCO.

## Why this is a separate pass writing a file

The LLM work is the expensive part -- one call per node, and PRIME is
129,656 of them. Writing `questions.jsonl` means it is done once and then
re-read by every downstream experiment: filtered differently, ingested into
different tenants, or inspected by hand. Folding generation into the ingest
would re-run it on every re-ingest, and the ingest's resume path exists to
skip work, not to redo the costliest part of it.

It also makes the output **auditable before it is indexed**, which matters
because the failure mode here is not an error. If the model emits "What is
this protein's function?" for ten thousand nodes, the index fills with
near-identical vectors, discrimination gets *worse*, and every count in
every report stays clean. `scripts/score_question_signal.py` is the check;
this script only has to make the questions inspectable.

## Resumable, because it will be interrupted

Appends one JSON line per node and skips node ids already present. The
endpoint here is shared with the user's other work and has been restarted
mid-run several times in one session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from stark_bench.adapters.config_file import load_config
from stark_bench.adapters.stark_artifacts import read_nodes
from stark_bench.composition.cli import _data_dir, _llm_for

logger = logging.getLogger("generate_questions")

#: Characters of the record shown to the generator.
#:
#: Measured, not guessed. Against `gemma-4-26b-qat`, the whole prompt at
#: 2,000 characters came to 293-909 tokens -- 909 even for the corpus's
#: 27,629-character maximum, because the cap bounds it. At ~3.2 characters
#: per token here (PRIME carries identifiers that tokenise far worse than
#: the 4.3 of prose), 6,000 characters is ~2,700 tokens: still tiny against
#: a 65,536-token context, and inside a 4,096-token slot if `-np` is raised
#: to 16.
#:
#: 6,000 rather than 2,000 because the corpus that matters is `prime-rel`,
#: whose p90 document is 4,923 characters against plain `prime`'s 1,338.
#: The relations block is the part worth generating from -- it names the
#: neighbours STaRK queries ask about -- and a 2,000-character cap would
#: truncate it away on exactly the nodes where it exists.
#:
#: Prefill is not the cost here anyway: four questions is ~150 decode
#: tokens, and decode is sequential where prefill is one batched pass.
_MAX_DOCUMENT_CHARS = 6_000


class Questions(BaseModel):
    """The questions this record answers.

    Field descriptions are load-bearing. On 2026-08-20 a reranker returned
    `{"scores": []}` for most queries after schema docstrings were stripped
    to save tokens -- the description was the only text telling the model
    what to put in the array. Do not trim these.
    """

    questions: list[str] = Field(
        description=(
            "Questions a researcher could ask that this specific entity "
            "answers. Each must name the entity's own distinguishing "
            "details -- never a generic question that thousands of other "
            "entities would answer equally well."
        )
    )


_PROMPT = (
    "Below is a record for one biomedical entity from a knowledge base.\n\n"
    "Write {count} questions that a researcher might ask, where THIS entity "
    "is the answer.\n\n"
    "RULES:\n"
    "1. Each question must be answerable from the record below.\n"
    "2. Use the entity's DISTINGUISHING details -- its specific function, "
    "the diseases or drugs it relates to, its family, its mechanism. A "
    "question like 'What is this gene?' or 'What does this protein do?' is "
    "useless, because thousands of other entities answer it equally well.\n"
    "3. Do NOT name the entity itself in the question. Somebody asking the "
    "question does not know the answer yet.\n"
    "4. Write the way a researcher searches: a full natural-language "
    "question, not keywords.\n\n"
    "Record:\n{document}"
)


async def _one(node, llm, count: int) -> dict | None:  # noqa: ANN001
    try:
        result = await llm.extract(
            _PROMPT.format(count=count, document=node.document[:_MAX_DOCUMENT_CHARS]),
            Questions,
        )
    except Exception:
        # Logged and skipped, not fatal: a run over 129,656 nodes must not
        # die on one. A node with no questions keeps its own document chunk
        # and stays retrievable, so the cost is a missed enrichment rather
        # than a lost entity.
        logger.warning("questions: extract failed for node %s", node.node_id)
        return None
    wanted = [q.strip() for q in result.questions if q.strip()]
    if not wanted:
        logger.warning("questions: empty list for node %s", node.node_id)
        return None
    return {"node_id": node.node_id, "questions": wanted[:count]}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--chat-model", default="gemma-4-26b-qat")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    config = load_config(args.config)
    object.__setattr__(config, "chat_model", args.chat_model)
    out = _data_dir(config) / f"questions.{args.chat_model}.jsonl"

    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["node_id"])
        logger.info("resuming: %s already has %d nodes", out.name, len(done))

    nodes = [n for n in read_nodes(_data_dir(config) / "nodes.jsonl")]
    if args.limit:
        nodes = nodes[: args.limit]
    todo = [n for n in nodes if n.node_id not in done]
    logger.info("%d nodes, %d to do", len(nodes), len(todo))
    if not todo:
        return 0

    llm = _llm_for(config)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    written = 0
    failed = 0

    async def bounded(node):  # noqa: ANN001, ANN202
        async with semaphore:
            return await _one(node, llm, args.count)

    with out.open("a", encoding="utf-8") as handle:
        # 25, not 200. The batch size is how much work an interruption
        # throws away, and this endpoint is shared and has been restarted
        # four times in one session -- a 200-node batch lost ten minutes of
        # generation that had already been paid for. Small batches cost
        # nothing here: the flush is a write, and `concurrency` requests
        # stay in flight regardless.
        for chunk_start in range(0, len(todo), 25):
            batch = todo[chunk_start : chunk_start + 25]
            for result in await asyncio.gather(*(bounded(n) for n in batch)):
                if result is None:
                    failed += 1
                    continue
                handle.write(json.dumps(result) + "\n")
                written += 1
            handle.flush()
            elapsed = time.perf_counter() - started
            rate = written / elapsed * 60 if elapsed else 0
            remaining = (len(todo) - written - failed) / rate if rate else 0
            logger.info(
                "%d/%d written, %d failed, %.0f nodes/min, ~%.0f min left",
                written,
                len(todo),
                failed,
                rate,
                remaining,
            )

    # Loud, because "0 failed" and "0 attempted" are the same exit status
    # and this project has been caught by that difference nine times.
    logger.info("done: %d written, %d failed, into %s", written, failed, out)
    if failed:
        logger.warning(
            "%d node(s) produced no questions -- they keep their own document "
            "chunk and stay retrievable, but are not enriched",
            failed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

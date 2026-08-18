# STaRK Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure redstring against the STaRK retrieval benchmark, with agent architectures as a plug-in point.

**Architecture:** STaRK's pre-built knowledge base is exported to neutral artifacts by a sidecar (stark-qa in an isolated 3.11 env), loaded into redstring's stores through its public write ports, and queried by pluggable agents through an instrumented reader-only toolset. Scoring is done by STaRK's own evaluator, never by us.

**Tech Stack:** Python 3.13, uv, redstring (pinned), pytest, hypothesis, ruff, import-linter, pre-commit, Postgres (chunk store) + Neo4j (graph), stark-qa via `uv run --with` on 3.11.

**Spec:** `docs/superpowers/specs/2026-08-18-stark-benchmark-harness-design.md`

## Global Constraints

- Python **3.13** for the harness. `requires-python = ">=3.13"`.
- **uv only.** Never hand-edit `[project.dependencies]`; use `uv add`, `uv add --dev`, `uv remove`. Re-sync with `--all-extras` after any dependency change.
- **stark-qa is never a harness dependency.** It runs only in the sidecar via `uv run --python 3.11 --with stark-qa`.
- **We compute no retrieval metric ourselves.** All metrics come from `stark_qa.evaluator.Evaluator`.
- **`RRF_K` is never tuned or exposed.** Redstring's constant (60) stands.
- **`agents/` may import `stark_bench.ports` and nothing else from `stark_bench`.** Enforced by import-linter.
- Redstring production adapters are imported by dotted path against a pinned version: `redstring.vector.adapters.pgvector`, `redstring.graph.adapters.neo4j`, `redstring.chunks.adapters.postgres`.
- Every results file embeds its resolved config verbatim.
- All tests: `uv run pytest -p no:randomly <path> -v` while iterating; full suite before commit.

---

### Task 1: Project skeleton, tooling, and the gate for the gate

**Files:**
- Create: `src/stark_bench/__init__.py`, `tests/__init__.py`
- Create: `.pre-commit-config.yaml`, `docker-compose.yml`
- Modify: `pyproject.toml`
- Test: `tests/test_pre_commit_hook_is_installed.py`, `tests/test_redstring_adapter_paths_resolve.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed pre-commit hook; import-linter contract `agents-may-only-import-ports`; a pinned redstring.

- [ ] **Step 1: Add dependencies**

```bash
cd ~/workspace/stark-bench
uv add "redstring @ file:///home/ty/workspace/redstring"
uv add --dev pytest pytest-asyncio pytest-randomly hypothesis ruff import-linter pre-commit
uv sync --all-extras
```

- [ ] **Step 2: Write the failing hook-installed test**

This is the gate for the gate. An absent hook is indistinguishable from a passing one — nothing is printed either way.

```python
# tests/test_pre_commit_hook_is_installed.py
import os
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".git" / "hooks" / "pre-commit"


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="CI runs the tools as separate jobs")
def test_the_pre_commit_hook_is_installed():
    """Match on `hook-impl`, from pre-commit's generated body.

    Matching on the string "pre-commit" would also match git's own
    `pre-commit.sample`, so copying the sample into place would pass.
    """
    assert HOOK.exists(), "run: uv run pre-commit install"
    assert "hook-impl" in HOOK.read_text()
```

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest tests/test_pre_commit_hook_is_installed.py -v -p no:randomly`
Expected: FAIL — `run: uv run pre-commit install`

- [ ] **Step 4: Write `.pre-commit-config.yaml` and install**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: lint-imports
        name: lint-imports
        entry: uv run lint-imports
        language: system
        pass_filenames: false
```

```bash
uv run pre-commit install
```

- [ ] **Step 5: Run the test to verify it passes, then break it on purpose**

```bash
uv run pytest tests/test_pre_commit_hook_is_installed.py -v -p no:randomly   # PASS
mv .git/hooks/pre-commit /tmp/hook.bak
uv run pytest tests/test_pre_commit_hook_is_installed.py -v -p no:randomly   # must FAIL
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit 2>/dev/null || true
uv run pytest tests/test_pre_commit_hook_is_installed.py -v -p no:randomly   # must FAIL (sample is not hook-impl)
mv /tmp/hook.bak .git/hooks/pre-commit
uv run pytest tests/test_pre_commit_hook_is_installed.py -v -p no:randomly   # PASS
```

A gate whose happy path is "the file is there" must be broken on purpose before it is believed.

- [ ] **Step 6: Write the adapter-path guard test**

```python
# tests/test_redstring_adapter_paths_resolve.py
"""Redstring's production adapters are not in `redstring.__all__`.

We import them by dotted path against a pinned version. This test turns a
version bump that moves them into a loud failure here rather than a confusing
one during a two-hour ingest.
"""
import importlib

import pytest

PATHS = [
    ("redstring.vector.adapters.pgvector", "PgVectorStore"),
    ("redstring.graph.adapters.neo4j", "Neo4jGraphStore"),
    ("redstring.chunks.adapters.postgres", "PostgresChunkStore"),
]


@pytest.mark.parametrize(("module", "name"), PATHS)
def test_adapter_path_still_resolves(module, name):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module}.{name} moved; the redstring pin needs revisiting"
```

- [ ] **Step 7: Run it**

Run: `uv run pytest tests/test_redstring_adapter_paths_resolve.py -v -p no:randomly`
Expected: PASS. If `PostgresChunkStore` is named differently, fix the constant to the real exported name — read `src/redstring/chunks/adapters/postgres.py` in the redstring checkout, do not guess.

- [ ] **Step 8: Add the import-linter contract to `pyproject.toml`**

```toml
[tool.importlinter]
root_packages = ["stark_bench"]

[[tool.importlinter.contracts]]
name = "agents may only import ports"
type = "forbidden"
source_modules = ["stark_bench.agents"]
forbidden_modules = [
    "stark_bench.harness",
    "stark_bench.skb",
    "stark_bench.sidecar",
]
```

An agent with a path to the runner has a path to ground truth.

- [ ] **Step 9: Prove the contract bites**

```bash
mkdir -p src/stark_bench/agents && touch src/stark_bench/agents/__init__.py
mkdir -p src/stark_bench/harness && touch src/stark_bench/harness/__init__.py
echo "from stark_bench.harness import *  # noqa" > src/stark_bench/agents/_probe.py
uv run lint-imports   # must FAIL
rm src/stark_bench/agents/_probe.py
uv run lint-imports   # must PASS
```

A passing check you have never seen fail is not yet evidence.

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "Project skeleton, with both gates broken on purpose first"
```

---

### Task 2: The agent seam (`ports.py`)

**Files:**
- Create: `src/stark_bench/ports.py`
- Test: `tests/test_ports.py`

**Interfaces:**
- Produces: `Query`, `Ranked`, `Agent`, `Toolset`, `ToolCall`. Every later task depends on these exact names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ports.py
from stark_bench.ports import Agent, Query, Ranked


def test_ranked_is_a_node_id_and_a_score():
    r = Ranked(node_id="12345", score=0.5)
    assert r.node_id == "12345"
    assert r.score == 0.5


def test_query_carries_no_answer():
    """An agent must never be able to see ground truth."""
    q = Query(query_id=7, text="which drugs target PTGS2?")
    assert not hasattr(q, "answer_ids")
    assert q.query_id == 7


def test_agent_is_runtime_checkable():
    class Stub:
        async def retrieve(self, query, tools):
            return []

    assert isinstance(Stub(), Agent)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_ports.py -v -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named 'stark_bench.ports'`

- [ ] **Step 3: Implement**

```python
# src/stark_bench/ports.py
"""The seam between the harness and an agent.

`agents/` may import this module and nothing else from `stark_bench`. That is
enforced by import-linter, not by convention: an agent that can reach the
runner can reach `answer_ids`, and a retrieval agent that can see the answers
is one accidental import away from a perfect score.

`Query` therefore carries no answer field at all. The restriction is
structural rather than a matter of discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Query:
    """One STaRK query. Deliberately has no `answer_ids`."""

    query_id: int
    text: str


@dataclass(frozen=True, slots=True)
class Ranked:
    """One scored candidate, in `pred_dict` shape.

    `node_id` is STaRK's id as a string, not a redstring `EntityId`: this is
    what the official evaluator consumes, so nothing is reshaped between an
    agent and scoring.
    """

    node_id: str
    score: float


@dataclass(slots=True)
class ToolCall:
    """One recorded call. Cost is a reported metric, not a footnote."""

    tool: str
    duration_s: float
    result_count: int
    tokens: int = 0


@runtime_checkable
class Toolset(Protocol):
    """Reader-only access to the knowledge base.

    Reader-only is a type-level guarantee, the same argument redstring's own
    `Retriever` makes for holding `VectorReader` rather than `VectorStore`: an
    agent that cannot reach a writer cannot poison the KB mid-run.
    """

    calls: list[ToolCall]

    async def search_chunks(self, text: str, *, k: int = 10, mode: str = "hybrid") -> list[Ranked]: ...
    async def get_node(self, node_id: str) -> dict[str, object] | None: ...
    async def neighbors(self, node_id: str, *, depth: int = 1) -> list[str]: ...
    async def get_relationships(self, node_id: str) -> list[tuple[str, str, str]]: ...
    async def complete(self, prompt: str) -> str: ...


@runtime_checkable
class Agent(Protocol):
    """Given a query and tools, return ranked STaRK node ids."""

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]: ...
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_ports.py -v -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "The agent seam: ranked node ids in, no ground truth reachable"
```

---

### Task 3: Deterministic id mapping

**Files:**
- Create: `src/stark_bench/skb/__init__.py`, `src/stark_bench/skb/ids.py`
- Test: `tests/skb/test_ids.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `entity_id_for(dataset: str, node_id: str) -> EntityId`, `NAMESPACE_STARK: UUID`, `STARK_ID_KEY: str = "stark_node_id"`, `node_id_of(entity) -> str`.

- [ ] **Step 1: Write the failing test**

Note the collision test. Redstring's own notes record five separate defects where a composite key was compared on one component, every one hidden by ids drawn from `uuid4()`. The key here is `(dataset, node_id)`, so the collision to force is the same id under two datasets.

```python
# tests/skb/test_ids.py
import pytest

from stark_bench.skb.ids import entity_id_for, node_id_of, STARK_ID_KEY


def test_the_same_node_maps_to_the_same_id_every_time():
    assert entity_id_for("prime", "4242") == entity_id_for("prime", "4242")


def test_the_dataset_is_part_of_the_key():
    """Both components decide the id.

    Written because a mapping that hashed `node_id` alone would pass every
    other test in this file: no fixture uses two datasets at once.
    """
    assert entity_id_for("prime", "4242") != entity_id_for("mag", "4242")


def test_different_nodes_in_one_dataset_differ():
    assert entity_id_for("prime", "1") != entity_id_for("prime", "2")


def test_the_reverse_map_is_a_field_read():
    class FakeEntity:
        external_ids = {STARK_ID_KEY: "4242"}

    assert node_id_of(FakeEntity()) == "4242"


def test_a_node_without_the_key_is_an_error_not_a_none():
    class FakeEntity:
        external_ids: dict[str, str] = {}

    with pytest.raises(KeyError):
        node_id_of(FakeEntity())
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/skb/test_ids.py -v -p no:randomly`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/stark_bench/skb/ids.py
"""STaRK node ids to redstring entity ids, deterministically.

Deterministic means ingest is idempotent and resumable, and the forward map
needs no stored side-table that could drift from the data.

The reverse direction rides `Entity.external_ids`, a first-class field on the
domain type. No reader port offers lookup-by-external-id, and none is needed:
retrieval hands back whole `Entity` objects, so the reverse map is a field
read on an object we already hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid5

from redstring import EntityId

if TYPE_CHECKING:
    pass

#: Fixed namespace. Changing it re-keys every entity, so it is a constant and
#: never a parameter.
NAMESPACE_STARK = UUID("6f2a1d54-7c3b-5e19-9a4f-2b8c0d1e3f57")

STARK_ID_KEY = "stark_node_id"


class HasExternalIds(Protocol):
    external_ids: dict[str, str]


def entity_id_for(dataset: str, node_id: str) -> EntityId:
    """Map one STaRK node to a redstring entity id.

    Both components are in the key. A map over `node_id` alone would collide
    the moment a second dataset is ingested into the same store.
    """
    return EntityId(uuid5(NAMESPACE_STARK, f"{dataset}:{node_id}"))


def node_id_of(entity: HasExternalIds) -> str:
    """Recover the STaRK node id. Raises `KeyError` if absent.

    Absent means the loader did not populate it, which is a bug worth failing
    on rather than a `None` that silently drops a result from the ranking.
    """
    return entity.external_ids[STARK_ID_KEY]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/skb/test_ids.py -v -p no:randomly`
Expected: PASS (5 tests)

- [ ] **Step 5: Break it on purpose**

Temporarily change `f"{dataset}:{node_id}"` to `node_id`. `test_the_dataset_is_part_of_the_key` must fail. Restore it. A property that stays green under a deliberate defect is worse than no property.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Deterministic STaRK node id mapping, keyed on both components"
```

---

### Task 4: The fixture SKB and the neutral artifact format

**Files:**
- Create: `src/stark_bench/skb/artifacts.py`
- Create: `tests/fixtures/tiny_skb/{nodes.jsonl,edges.jsonl,queries.jsonl}`
- Test: `tests/skb/test_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SkbNode`, `SkbEdge`, `SkbQuery`, `read_nodes(path) -> Iterator[SkbNode]`, `read_edges(path) -> Iterator[SkbEdge]`, `read_queries(path) -> Iterator[tuple[SkbQuery, list[str]]]`.

The neutral format is what the sidecar writes and the harness reads. Defining it now means Task 10's sidecar has a target, and every earlier task can run on the fixture without downloading 129k nodes.

- [ ] **Step 1: Write the fixture files**

A dozen nodes with known answers. Deliberately includes a self-loop (dropped later, with a recorded count) and a node whose document is long enough to chunk into more than one piece.

```
# tests/fixtures/tiny_skb/nodes.jsonl
{"node_id": "1", "node_type": "drug", "name": "aspirin", "document": "Aspirin is a salicylate that irreversibly inhibits cyclooxygenase. It is used for pain, fever and inflammation, and at low dose for cardiovascular prophylaxis. Long text follows to force more than one chunk. " }
{"node_id": "2", "node_type": "gene", "name": "PTGS2", "document": "PTGS2 encodes cyclooxygenase-2, an enzyme induced during inflammation."}
{"node_id": "3", "node_type": "gene", "name": "PTGS1", "document": "PTGS1 encodes cyclooxygenase-1, expressed constitutively in most tissues."}
{"node_id": "4", "node_type": "disease", "name": "inflammation", "document": "Inflammation is a protective response involving immune cells and mediators."}
{"node_id": "5", "node_type": "drug", "name": "ibuprofen", "document": "Ibuprofen is a nonselective NSAID inhibiting cyclooxygenase enzymes reversibly."}
{"node_id": "6", "node_type": "drug", "name": "celecoxib", "document": "Celecoxib is a selective COX-2 inhibitor used in arthritis."}
{"node_id": "7", "node_type": "pathway", "name": "prostaglandin synthesis", "document": "The prostaglandin synthesis pathway converts arachidonic acid to prostaglandins."}
{"node_id": "8", "node_type": "disease", "name": "arthritis", "document": "Arthritis is joint inflammation causing pain and stiffness."}
{"node_id": "9", "node_type": "gene", "name": "TNF", "document": "TNF encodes tumour necrosis factor, a pro-inflammatory cytokine."}
{"node_id": "10", "node_type": "drug", "name": "paracetamol", "document": "Paracetamol is an analgesic and antipyretic with weak anti-inflammatory action."}
{"node_id": "11", "node_type": "pathway", "name": "cytokine signalling", "document": "Cytokine signalling coordinates immune cell communication."}
{"node_id": "12", "node_type": "disease", "name": "fever", "document": "Fever is an elevated body temperature, often mediated by prostaglandins."}
```

```
# tests/fixtures/tiny_skb/edges.jsonl
{"source": "1", "target": "2", "relation": "targets"}
{"source": "1", "target": "3", "relation": "targets"}
{"source": "5", "target": "2", "relation": "targets"}
{"source": "5", "target": "3", "relation": "targets"}
{"source": "6", "target": "2", "relation": "targets"}
{"source": "2", "target": "7", "relation": "participates_in"}
{"source": "3", "target": "7", "relation": "participates_in"}
{"source": "4", "target": "8", "relation": "related_to"}
{"source": "9", "target": "11", "relation": "participates_in"}
{"source": "1", "target": "12", "relation": "treats"}
{"source": "4", "target": "4", "relation": "related_to"}
```

The last edge is a self-loop. Redstring rejects self-loops at validation, so it must be dropped and counted.

```
# tests/fixtures/tiny_skb/queries.jsonl
{"query_id": 1, "text": "which selective COX-2 inhibitor is used in arthritis?", "answer_ids": ["6"]}
{"query_id": 2, "text": "drugs that target cyclooxygenase genes", "answer_ids": ["1", "5", "6"]}
{"query_id": 3, "text": "genes participating in prostaglandin synthesis", "answer_ids": ["2", "3"]}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/skb/test_artifacts.py
from pathlib import Path

from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_skb"


def test_nodes_round_trip():
    nodes = list(read_nodes(FIXTURE / "nodes.jsonl"))
    assert len(nodes) == 12
    assert nodes[0].node_id == "1"
    assert nodes[0].node_type == "drug"
    assert "cyclooxygenase" in nodes[0].document


def test_edges_include_the_self_loop_unfiltered():
    """Reading is not filtering.

    The self-loop must survive to the loader, which drops it and records the
    count. A reader that silently dropped it would make a recall ceiling look
    like a retrieval failure later.
    """
    edges = list(read_edges(FIXTURE / "edges.jsonl"))
    assert len(edges) == 11
    assert any(e.source == e.target for e in edges)


def test_queries_carry_answers_separately_from_the_query():
    pairs = list(read_queries(FIXTURE / "queries.jsonl"))
    query, answers = pairs[0]
    assert query.query_id == 1
    assert not hasattr(query, "answer_ids")
    assert answers == ["6"]
```

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest tests/skb/test_artifacts.py -v -p no:randomly`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement**

```python
# src/stark_bench/skb/artifacts.py
"""The neutral format between the sidecar and the harness.

JSON Lines, one record per line. The sidecar writes these under a 3.11
interpreter with `stark-qa` installed; everything else in this project reads
them under 3.13 with a small dependency set.

`read_queries` returns `(Query, answers)` pairs rather than a query object
carrying its answers, because `Query` is what reaches an agent and must not
carry ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stark_bench.ports import Query

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkbNode:
    node_id: str
    node_type: str
    name: str
    document: str


@dataclass(frozen=True, slots=True)
class SkbEdge:
    source: str
    target: str
    relation: str


def read_nodes(path: Path) -> Iterator[SkbNode]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield SkbNode(**json.loads(line))


def read_edges(path: Path) -> Iterator[SkbEdge]:
    """Yields every edge, self-loops included.

    Filtering belongs to the loader, which counts what it drops.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield SkbEdge(**json.loads(line))


def read_queries(path: Path) -> Iterator[tuple[Query, list[str]]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            answers = [str(a) for a in record["answer_ids"]]
            yield Query(query_id=int(record["query_id"]), text=record["text"]), answers
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/skb/test_artifacts.py -v -p no:randomly`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Neutral SKB artifact format, plus a twelve-node fixture with a self-loop"
```

---

### Task 5: The loader

**Files:**
- Create: `src/stark_bench/skb/ingest.py`, `src/stark_bench/skb/chunkers.py`
- Test: `tests/skb/test_ingest.py`

**Interfaces:**
- Consumes: `entity_id_for`, `STARK_ID_KEY`, `SkbNode`, `SkbEdge`.
- Produces: `IngestReport(nodes: int, edges: int, self_loops_dropped: int, chunks: int)`, `async def ingest(nodes, edges, *, dataset, tenant_id, graph, chunks, chunker, embeddings) -> IngestReport`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/skb/test_ingest.py
import pytest
from redstring import FakeEmbeddingProvider, InMemoryChunkStore, InMemoryGraphStore, TenantId
from uuid import uuid4

from stark_bench.skb.artifacts import SkbEdge, SkbNode
from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for
from stark_bench.skb.ingest import ingest
from stark_bench.skb.chunkers import WholeDocumentChunker


@pytest.fixture
def stores():
    return InMemoryGraphStore(), InMemoryChunkStore(dimension=8)


@pytest.mark.asyncio
async def test_it_writes_entities_carrying_their_stark_id(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate"),
             SkbNode("2", "gene", "PTGS2", "cyclooxygenase-2")]

    report = await ingest(nodes, [], dataset="prime", tenant_id=tenant, graph=graph,
                          chunks=chunks, chunker=WholeDocumentChunker(),
                          embeddings=FakeEmbeddingProvider(dimension=8))

    assert report.nodes == 2
    stored = await graph.get_entity(entity_id_for("prime", "1"), tenant)
    assert stored is not None
    assert stored.external_ids[STARK_ID_KEY] == "1"


@pytest.mark.asyncio
async def test_a_self_loop_is_dropped_and_counted(stores):
    """Redstring rejects self-loops. A silent drop would make a recall
    ceiling look like a retrieval failure, so the count is reported."""
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    edges = [SkbEdge("1", "1", "related_to")]

    report = await ingest(nodes, edges, dataset="prime", tenant_id=tenant, graph=graph,
                          chunks=chunks, chunker=WholeDocumentChunker(),
                          embeddings=FakeEmbeddingProvider(dimension=8))

    assert report.self_loops_dropped == 1
    assert report.edges == 0


@pytest.mark.asyncio
async def test_edges_referencing_unknown_nodes_do_not_abort_the_run(stores):
    """A bad edge followed by a good one.

    Stated this way on purpose: with only one bad edge at the end of the
    loop, `break` and `continue` are the same function, and a `break` would
    silently discard every later edge in a real corpus.
    """
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "x"), SkbNode("2", "gene", "PTGS2", "y")]
    edges = [SkbEdge("1", "999", "targets"), SkbEdge("1", "2", "targets")]

    report = await ingest(nodes, edges, dataset="prime", tenant_id=tenant, graph=graph,
                          chunks=chunks, chunker=WholeDocumentChunker(),
                          embeddings=FakeEmbeddingProvider(dimension=8))

    assert report.edges == 1


@pytest.mark.asyncio
async def test_ingest_is_idempotent(stores):
    graph, chunks = stores
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "a salicylate")]
    kwargs = dict(dataset="prime", tenant_id=tenant, graph=graph, chunks=chunks,
                  chunker=WholeDocumentChunker(), embeddings=FakeEmbeddingProvider(dimension=8))

    first = await ingest(nodes, [], **kwargs)
    second = await ingest(nodes, [], **kwargs)

    assert first.nodes == second.nodes == 1
    assert len(await graph.find_entities(tenant)) == 1
```

- [ ] **Step 2: Run and watch fail**

Run: `uv run pytest tests/skb/test_ingest.py -v -p no:randomly`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `WholeDocumentChunker`**

```python
# src/stark_bench/skb/chunkers.py
"""One chunk per document.

This is the `vss-control` chunker. STaRK's precomputed ada-002 vectors are one
per node document, so the control path must present each document as a single
chunk for those vectors to apply. `chunker_type` is recorded on results, so
the configuration labels itself in the output.
"""

from __future__ import annotations

from redstring.extraction.chunking import Chunk, ChunkingResult


class WholeDocumentChunker:
    """Satisfies redstring's `Chunker` protocol without splitting anything."""

    @property
    def chunker_type(self) -> str:
        return "whole-document"

    def chunk(
        self,
        text: str,
        max_chunk_size: int | None = None,
        overlap_size: int | None = None,
    ) -> ChunkingResult:
        return ChunkingResult(
            chunks=[Chunk(text=text, chunk_index=0, start_char=0, end_char=len(text))],
            total_chunks=1,
            original_length=len(text),
            chunking_method="whole-document",
            overlap_size=0,
        )
```

- [ ] **Step 4: Write the loader**

```python
# src/stark_bench/skb/ingest.py
"""STaRK's SKB into redstring's stores.

The loader is a projection: it reads a knowledge base someone else built and
writes it through redstring's ports. It invents nothing and fetches nothing.
No extraction runs and no LLM is involved, so provenance records
`ExtractionMethod.MANUAL` -- the honest value.

Two ordering constraints come from the ports themselves:

- `upsert_relationships` raises `MissingEntityError` if an endpoint is
  missing, so entities are written before the relationships referencing them.
- Self-loops are rejected by validation. They are dropped here and *counted*,
  because a silent drop turns a recall ceiling into an apparent retrieval
  failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from redstring import (
    Entity,
    ExtractionMethod,
    Provenance,
    Relationship,
    RelationshipId,
    SourceId,
)
from redstring.domain.chunk import StoredChunk, chunk_id

from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for

if TYPE_CHECKING:
    from collections.abc import Iterable

    from redstring import EmbeddingProvider, TenantId

    from stark_bench.skb.artifacts import SkbEdge, SkbNode

BATCH = 500


@dataclass(frozen=True, slots=True)
class IngestReport:
    nodes: int
    edges: int
    self_loops_dropped: int
    chunks: int


def _entity(node: SkbNode, *, dataset: str, tenant_id: TenantId, observed_at: datetime) -> Entity:
    return Entity(
        id=entity_id_for(dataset, node.node_id),
        tenant_id=tenant_id,
        name=node.name,
        normalized_name=node.name.casefold(),
        entity_type=node.node_type,
        external_ids={STARK_ID_KEY: node.node_id},
        provenance=Provenance(
            observed_at=observed_at,
            extraction_method=ExtractionMethod.MANUAL,
            confidence=1.0,
        ),
    )


async def ingest(
    nodes: Iterable[SkbNode],
    edges: Iterable[SkbEdge],
    *,
    dataset: str,
    tenant_id: TenantId,
    graph,
    chunks,
    chunker,
    embeddings: EmbeddingProvider,
) -> IngestReport:
    observed_at = datetime.now(UTC)
    known: set[str] = set()
    node_count = chunk_count = 0

    batch: list[Entity] = []
    chunk_batch: list[StoredChunk] = []

    for node in nodes:
        batch.append(_entity(node, dataset=dataset, tenant_id=tenant_id, observed_at=observed_at))
        known.add(node.node_id)
        node_count += 1

        source_id = SourceId(entity_id_for(dataset, node.node_id))
        result = chunker.chunk(node.document)
        texts = [c.text for c in result.chunks]
        vectors = embeddings.embed(texts)
        vectors = await vectors if hasattr(vectors, "__await__") else vectors
        for piece, vector in zip(result.chunks, vectors, strict=True):
            chunk_batch.append(
                StoredChunk(
                    id=chunk_id(source_id, piece.text),
                    tenant_id=tenant_id,
                    source_id=source_id,
                    text=piece.text,
                    chunk_index=piece.chunk_index,
                    start_char=piece.start_char,
                    end_char=piece.end_char,
                    entity_ids=[entity_id_for(dataset, node.node_id)],
                    metadata={STARK_ID_KEY: node.node_id},
                    embedding=list(vector),
                )
            )
            chunk_count += 1

        if len(batch) >= BATCH:
            await graph.upsert_entities(batch)
            await chunks.upsert_many(chunk_batch)
            batch, chunk_batch = [], []

    if batch:
        await graph.upsert_entities(batch)
    if chunk_batch:
        await chunks.upsert_many(chunk_batch)

    dropped = 0
    edge_count = 0
    rels: list[Relationship] = []
    for edge in edges:
        if edge.source == edge.target:
            dropped += 1
            continue
        if edge.source not in known or edge.target not in known:
            continue
        rels.append(
            Relationship(
                id=RelationshipId(uuid4()),
                tenant_id=tenant_id,
                source_entity_id=entity_id_for(dataset, edge.source),
                target_entity_id=entity_id_for(dataset, edge.target),
                relationship_type=edge.relation,
                confidence=1.0,
            )
        )
        edge_count += 1
        if len(rels) >= BATCH:
            await graph.upsert_relationships(rels)
            rels = []
    if rels:
        await graph.upsert_relationships(rels)

    return IngestReport(
        nodes=node_count, edges=edge_count, self_loops_dropped=dropped, chunks=chunk_count
    )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/skb/test_ingest.py -v -p no:randomly`
Expected: PASS (4 tests). If `FakeEmbeddingProvider`'s constructor differs, read its source in the redstring checkout — do not guess.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "The loader: SKB into redstring's stores, dropping self-loops loudly"
```

---

### Task 6: Chunk-to-node aggregation

**Files:**
- Create: `src/stark_bench/harness/aggregate.py`
- Test: `tests/harness/test_aggregate.py`

**Interfaces:**
- Consumes: `Ranked`.
- Produces: `AGGREGATIONS: dict[str, Callable]`, `aggregate(scored: Sequence[tuple[str, float]], *, strategy: str = "max") -> list[Ranked]`.

- [ ] **Step 1: Write the failing test**

Note the differing scores. A node whose chunks all scored the same could not distinguish `max` from `mean` from `first`.

```python
# tests/harness/test_aggregate.py
import pytest

from stark_bench.harness.aggregate import aggregate


def test_max_takes_the_best_chunk_of_a_node():
    """Scores differ on purpose: equal scores make max, mean and first agree."""
    result = aggregate([("A", 0.2), ("A", 0.9), ("B", 0.5)], strategy="max")
    assert [(r.node_id, r.score) for r in result] == [("A", 0.9), ("B", 0.5)]


def test_mean_is_a_different_answer_on_the_same_input():
    result = aggregate([("A", 0.2), ("A", 0.9), ("B", 0.5)], strategy="mean")
    by_id = {r.node_id: r.score for r in result}
    assert by_id["A"] == pytest.approx(0.55)
    assert by_id["B"] == pytest.approx(0.5)


def test_results_are_ordered_best_first():
    result = aggregate([("A", 0.1), ("B", 0.7), ("C", 0.4)], strategy="max")
    assert [r.node_id for r in result] == ["B", "C", "A"]


def test_ties_break_deterministically_by_node_id():
    """Two runs must rank identically, or a metric moves for no reason."""
    first = aggregate([("B", 0.5), ("A", 0.5)], strategy="max")
    second = aggregate([("A", 0.5), ("B", 0.5)], strategy="max")
    assert [r.node_id for r in first] == [r.node_id for r in second] == ["A", "B"]


def test_an_unknown_strategy_raises():
    with pytest.raises(KeyError):
        aggregate([("A", 1.0)], strategy="whatever-scores-best")
```

- [ ] **Step 2: Run and watch fail**

Run: `uv run pytest tests/harness/test_aggregate.py -v -p no:randomly`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/stark_bench/harness/aggregate.py
"""Scored chunks up to scored nodes.

Retrieval returns chunks; STaRK scores nodes. The strategy is a *named,
recorded* config value rather than an implicit default, because an
aggregation function that is an unrecorded knob turns a benchmark into a
search for its best accident.

On `vss-control` there is one chunk per node, so every strategy degenerates
to identity -- the control exercises this code without aggregation being able
to change its answer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from stark_bench.ports import Ranked

if TYPE_CHECKING:
    from collections.abc import Sequence

AGGREGATIONS = {
    "max": max,
    "mean": lambda scores: sum(scores) / len(scores),
    "sum": sum,
}


def aggregate(
    scored: Sequence[tuple[str, float]], *, strategy: str = "max"
) -> list[Ranked]:
    """Fold per-chunk scores into per-node scores, best first.

    Ties break on `node_id` so two runs over the same data rank identically;
    without it a metric can move between runs for no reason at all.
    """
    reducer = AGGREGATIONS[strategy]
    grouped: dict[str, list[float]] = defaultdict(list)
    for node_id, score in scored:
        grouped[node_id].append(score)

    ranked = [Ranked(node_id=n, score=float(reducer(s))) for n, s in grouped.items()]
    ranked.sort(key=lambda r: (-r.score, r.node_id))
    return ranked
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/harness/test_aggregate.py -v -p no:randomly`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Chunk-to-node aggregation, by a named strategy recorded in results"
```

---

### Task 7: Scoring through the sidecar, and proving it can fail

**Files:**
- Create: `src/stark_bench/sidecar/__init__.py`, `src/stark_bench/sidecar/score.py`, `src/stark_bench/harness/scoring.py`
- Test: `tests/harness/test_scoring.py`

**Interfaces:**
- Consumes: `Ranked`.
- Produces: `score_predictions(predictions, answers, *, metrics) -> dict[str, float]` in `harness/scoring.py`, and the sidecar entry point `stark_bench.sidecar.score`.

- [ ] **Step 1: Write the sidecar script**

It runs under 3.11 with `stark-qa` installed, and does nothing but call the official evaluator.

```python
# src/stark_bench/sidecar/score.py
"""Official STaRK scoring, run under 3.11 with `stark-qa` installed.

We compute no metric ourselves. An expected value produced by the code under
test measures determinism rather than correctness, and a reimplemented MRR is
exactly that.

Invoked as a subprocess:
    uv run --python 3.11 --with stark-qa python -m stark_bench.sidecar.score \
        --predictions preds.json --answers answers.json --out metrics.json
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", default="mrr,hit@1,hit@5,recall@20")
    args = parser.parse_args()

    import torch
    from stark_qa.evaluator import Evaluator

    with open(args.predictions) as handle:
        predictions = json.load(handle)
    with open(args.answers) as handle:
        answers = json.load(handle)

    metrics = args.metrics.split(",")
    evaluator = Evaluator(candidate_ids=None)

    totals: dict[str, list[float]] = {m: [] for m in metrics}
    for query_id, pred in predictions.items():
        pred_dict = {int(node_id): float(score) for node_id, score in pred.items()}
        answer_ids = torch.LongTensor([int(a) for a in answers[query_id]])
        result = evaluator.evaluate(pred_dict, answer_ids, metrics=metrics)
        for name, value in result.items():
            totals[name].append(float(value))

    averaged = {name: (sum(v) / len(v) if v else 0.0) for name, v in totals.items()}
    with open(args.out, "w") as handle:
        json.dump(averaged, handle, indent=2)


if __name__ == "__main__":
    main()
```

**Note for the implementer:** `Evaluator`'s constructor signature must be read from the installed `stark_qa` package before this is trusted — `candidate_ids=None` is the expected form but verify it, and verify whether `evaluate` takes a dict or tensors. Run:

```bash
uv run --python 3.11 --with stark-qa python -c "import inspect, stark_qa.evaluator as e; print(inspect.signature(e.Evaluator.__init__)); print(inspect.signature(e.Evaluator.evaluate))"
```

Fix the call to match what it prints. Do not guess.

- [ ] **Step 2: Write the failing scoring test**

The deliberate-break check: a scoring path that cannot be made to fail on purpose is not yet measuring anything.

```python
# tests/harness/test_scoring.py
import pytest

from stark_bench.harness.scoring import score_predictions
from stark_bench.ports import Ranked


@pytest.mark.integration
def test_a_perfect_agent_scores_one():
    predictions = {1: [Ranked("6", 1.0), Ranked("2", 0.1)]}
    answers = {1: ["6"]}
    metrics = score_predictions(predictions, answers, metrics=["hit@1", "mrr"])
    assert metrics["hit@1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


@pytest.mark.integration
def test_a_useless_agent_scores_zero():
    """Without this, a scoring path that returns 1.0 unconditionally passes."""
    predictions = {1: [Ranked("999", 1.0), Ranked("998", 0.5)]}
    answers = {1: ["6"]}
    metrics = score_predictions(predictions, answers, metrics=["hit@1", "mrr"])
    assert metrics["hit@1"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.0)


@pytest.mark.integration
def test_rank_order_matters():
    """A right answer in second place must not score like first place."""
    first = score_predictions({1: [Ranked("6", 1.0), Ranked("9", 0.5)]}, {1: ["6"]}, metrics=["mrr"])
    second = score_predictions({1: [Ranked("9", 1.0), Ranked("6", 0.5)]}, {1: ["6"]}, metrics=["mrr"])
    assert first["mrr"] > second["mrr"]
```

- [ ] **Step 3: Run and watch fail**

Run: `uv run pytest tests/harness/test_scoring.py -v -p no:randomly`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement the harness side**

```python
# src/stark_bench/harness/scoring.py
"""Hand predictions to STaRK's evaluator, in its own interpreter.

`stark-qa` pulls colbert-ai, gritlm, llm2vec, PyTDC, ogb, torch_geometric and
more -- all serving baselines we do not run, several of which will not resolve
on 3.13. So it lives in a 3.11 environment reached by subprocess, and the
harness stays small.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from stark_bench.ports import Ranked

DEFAULT_METRICS = ("mrr", "hit@1", "hit@5", "recall@20")


def score_predictions(
    predictions: Mapping[int, Sequence[Ranked]],
    answers: Mapping[int, Sequence[str]],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Run the official evaluator over `predictions`. Raises on any failure."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        preds_path, answers_path, out_path = (
            root / "preds.json",
            root / "answers.json",
            root / "metrics.json",
        )
        preds_path.write_text(
            json.dumps(
                {
                    str(qid): {r.node_id: r.score for r in ranked}
                    for qid, ranked in predictions.items()
                }
            )
        )
        answers_path.write_text(
            json.dumps({str(qid): list(a) for qid, a in answers.items()})
        )

        completed = subprocess.run(  # noqa: S603
            [
                "uv", "run", "--python", "3.11", "--with", "stark-qa",
                sys.executable.rsplit("/", 1)[-1] if False else "python",
                "-m", "stark_bench.sidecar.score",
                "--predictions", str(preds_path),
                "--answers", str(answers_path),
                "--out", str(out_path),
                "--metrics", ",".join(metrics),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stark-qa scoring failed:\n{completed.stdout}\n{completed.stderr}"
            )
        return json.loads(out_path.read_text())
```

**Note for the implementer:** the subprocess must be able to import `stark_bench.sidecar.score`. Add `--with-editable .` to the `uv run` invocation, or set `PYTHONPATH` to `src`. Verify by running the command by hand before trusting the test.

- [ ] **Step 5: Confirm the pytest configuration is already present**

Task 1 added it. Verify `pyproject.toml` contains exactly this and change nothing if so:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: needs the stark-qa sidecar or a database",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 6: Run the tests explicitly**

Run: `uv run pytest tests/harness/test_scoring.py -v -p no:randomly -m integration`
Expected: PASS (3 tests). The first run downloads stark-qa's dependency tree into a uv cache and will be slow.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Official STaRK scoring via a 3.11 sidecar, proven able to score zero"
```

---

### Task 8: The instrumented toolset

**Files:**
- Create: `src/stark_bench/tools/__init__.py`, `src/stark_bench/tools/redstring_tools.py`
- Test: `tests/tools/test_redstring_tools.py`

**Interfaces:**
- Consumes: `Toolset`, `Ranked`, `ToolCall`, `node_id_of`, `aggregate`.
- Produces: `RedstringToolset(chunks, graph, embeddings, tenant_id, dataset, *, llm=None, aggregation="max")` satisfying `Toolset`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_redstring_tools.py
import pytest
from redstring import FakeEmbeddingProvider, InMemoryChunkStore, InMemoryGraphStore, TenantId
from uuid import uuid4

from stark_bench.ports import Toolset
from stark_bench.skb.artifacts import SkbEdge, SkbNode
from stark_bench.skb.chunkers import WholeDocumentChunker
from stark_bench.skb.ingest import ingest
from stark_bench.tools.redstring_tools import RedstringToolset


@pytest.fixture
async def toolset():
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    nodes = [SkbNode("1", "drug", "aspirin", "aspirin inhibits cyclooxygenase"),
             SkbNode("2", "gene", "PTGS2", "PTGS2 encodes cyclooxygenase-2")]
    await ingest(nodes, [SkbEdge("1", "2", "targets")], dataset="prime", tenant_id=tenant,
                 graph=graph, chunks=chunks, chunker=WholeDocumentChunker(),
                 embeddings=FakeEmbeddingProvider(dimension=8))
    return RedstringToolset(chunks=chunks, graph=graph,
                            embeddings=FakeEmbeddingProvider(dimension=8),
                            tenant_id=tenant, dataset="prime")


@pytest.mark.asyncio
async def test_it_satisfies_the_toolset_protocol(toolset):
    assert isinstance(toolset, Toolset)


@pytest.mark.asyncio
async def test_search_returns_stark_node_ids_not_entity_ids(toolset):
    tools = toolset
    results = await tools.search_chunks("cyclooxygenase", k=5)
    assert results
    assert all(r.node_id in {"1", "2"} for r in results)


@pytest.mark.asyncio
async def test_every_call_is_recorded(toolset):
    tools = toolset
    await tools.search_chunks("cyclooxygenase", k=5)
    await tools.neighbors("1")
    assert [c.tool for c in tools.calls] == ["search_chunks", "neighbors"]
    assert all(c.duration_s >= 0 for c in tools.calls)


@pytest.mark.asyncio
async def test_neighbors_returns_stark_ids(toolset):
    tools = toolset
    assert await tools.neighbors("1") == ["2"]


@pytest.mark.asyncio
async def test_the_toolset_exposes_no_writer(toolset):
    """Reader-only is the point: an agent that cannot write cannot poison
    the KB mid-run."""
    tools = toolset
    for forbidden in ("upsert_entities", "upsert_many", "delete_by_tenant"):
        assert not hasattr(tools, forbidden)
```

- [ ] **Step 2: Run and watch fail**

Run: `uv run pytest tests/tools/test_redstring_tools.py -v -p no:randomly`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/stark_bench/tools/redstring_tools.py
"""The instrumented, reader-only surface an agent sees.

Two things are deliberate. First, everything an agent touches is a *reader*:
no writer method is reachable, which is a type-level guarantee rather than a
matter of discipline. Second, every call is timed and counted, because cost is
a reported metric -- a deep agent buying four points of Hit@1 for forty times
the tokens is a different finding depending on which number you needed, and
Hit@1 alone cannot express it.

Traversal comes from `RelationshipStore`, not from `Retriever`: redstring's
`Retriever` holds `EntityReader` only and has no traversal at all.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from redstring import ChunkRetriever, RetrievalMode

from stark_bench.harness.aggregate import aggregate
from stark_bench.ports import Ranked, ToolCall
from stark_bench.skb.ids import STARK_ID_KEY, entity_id_for, node_id_of

if TYPE_CHECKING:
    from redstring import EmbeddingProvider, TenantId

MODES = {
    "semantic": RetrievalMode.SEMANTIC,
    "lexical": RetrievalMode.LEXICAL,
    "hybrid": RetrievalMode.HYBRID,
}


class RedstringToolset:
    """Satisfies `Toolset` over redstring's read ports."""

    def __init__(
        self,
        *,
        chunks,
        graph,
        embeddings: EmbeddingProvider,
        tenant_id: TenantId,
        dataset: str,
        llm=None,
        aggregation: str = "max",
    ) -> None:
        self._chunks = chunks
        self._graph = graph
        self._tenant = tenant_id
        self._dataset = dataset
        self._llm = llm
        self._aggregation = aggregation
        self._retriever = ChunkRetriever(embeddings=embeddings, chunks=chunks)
        self.calls: list[ToolCall] = []

    def _record(self, tool: str, started: float, count: int, tokens: int = 0) -> None:
        self.calls.append(
            ToolCall(
                tool=tool,
                duration_s=perf_counter() - started,
                result_count=count,
                tokens=tokens,
            )
        )

    async def search_chunks(
        self, text: str, *, k: int = 10, mode: str = "hybrid"
    ) -> list[Ranked]:
        """Retrieve chunks and fold them up to STaRK nodes.

        Overfetches chunks relative to `k`, because several chunks may belong
        to one node and folding shrinks the list.
        """
        started = perf_counter()
        result = await self._retriever.retrieve_chunks(
            text, self._tenant, k=k * 4, mode=MODES[mode]
        )
        scored = [
            (str(match.chunk.metadata[STARK_ID_KEY]), match.score)
            for match in result.matches
            if STARK_ID_KEY in match.chunk.metadata
        ]
        ranked = aggregate(scored, strategy=self._aggregation)[:k]
        self._record("search_chunks", started, len(ranked))
        return ranked

    async def get_node(self, node_id: str) -> dict[str, object] | None:
        started = perf_counter()
        entity = await self._graph.get_entity(
            entity_id_for(self._dataset, node_id), self._tenant
        )
        self._record("get_node", started, 0 if entity is None else 1)
        if entity is None:
            return None
        return {
            "node_id": node_id,
            "name": entity.name,
            "node_type": entity.entity_type,
        }

    async def neighbors(self, node_id: str, *, depth: int = 1) -> list[str]:
        started = perf_counter()
        found = await self._graph.neighbors(
            entity_id_for(self._dataset, node_id), self._tenant, depth=depth
        )
        ids = [node_id_of(entity) for entity in found]
        self._record("neighbors", started, len(ids))
        return ids

    async def get_relationships(self, node_id: str) -> list[tuple[str, str, str]]:
        """Edges as `(source_node_id, relation, target_node_id)`.

        `neighbors` returns entities with no edge type and no hop distance, so
        an agent that needs to know *how* two nodes connect calls this instead.
        """
        started = perf_counter()
        entity_id = entity_id_for(self._dataset, node_id)
        rels = await self._graph.get_relationships(entity_id, self._tenant)
        ids = {r.source_entity_id for r in rels} | {r.target_entity_id for r in rels}
        entities = await self._graph.get_entities(list(ids), self._tenant)
        lookup = {e.id: node_id_of(e) for e in entities}
        edges = [
            (lookup[r.source_entity_id], r.relationship_type, lookup[r.target_entity_id])
            for r in rels
            if r.source_entity_id in lookup and r.target_entity_id in lookup
        ]
        self._record("get_relationships", started, len(edges))
        return edges

    async def complete(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("this toolset was built without an LLM provider")
        started = perf_counter()
        response = await self._llm.complete(prompt)
        text = getattr(response, "text", response)
        self._record("complete", started, 1, tokens=getattr(response, "total_tokens", 0))
        return text
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/tools/test_redstring_tools.py -v -p no:randomly`
Expected: PASS (5 tests). If `LlmProvider.complete` has a different name or response shape, read the port in the redstring checkout and correct `complete`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "The instrumented reader-only toolset agents see"
```

---

### Task 9: The `dense` and `hybrid` agents, the runner, and the first end-to-end number

**Files:**
- Create: `src/stark_bench/agents/dense.py`, `src/stark_bench/agents/hybrid.py`, `src/stark_bench/harness/runner.py`
- Test: `tests/agents/test_baselines.py`, `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `Agent`, `Toolset`, `Query`, `Ranked`, `score_predictions`.
- Produces: `DenseAgent`, `HybridAgent`, `async def run(agent, queries, tools, *, k=20) -> dict[int, list[Ranked]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_baselines.py
import pytest

from stark_bench.agents.dense import DenseAgent
from stark_bench.agents.hybrid import HybridAgent
from stark_bench.ports import Agent, Query, Ranked, ToolCall


class RecordingTools:
    def __init__(self):
        self.calls: list[ToolCall] = []
        self.modes: list[str] = []

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.modes.append(mode)
        return [Ranked("1", 0.9), Ranked("2", 0.4)]

    async def get_node(self, node_id): return None
    async def neighbors(self, node_id, *, depth=1): return []
    async def get_relationships(self, node_id): return []
    async def complete(self, prompt): raise AssertionError("baselines use no LLM")


@pytest.mark.asyncio
async def test_dense_uses_the_semantic_channel_only():
    tools = RecordingTools()
    result = await DenseAgent(k=20).retrieve(Query(1, "aspirin"), tools)
    assert tools.modes == ["semantic"]
    assert [r.node_id for r in result] == ["1", "2"]


@pytest.mark.asyncio
async def test_hybrid_uses_the_fused_channel():
    tools = RecordingTools()
    await HybridAgent(k=20).retrieve(Query(1, "aspirin"), tools)
    assert tools.modes == ["hybrid"]


@pytest.mark.asyncio
async def test_baselines_make_no_llm_call():
    """A baseline that quietly called an LLM would not be a baseline."""
    tools = RecordingTools()
    await DenseAgent(k=20).retrieve(Query(1, "x"), tools)
    await HybridAgent(k=20).retrieve(Query(1, "x"), tools)


def test_both_satisfy_the_agent_protocol():
    assert isinstance(DenseAgent(k=20), Agent)
    assert isinstance(HybridAgent(k=20), Agent)
```

- [ ] **Step 2: Run and watch fail**

Run: `uv run pytest tests/agents/test_baselines.py -v -p no:randomly`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement both agents**

```python
# src/stark_bench/agents/dense.py
"""One vector search, returned as-is. The control.

No LLM, so it runs the full query set cheaply and often -- which is what lets
us tell whether a moved agent number reflects the agent or the knowledge base
underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stark_bench.ports import Query, Ranked, Toolset


@dataclass(frozen=True, slots=True)
class DenseAgent:
    k: int = 20
    name: str = "dense"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        return await tools.search_chunks(query.text, k=self.k, mode="semantic")
```

```python
# src/stark_bench/agents/hybrid.py
"""Vector and BM25, fused by rank inside redstring.

Answers "does redstring's fusion beat dense retrieval on STaRK" with no agent
variance in the way. The fusion constant is redstring's and is not tuned here:
its docstring says exposing it would invite tuning against a benchmark the
library does not have, and this is that benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stark_bench.ports import Query, Ranked, Toolset


@dataclass(frozen=True, slots=True)
class HybridAgent:
    k: int = 20
    name: str = "hybrid"

    async def retrieve(self, query: Query, tools: Toolset) -> list[Ranked]:
        return await tools.search_chunks(query.text, k=self.k, mode="hybrid")
```

- [ ] **Step 4: Write the runner test**

```python
# tests/harness/test_runner.py
import pytest

from stark_bench.harness.runner import run
from stark_bench.ports import Query, Ranked, ToolCall


class Tools:
    def __init__(self): self.calls: list[ToolCall] = []
    async def search_chunks(self, text, *, k=10, mode="hybrid"): return [Ranked("1", 1.0)]
    async def get_node(self, node_id): return None
    async def neighbors(self, node_id, *, depth=1): return []
    async def get_relationships(self, node_id): return []
    async def complete(self, prompt): return ""


class Boom:
    name = "boom"
    async def retrieve(self, query, tools):
        if query.query_id == 2:
            raise ValueError("this query breaks the agent")
        return [Ranked("1", 1.0)]


@pytest.mark.asyncio
async def test_a_failing_query_does_not_abort_the_run():
    """A bad query followed by a good one.

    With the failure last, `break` and `continue` are the same function, and a
    `break` would silently discard every later query in an 11k-query run.
    """
    queries = [Query(1, "a"), Query(2, "b"), Query(3, "c")]
    predictions = await run(Boom(), queries, Tools())
    assert set(predictions) == {1, 2, 3}
    assert predictions[2] == []
```

- [ ] **Step 5: Implement the runner**

```python
# src/stark_bench/harness/runner.py
"""Run one agent over a query set.

A query that raises is recorded as an empty prediction rather than aborting
the run: over eleven thousand queries, one agent failure must not discard
every result after it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stark_bench.ports import Agent, Query, Ranked, Toolset

logger = logging.getLogger(__name__)


async def run(
    agent: Agent, queries: Sequence[Query], tools: Toolset, *, k: int = 20
) -> dict[int, list[Ranked]]:
    predictions: dict[int, list[Ranked]] = {}
    for query in queries:
        try:
            predictions[query.query_id] = list(await agent.retrieve(query, tools))
        except Exception:
            logger.exception("agent failed on query %s", query.query_id)
            predictions[query.query_id] = []
    return predictions
```

- [ ] **Step 6: Run everything**

Run: `uv run pytest tests/ -v -p no:randomly`
Expected: PASS

- [ ] **Step 7: Write the end-to-end fixture test — the first number**

```python
# tests/test_end_to_end_fixture.py
"""The whole pipeline on twelve nodes: ingest, retrieve, aggregate, score."""
from pathlib import Path
from uuid import uuid4

import pytest
from redstring import FakeEmbeddingProvider, InMemoryChunkStore, InMemoryGraphStore, TenantId

from stark_bench.agents.hybrid import HybridAgent
from stark_bench.harness.runner import run
from stark_bench.harness.scoring import score_predictions
from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries
from stark_bench.skb.chunkers import WholeDocumentChunker
from stark_bench.skb.ingest import ingest
from stark_bench.tools.redstring_tools import RedstringToolset

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_skb"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_whole_pipeline_produces_a_number():
    graph, chunks = InMemoryGraphStore(), InMemoryChunkStore(dimension=8)
    tenant = TenantId(uuid4())
    embeddings = FakeEmbeddingProvider(dimension=8)

    report = await ingest(
        read_nodes(FIXTURE / "nodes.jsonl"), read_edges(FIXTURE / "edges.jsonl"),
        dataset="fixture", tenant_id=tenant, graph=graph, chunks=chunks,
        chunker=WholeDocumentChunker(), embeddings=embeddings,
    )
    assert report.nodes == 12
    assert report.self_loops_dropped == 1

    pairs = list(read_queries(FIXTURE / "queries.jsonl"))
    queries = [q for q, _ in pairs]
    answers = {q.query_id: a for q, a in pairs}

    tools = RedstringToolset(chunks=chunks, graph=graph, embeddings=embeddings,
                            tenant_id=tenant, dataset="fixture")
    predictions = await run(HybridAgent(k=20), queries, tools)
    metrics = score_predictions(predictions, answers)

    assert set(metrics) >= {"mrr", "hit@1", "hit@5", "recall@20"}
    assert 0.0 <= metrics["mrr"] <= 1.0
```

Note the assertion: a *range*, not a threshold. `FakeEmbeddingProvider` produces meaningless vectors, so this test proves the pipeline runs end to end — it is not an accuracy claim, and must never be read as one.

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/test_end_to_end_fixture.py -v -p no:randomly -m integration`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "Baseline agents, the runner, and the pipeline proven end to end on twelve nodes"
```

---

### Task 10: The export sidecar

**Files:**
- Create: `src/stark_bench/sidecar/export.py`
- Test: `tests/sidecar/test_export_contract.py`

**Interfaces:**
- Consumes: nothing at runtime (isolated interpreter).
- Produces: `nodes.jsonl`, `edges.jsonl`, `queries.<split>.jsonl`, `embeddings.ada002.npz` under `data/<dataset>/`.

- [ ] **Step 1: Inspect the real API before writing anything**

```bash
uv run --python 3.11 --with stark-qa python - <<'EOF'
from stark_qa import load_qa, load_skb
import inspect
print(inspect.signature(load_skb))
print(inspect.signature(load_qa))
skb = load_skb("prime", download_processed=True)
print(type(skb), [m for m in dir(skb) if not m.startswith("_")][:40])
print("num nodes", skb.num_nodes())
print("node types", skb.node_type_lst() if hasattr(skb, "node_type_lst") else "?")
print("doc sample", skb.get_doc_info(0, add_rel=False)[:300])
qa = load_qa("prime")
print("qa len", len(qa), qa[0])
print("splits", qa.get_idx_split().keys())
EOF
```

Write the exporter against what this prints. Every attribute name below is a
best guess from STaRK's documentation and **must be corrected to match**.

- [ ] **Step 2: Write the exporter**

```python
# src/stark_bench/sidecar/export.py
"""STaRK's SKB to neutral artifacts, under 3.11 with `stark-qa` installed.

Run:
    uv run --python 3.11 --with stark-qa --with numpy \
        python -m stark_bench.sidecar.export --dataset prime --out data/prime
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="test-0.1,test")
    args = parser.parse_args()

    from stark_qa import load_qa, load_skb

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    skb = load_skb(args.dataset, download_processed=True)

    with (out / "nodes.jsonl").open("w", encoding="utf-8") as handle:
        for node_id in range(skb.num_nodes()):
            handle.write(
                json.dumps(
                    {
                        "node_id": str(node_id),
                        "node_type": str(skb.get_node_type_by_id(node_id)),
                        "name": str(skb[node_id].name if hasattr(skb[node_id], "name") else node_id),
                        "document": skb.get_doc_info(node_id, add_rel=False),
                    }
                )
                + "\n"
            )

    edge_index = skb.edge_index
    edge_types = skb.edge_types
    with (out / "edges.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(edge_index.shape[1]):
            handle.write(
                json.dumps(
                    {
                        "source": str(int(edge_index[0, i])),
                        "target": str(int(edge_index[1, i])),
                        "relation": str(skb.edge_type_dict[int(edge_types[i])]),
                    }
                )
                + "\n"
            )

    qa = load_qa(args.dataset)
    splits = qa.get_idx_split()
    for split in args.splits.split(","):
        indices = splits[split]
        with (out / f"queries.{split}.jsonl").open("w", encoding="utf-8") as handle:
            for idx in indices:
                query, query_id, answer_ids, _ = qa[int(idx)]
                handle.write(
                    json.dumps(
                        {
                            "query_id": int(query_id),
                            "text": str(query),
                            "answer_ids": [str(int(a)) for a in answer_ids],
                        }
                    )
                    + "\n"
                )

    print(f"exported {args.dataset} to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the contract test**

```python
# tests/sidecar/test_export_contract.py
"""The exporter's output must be readable by the harness's own reader.

This is the seam between two interpreters, and it is the one place where a
mismatch produces a confusing failure hours into an ingest.
"""
import json

import pytest

from stark_bench.skb.artifacts import read_edges, read_nodes, read_queries


@pytest.fixture
def exported(tmp_path):
    (tmp_path / "nodes.jsonl").write_text(
        json.dumps({"node_id": "0", "node_type": "drug", "name": "x", "document": "d"}) + "\n"
    )
    (tmp_path / "edges.jsonl").write_text(
        json.dumps({"source": "0", "target": "1", "relation": "targets"}) + "\n"
    )
    (tmp_path / "queries.test.jsonl").write_text(
        json.dumps({"query_id": 5, "text": "q", "answer_ids": ["0"]}) + "\n"
    )
    return tmp_path


def test_the_readers_accept_the_exporter_schema(exported):
    assert list(read_nodes(exported / "nodes.jsonl"))[0].node_id == "0"
    assert list(read_edges(exported / "edges.jsonl"))[0].relation == "targets"
    query, answers = list(read_queries(exported / "queries.test.jsonl"))[0]
    assert query.query_id == 5
    assert answers == ["0"]
```

- [ ] **Step 4: Run the contract test, then the real export**

```bash
uv run pytest tests/sidecar/test_export_contract.py -v -p no:randomly
uv run --python 3.11 --with stark-qa --with numpy python -m stark_bench.sidecar.export --dataset prime --out data/prime
wc -l data/prime/*.jsonl
```

Expected: roughly 129k node lines. If the numbers are wildly off, stop and read the SKB API rather than proceeding — an ingest built on a misread schema wastes hours.

- [ ] **Step 5: Add `data/` to `.gitignore` and commit**

```bash
echo "data/" >> .gitignore
git add -A && git commit -m "Export STaRK's SKB to neutral artifacts from a 3.11 sidecar"
```

---

### Task 11: Real backing, real ingest, and the first STaRK number

**Files:**
- Create: `docker-compose.yml`, `src/stark_bench/harness/providers.py`, `src/stark_bench/harness/config.py`, `src/stark_bench/harness/report.py`, `config/vss-control.yaml`, `config/redstring-native.yaml`, `src/stark_bench/harness/cli.py`
- Test: `tests/harness/test_config.py`, `tests/harness/test_report.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `RunConfig`, `load_config(path) -> RunConfig`, `write_report(...) -> None`, `PrecomputedEmbeddingProvider` (in `harness/providers.py`; Task 12 supplies its tests), and a CLI entry point.

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: stark
      POSTGRES_PASSWORD: stark
      POSTGRES_DB: stark
    ports: ["55432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U stark"]
      interval: 5s
      retries: 20
  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/starkbench
    ports: ["57474:7474", "57687:7687"]
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 5s
      retries: 20
```

- [ ] **Step 2: Write the config test**

```python
# tests/harness/test_config.py
from stark_bench.harness.config import load_config


def test_a_config_round_trips_verbatim(tmp_path):
    """The resolved config is embedded verbatim in every results file.

    That is what makes a number re-runnable, so the raw text is retained and
    not reconstructed from parsed fields.
    """
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: vss-control\n"
        "dataset: prime\n"
        "split: test-0.1\n"
        "chunker: whole-document\n"
        "embeddings: precomputed-ada002\n"
        "dimension: 1536\n"
        "aggregation: max\n"
        "agent: dense\n"
        "k: 20\n"
    )
    config = load_config(path)
    assert config.name == "vss-control"
    assert config.dimension == 1536
    assert "whole-document" in config.raw
```

- [ ] **Step 3: Implement config and report**

```python
# src/stark_bench/harness/config.py
"""Every knob that changes a number, in one file per run.

The resolved contents are embedded verbatim in the results file. Re-running a
variant is an edit here, and a number whose config is not recorded is not
re-runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    dataset: str
    split: str
    chunker: str
    embeddings: str
    dimension: int
    aggregation: str
    agent: str
    k: int
    raw: str


def load_config(path: Path) -> RunConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return RunConfig(raw=raw, **data)
```

```python
# src/stark_bench/harness/report.py
"""One JSON file per run, carrying its own config.

Cost sits beside accuracy deliberately: a deep agent that buys four points of
Hit@1 for forty times the tokens is a different finding depending on which
number you needed, and an accuracy table alone cannot express it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from stark_bench.harness.config import RunConfig
    from stark_bench.ports import ToolCall


def summarise_cost(calls: Sequence[ToolCall], queries: int) -> dict[str, float]:
    if queries == 0:
        return {"calls_per_query": 0.0, "tokens_per_query": 0.0, "seconds_total": 0.0}
    return {
        "calls_per_query": len(calls) / queries,
        "tokens_per_query": sum(c.tokens for c in calls) / queries,
        "seconds_total": sum(c.duration_s for c in calls),
    }


def write_report(
    path: Path,
    *,
    config: RunConfig,
    metrics: Mapping[str, float],
    cost: Mapping[str, float],
    ingest: Mapping[str, int],
    queries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config_name": config.name,
                "config_verbatim": config.raw,
                "queries": queries,
                "metrics": dict(metrics),
                "cost": dict(cost),
                "ingest": dict(ingest),
            },
            indent=2,
        )
    )
```

- [ ] **Step 4: Write the report test**

```python
# tests/harness/test_report.py
import json

from stark_bench.harness.config import RunConfig
from stark_bench.harness.report import summarise_cost, write_report
from stark_bench.ports import ToolCall


def test_cost_is_per_query_not_total_calls():
    calls = [ToolCall("search_chunks", 0.1, 5, tokens=100) for _ in range(10)]
    cost = summarise_cost(calls, queries=5)
    assert cost["calls_per_query"] == 2.0
    assert cost["tokens_per_query"] == 200.0


def test_zero_queries_does_not_divide_by_zero():
    assert summarise_cost([], queries=0)["calls_per_query"] == 0.0


def test_the_report_embeds_the_config_verbatim(tmp_path):
    config = RunConfig("vss-control", "prime", "test-0.1", "whole-document",
                       "precomputed-ada002", 1536, "max", "dense", 20, raw="name: vss-control\n")
    out = tmp_path / "r.json"
    write_report(out, config=config, metrics={"mrr": 0.4}, cost={"calls_per_query": 1.0},
                 ingest={"nodes": 12}, queries=3)
    written = json.loads(out.read_text())
    assert written["config_verbatim"] == "name: vss-control\n"
    assert written["metrics"]["mrr"] == 0.4
```

- [ ] **Step 5: Write the two configs**

```yaml
# config/vss-control.yaml
name: vss-control
dataset: prime
split: test-0.1
# One chunk per node: STaRK's precomputed vectors are one per node document,
# so the control path must present each document whole for them to apply.
chunker: whole-document
embeddings: precomputed-ada002
dimension: 1536
aggregation: max
agent: dense
k: 20
```

```yaml
# config/redstring-native.yaml
name: redstring-native
dataset: prime
split: test-0.1
chunker: boundary-preference
embeddings: nomic-embed-text
dimension: 768
aggregation: max
agent: hybrid
k: 20
```

- [ ] **Step 6: Write the CLI**

Wire config to stores: `precomputed-ada002` builds a lookup-table provider from the exported `.npz`; `nomic-embed-text` builds redstring's langchain embedding adapter against the configured endpoint. A precomputed lookup miss must **raise**, never fall back to live embedding — a silent fallback turns the control into a second native run wearing the control's label.

Stores come from the pinned dotted paths (`redstring.chunks.adapters.postgres`, `redstring.graph.adapters.neo4j`) against the compose services, one tenant and one store per config.

- [ ] **Step 7: Bring up the services and run the control**

```bash
docker compose up -d
uv run python -m stark_bench.harness.cli --config config/vss-control.yaml --ingest
uv run python -m stark_bench.harness.cli --config config/vss-control.yaml --run
cat results/vss-control.json
```

**Report this number before continuing.** Compare `mrr`, `hit@1`, `hit@5` and `recall@20` against STaRK's published VSS row for prime. If they are far apart, stop: the fault is in our ingest or scoring, and every later number inherits it.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "Real backing, real ingest, and the vss-control number on stark-prime"
```

---

### Task 12: The `redstring-native` number

**Files:**
- Modify: `src/stark_bench/harness/cli.py`
- Test: `tests/harness/test_precomputed_provider.py`

**Interfaces:**
- Consumes: everything above.
- Produces: tests for `PrecomputedEmbeddingProvider` (created in Task 11), and a `redstring-native` results file.

- [ ] **Step 1: Write the failing provider test**

```python
# tests/harness/test_precomputed_provider.py
import numpy as np
import pytest

from stark_bench.harness.providers import PrecomputedEmbeddingProvider


@pytest.fixture
def provider(tmp_path):
    path = tmp_path / "emb.npz"
    np.savez(path, texts=np.array(["alpha", "beta"]), vectors=np.zeros((2, 4)))
    return PrecomputedEmbeddingProvider(path, model="text-embedding-ada-002")


def test_it_returns_one_vector_per_input_in_order(provider):
    result = provider.embed(["beta", "alpha"])
    assert len(result) == 2
    assert provider.dimension == 4


def test_a_miss_raises_and_never_falls_back(provider):
    """A silent fallback would turn the control into a second native run
    wearing the control's label, corrupting every comparison downstream."""
    with pytest.raises(KeyError):
        provider.embed(["a text nobody embedded"])
```

- [ ] **Step 2: Run and watch fail**, then correct `PrecomputedEmbeddingProvider` in `src/stark_bench/harness/providers.py` (Task 11 created it) so that it satisfies redstring's `EmbeddingProvider` protocol: `model`, `dimension`, and a batch `embed` that preserves input order and raises `KeyError` on a miss.

- [ ] **Step 3: Run the native config**

```bash
uv run python -m stark_bench.harness.cli --config config/redstring-native.yaml --ingest
uv run python -m stark_bench.harness.cli --config config/redstring-native.yaml --run
```

**Ask before this step.** Embedding ~129k nodes' worth of chunks runs against a shared inference endpoint; confirm the slot count and that the endpoint is free.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "The redstring-native number: chunked corpus, nomic embeddings, rank fusion"
```

---

### Task 13: The zero-shot agent

**Files:**
- Create: `src/stark_bench/agents/zero_shot.py`
- Test: `tests/agents/test_zero_shot.py`

**Interfaces:**
- Consumes: `Toolset.complete`.
- Produces: `ZeroShotAgent(k=20)`.

One LLM call turns the query into a retrieval query, then one retrieval round. Fixed cost per query, no loop.

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_zero_shot.py
import pytest

from stark_bench.agents.zero_shot import ZeroShotAgent
from stark_bench.ports import Query, Ranked, ToolCall


class Tools:
    def __init__(self, reply="cyclooxygenase inhibitor drug"):
        self.calls: list[ToolCall] = []
        self.reply = reply
        self.searched: list[str] = []
        self.prompts: list[str] = []

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.searched.append(text)
        return [Ranked("6", 0.9)]

    async def get_node(self, node_id): return None
    async def neighbors(self, node_id, *, depth=1): return []
    async def get_relationships(self, node_id): return []
    async def complete(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.mark.asyncio
async def test_it_searches_with_the_rewritten_query():
    tools = Tools()
    await ZeroShotAgent(k=20).retrieve(Query(1, "which COX-2 drug treats arthritis?"), tools)
    assert tools.searched == ["cyclooxygenase inhibitor drug"]


@pytest.mark.asyncio
async def test_it_makes_exactly_one_llm_call():
    """Fixed cost is the defining property of this architecture."""
    tools = Tools()
    await ZeroShotAgent(k=20).retrieve(Query(1, "x"), tools)
    assert len(tools.prompts) == 1


@pytest.mark.asyncio
async def test_an_empty_rewrite_falls_back_to_the_original_query():
    """A refusal or a blank completion must not become a blank search."""
    tools = Tools(reply="   ")
    await ZeroShotAgent(k=20).retrieve(Query(1, "original text"), tools)
    assert tools.searched == ["original text"]


@pytest.mark.asyncio
async def test_an_llm_failure_falls_back_rather_than_losing_the_query():
    class Failing(Tools):
        async def complete(self, prompt):
            raise RuntimeError("endpoint down")

    tools = Failing()
    result = await ZeroShotAgent(k=20).retrieve(Query(1, "original text"), tools)
    assert tools.searched == ["original text"]
    assert result
```

- [ ] **Step 2: Run, watch fail, implement, run again, commit**

```bash
uv run pytest tests/agents/test_zero_shot.py -v -p no:randomly
git add -A && git commit -m "The zero-shot agent: one rewrite, one retrieval round"
```

---

### Task 14: The deep agent and budgets

**Files:**
- Create: `src/stark_bench/agents/deep.py`, `src/stark_bench/harness/budget.py`
- Test: `tests/agents/test_deep.py`, `tests/harness/test_budget.py`

**Interfaces:**
- Consumes: the full `Toolset`.
- Produces: `Budget(max_tool_calls, max_llm_calls, max_seconds)`, `BudgetExhausted`, `DeepAgent(k=20, budget=...)`.

- [ ] **Step 1: Write the budget tests**

```python
# tests/harness/test_budget.py
import pytest

from stark_bench.harness.budget import Budget, BudgetExhausted


def test_it_permits_calls_up_to_the_cap():
    budget = Budget(max_tool_calls=2, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    budget.spend_tool()
    with pytest.raises(BudgetExhausted):
        budget.spend_tool()


def test_tool_and_llm_budgets_are_separate():
    """One counter for both would let a cheap tool loop starve the LLM."""
    budget = Budget(max_tool_calls=1, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    budget.spend_llm()


def test_exhaustion_is_recorded_not_merely_raised():
    budget = Budget(max_tool_calls=1, max_llm_calls=1, max_seconds=10.0)
    budget.spend_tool()
    with pytest.raises(BudgetExhausted):
        budget.spend_tool()
    assert budget.exhausted is True
```

- [ ] **Step 2: Write the deep agent tests**

```python
# tests/agents/test_deep.py
import pytest

from stark_bench.agents.deep import DeepAgent
from stark_bench.harness.budget import Budget
from stark_bench.ports import Query, Ranked, ToolCall


class Tools:
    def __init__(self):
        self.calls: list[ToolCall] = []
        self.searches = 0

    async def search_chunks(self, text, *, k=10, mode="hybrid"):
        self.searches += 1
        return [Ranked("1", 0.9)]

    async def get_node(self, node_id): return {"node_id": node_id, "name": "n", "node_type": "t"}
    async def neighbors(self, node_id, *, depth=1): return ["2", "3"]
    async def get_relationships(self, node_id): return [("1", "targets", "2")]
    async def complete(self, prompt): return "SEARCH: more terms"


@pytest.mark.asyncio
async def test_it_returns_best_so_far_when_the_budget_runs_out():
    """Budget exhaustion is a recorded outcome, not an exception that voids
    the run: an agent is scored on what it had at the cap."""
    tools = Tools()
    agent = DeepAgent(k=20, budget=Budget(max_tool_calls=3, max_llm_calls=2, max_seconds=30.0))
    result = await agent.retrieve(Query(1, "x"), tools)
    assert result
    assert tools.searches <= 3


@pytest.mark.asyncio
async def test_it_terminates_even_when_the_llm_always_asks_for_more():
    """A loop whose exit depends on model output must be hard-bounded.

    A test that hangs is worse than one that fails: in CI it reads as
    infrastructure trouble and gets retried rather than investigated.
    """
    tools = Tools()
    agent = DeepAgent(k=20, budget=Budget(max_tool_calls=5, max_llm_calls=5, max_seconds=30.0))
    result = await agent.retrieve(Query(1, "x"), tools)
    assert isinstance(result, list)
```

- [ ] **Step 3: Implement, run, and commit**

```bash
uv run pytest tests/agents/test_deep.py tests/harness/test_budget.py -v -p no:randomly
git add -A && git commit -m "The deep agent, bounded by an explicit budget it reports"
```

- [ ] **Step 4: Run all four architectures and write the comparison**

```bash
for agent in dense hybrid zero_shot deep; do
  uv run python -m stark_bench.harness.cli --config config/redstring-native.yaml --agent "$agent" --run
done
uv run python -m stark_bench.harness.cli --summarise results/ > RESULTS.md
```

`RESULTS.md` reports accuracy **and** cost per architecture. A number without its cost beside it is not actionable.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "All four architectures measured on stark-prime, with cost beside accuracy"
```

---

## Self-review notes

- **Spec coverage:** ingest (T5), identity (T3), chunkers and the control (T5/T12), aggregation (T6), the seam (T2), toolset (T8), scoring (T7), sidecar (T7/T10), backing (T11), all four agents (T9/T13/T14), config and reporting (T11), both enforced rules (T1). Deferred items stay deferred.
- **Known soft spots for the implementer:** every `stark_qa` attribute in T10 and the `Evaluator` call in T7 are written from documentation and must be corrected against the installed package before use. Both tasks say so at the step where it matters.

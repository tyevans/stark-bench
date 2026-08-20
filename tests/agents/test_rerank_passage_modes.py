"""`passage_mode` renders what it says, and the agent actually uses it.

The AST tests are here for the reason CLAUDE.md records twice: a helper can
be perfect and unreachable, and the tests written to prevent that pass
because the helper is correct. Running the real call site needs Postgres,
Neo4j and an endpoint, so the syntax tree is the cheap check with teeth.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from stark_bench.agents import rerank as rerank_mod
from stark_bench.agents.rerank import RerankAgent, first_relations, title_of

DOC = (
    "- name: DCC mediated attractive signaling\n"
    "- type: pathway\n"
    "- source: REACTOME\n"
    "- details:\n"
    "  - dbId: 418885\n"
    "  - name: ['DCC mediated attractive signaling']\n"
    "- relations:\n"
    "  interacts_with: {gene/protein: (RAC1, DCC, NTN1)}\n"
    "  member_of: {pathway: (Axon guidance)}\n"
)


def test_title_is_name_and_type_only() -> None:
    assert title_of(DOC) == "DCC mediated attractive signaling (pathway)"


def test_title_ignores_a_name_nested_in_details() -> None:
    """The details `name` is a display list; picking it is the wrong answer.

    `DOC` cannot catch this on its own -- its real name comes first, so
    `search` finds it whether the pattern is anchored or not. The document
    that separates the two implementations is one whose ONLY `name:` is
    nested, where a loose pattern happily returns the display list."""
    nested_only = "- type: pathway\n- details:\n  - name: ['display form']\n"
    assert "display form" not in title_of(nested_only)
    assert "['DCC" not in title_of(DOC)


def test_title_falls_back_when_there_is_no_name_line() -> None:
    """`? (?)` for forty candidates is a reranker scoring noise, not a bug
    anyone would notice in a report."""
    assert title_of("- summary: an entity with no name line\n").startswith("- summary:")


def test_first_relations_takes_one_name_per_type() -> None:
    out = first_relations(DOC)
    assert "RAC1" in out and "DCC" not in out.split("RAC1")[1]
    assert "Axon guidance" in out


def test_first_relations_is_empty_without_a_relations_block() -> None:
    assert first_relations("- name: X\n- type: y\n") == ""


def test_first_relations_bounds_a_hub_node() -> None:
    doc = "- relations:\n" + "".join(f"  rel{i}: {{t: (N{i})}}\n" for i in range(40))
    assert first_relations(doc, max_types=8).count(";") == 7


def test_per_type_zero_is_refused() -> None:
    """Silently returning no names would delete the signal it exists to carry."""
    with pytest.raises(ValueError, match="per_type"):
        first_relations(DOC, per_type=0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("title", "DCC mediated attractive signaling (pathway)"),
        ("title_rel", "DCC mediated attractive signaling (pathway) | interacts_with"),
    ],
)
def test_render_passage_dispatches_on_mode(mode: str, expected: str) -> None:
    agent = RerankAgent(passage_mode=mode)
    assert agent._render_passage(DOC, "q").startswith(expected)


def test_full_mode_still_returns_the_document() -> None:
    assert RerankAgent()._render_passage(DOC, "q") == DOC


def test_title_mode_is_dramatically_shorter() -> None:
    """The whole point. A mode that did not shrink the prompt would pass
    every test above and measure nothing."""
    assert (
        len(RerankAgent(passage_mode="title")._render_passage(DOC, "q")) < len(DOC) / 4
    )


def test_an_unknown_mode_raises_rather_than_falling_back() -> None:
    with pytest.raises(ValueError, match="unknown passage_mode"):
        RerankAgent(passage_mode="titel")._render_passage(DOC, "q")


def _retrieve_tree() -> ast.AST:
    src = inspect.getsource(RerankAgent.retrieve)
    return ast.parse(textwrap.dedent(src))


def test_retrieve_builds_texts_through_the_renderer() -> None:
    """The defect this file exists for: `_render_passage` correct, unused."""
    calls = {
        n.func.attr
        for n in ast.walk(_retrieve_tree())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_render_passage" in calls


class _FakeToolset:
    def __init__(self, passages):
        self._passages = passages

    async def search_passages(self, text, *, k=10, mode="hybrid"):
        return self._passages[:k]

    async def extract(self, prompt, schema):  # pragma: no cover - never reached
        raise AssertionError("extract must not be called on a blank batch")


async def test_retrieve_refuses_an_all_empty_render() -> None:
    """A blank batch scores like a slightly-worse `hybrid` and logs nothing.

    Asserted behaviourally rather than on the syntax tree: `if False: raise`
    leaves the `Raise` node in place, so an AST check cannot tell a live
    guard from a dead one."""
    from stark_bench.domain import Passage, Query

    blank = [Passage(node_id=str(i), text="   ", score=1.0) for i in range(3)]
    agent = RerankAgent(k=3, fetch=3, passage_mode="title")

    with pytest.raises(ValueError, match="empty passages"):
        await agent.retrieve(Query(query_id=1, text="q"), _FakeToolset(blank))


def test_the_registry_exposes_the_title_arms() -> None:
    from stark_bench.composition.agent_registry import AGENTS

    assert {"rerank40title", "rerank40titlerel"} <= set(AGENTS)


def test_module_defines_no_unused_public_helper() -> None:
    """`title_of` must be reachable from the agent, not just exported."""
    src = Path(inspect.getfile(rerank_mod)).read_text(encoding="utf-8")
    body = src.split("class RerankAgent")[1]
    assert "title_of(" in body and "first_relations(" in body

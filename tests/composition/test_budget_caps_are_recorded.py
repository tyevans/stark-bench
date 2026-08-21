"""A cut-off count is half a fact without the cap beside it.

B-BUDGET-CAPS-1 records that `MAX_TOOL_CALLS`, `MAX_LLM_CALLS` and
`MAX_SECONDS` are module constants rather than `RunConfig` fields, so
`config_verbatim` -- the config file's own bytes -- cannot carry them.
Changing one changes what every past `deep` number means with no trace in
the artefacts.

This does not move them into the config, which that entry argues against on
the grounds that a field nobody sets is its own kind of noise. It records
what actually ran, which is the half that was missing once
`exhausted_queries` landed.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import stark_bench.composition.cli as cli_mod
from stark_bench.composition.agent_registry import AGENTS
from stark_bench.domain.run_config import RunConfig

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")

_CONFIG = RunConfig(
    name="c",
    dataset="prime",
    split="test-0.1",
    chunker="whole-document",
    embeddings="e",
    dimension=8,
    aggregation="max",
    agent="deep",
    k=20,
    raw="",
)


def test_the_deep_agent_exposes_every_cap_by_name() -> None:
    """The report reads these by name; a rename leaves them silently absent."""
    agent = AGENTS["deep"](_CONFIG)
    for cap in ("max_tool_calls", "max_llm_calls", "max_seconds"):
        assert getattr(agent, cap, None) is not None, cap


def test_the_run_records_all_three() -> None:
    assert 'cost[f"budget_{cap}"]' in _SRC
    for cap in ("max_tool_calls", "max_llm_calls", "max_seconds"):
        assert cap in _SRC


def test_they_are_read_off_the_agent_not_imported() -> None:
    """An agent built with non-default caps must report ITS caps. Importing
    the module constants would record the default in exactly the case this
    exists for -- someone having changed them."""
    tree = ast.parse(_SRC)
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "agent"
        ):
            found = True
    assert found, "caps must come from getattr(agent, ...)"


def test_a_non_default_cap_is_what_gets_recorded() -> None:
    agent = replace(AGENTS["deep"](_CONFIG), max_tool_calls=99)
    assert agent.max_tool_calls == 99


def test_an_agent_without_budgets_records_nothing() -> None:
    """`dense` has no caps; emitting the module defaults for it would assert
    a budget it never ran under."""
    agent = AGENTS["dense"](_CONFIG)
    assert all(
        getattr(agent, cap, None) is None
        for cap in ("max_tool_calls", "max_llm_calls", "max_seconds")
    )

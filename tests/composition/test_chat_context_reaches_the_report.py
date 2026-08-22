"""The context a chat arm actually ran under must land in its report.

## The incident this is for

`qwen3.8-27b-64k-txt` is served from one `--ctx-size 65536` process split
by `-np`, so its per-slot budget is `65536 / -np` while its id says `64k`
regardless. On 2026-08-21, at `-np 4`, a `rerank40` arm over whole PRIME
documents had **72 of its first 79** LLM calls rejected with
`exceed_context_size_error` -- and kept going, because `rerank` catches
extract failures and falls back to retrieval order.

The same afternoon showed why probing is not enough on its own.
`qwen-rel-whole` + `rerank40` scored **0.46323** with 280/280 calls
succeeding, which is impossible at 16,384 tokens. That number was taken at
a lower `-np`, nothing in its report says so, and it therefore reads as
reproducible when it is not. `agent_warnings` would now catch the failing
run; only a recorded `chat_n_ctx` distinguishes the two *passing* ones.

## Why AST, and why the call site rather than the helper

Running the real call site needs Postgres, Neo4j and a live endpoint. And
the defect shape this project keeps hitting is not a broken helper -- it is
a correct helper nobody calls, which no test of the helper can see. Twice
in one session a correct helper shipped unused with the whole suite green;
`test_ingest_stats_reach_the_report.py` is the original of this pattern and
`test_retrieval_stats_reach_the_report.py` is its sibling.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stark_bench.adapters.model_preflight as preflight_module
import stark_bench.composition.cli as cli_module


def _cli_tree() -> ast.Module:
    return ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_the_run_path_probes_the_chat_context_window() -> None:
    """A helper the run path never calls records nothing."""
    assert _calls_named(_cli_tree(), "chat_context_window"), (
        "cli.py never calls chat_context_window, so no report can carry the "
        "context its LLM calls actually ran under"
    )


def test_the_probe_result_is_stored_under_chat_n_ctx() -> None:
    """Calling it and discarding it is the same as not calling it.

    Asserts the assignment target rather than merely that a call exists,
    because the failure being guarded is a value computed and dropped.
    """
    stored = [
        node
        for node in ast.walk(_cli_tree())
        if isinstance(node, ast.Assign)
        and _calls_named(node.value, "chat_context_window")
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "chat_n_ctx"
            for target in node.targets
        )
    ]
    assert stored, (
        "chat_context_window's result is not assigned to cost['chat_n_ctx']; "
        "a probe whose answer is discarded leaves the report just as silent"
    )


def test_the_probe_is_passed_the_model_that_will_actually_run() -> None:
    """`--chat-model` overrides the config, and the probe must follow it.

    Probing `DEFAULT_CHAT_MODEL` unconditionally would record the right
    number for the wrong model on exactly the runs where the override is
    used -- which is every model A/B this project runs.
    """
    for call in _calls_named(_cli_tree(), "chat_context_window"):
        source = ast.dump(call)
        assert "effective_chat_model" in source, (
            "chat_context_window is called without config.effective_chat_model, "
            "so a --chat-model run would record another model's context"
        )


def test_the_probe_cannot_abort_a_run() -> None:
    """It returns `None` on failure rather than raising, by construction.

    `require_chat_model` in the same module documents the rule: an
    unreachable endpoint is a different fact from a wrong model, and a
    preflight that ends runs is a new way to lose them. This asserts the
    helper has no bare `raise`, so the rule survives an edit.
    """
    tree = ast.parse(Path(preflight_module.__file__).read_text(encoding="utf-8"))
    probe = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "chat_context_window"
    )
    raises = [node for node in ast.walk(probe) if isinstance(node, ast.Raise)]
    assert not raises, (
        "chat_context_window raises; a context probe that can abort a run is a "
        "new failure mode rather than a guard against one"
    )

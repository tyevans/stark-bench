"""The wrapper is useless unless `_do_run` both wraps *and* prewarms.

This repo has now shipped the same defect twice, hours apart, in unrelated
code: `_live_embeddings_for` reverted to dropping both `*_prefix=` arguments
with 39 tests green, and `write_report` reverted to `ingest={}` with 45 green.
Both times the tests covered the helper exhaustively and nothing asserted the
call site used it -- and no test of a helper can see that, because the helper
is correct.

So this file tests `cli.py` as a *syntax tree*. Running `_do_run` for real
needs Postgres, Neo4j and a live endpoint, which is exactly the reason the
call site went untested the first two times.

Two assertions, not one. Wrapping without prewarming still works -- every
query simply misses and is embedded live -- and would leave a run that looks
correct, reports `query_embed_hits: 0`, and is exactly as slow as before.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import stark_bench.composition.cli as cli_module
from stark_bench.adapters.prewarmed_query_embeddings import PrewarmedQueryEmbeddings


def _do_run_tree() -> ast.AST:
    source = Path(inspect.getfile(cli_module)).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_run":
            return node
    raise AssertionError("_do_run not found in cli.py")


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_the_live_provider_is_wrapped_before_it_reaches_the_toolset():
    """Catches: `embeddings = _live_embeddings_for(config)` restored by hand."""
    wrapping = [
        call
        for call in _calls(_do_run_tree())
        if isinstance(call.func, ast.Name)
        and call.func.id == "PrewarmedQueryEmbeddings"
    ]

    assert wrapping, "_do_run must wrap the live provider in PrewarmedQueryEmbeddings"
    inner = wrapping[0].args[0]
    assert isinstance(inner, ast.Call)
    assert isinstance(inner.func, ast.Name)
    assert (
        inner.func.id == "_live_embeddings_for"
    ), "the wrapper must sit around the live provider, not around something else"


def test_the_run_actually_prewarms():
    """Catches: wrapping but never calling `prewarm`.

    That variant is the dangerous one -- it is silent, correct, and slow.
    """
    prewarms = [
        call
        for call in _calls(_do_run_tree())
        if isinstance(call.func, ast.Attribute) and call.func.attr == "prewarm"
    ]

    assert prewarms, "_do_run must call prewarm() before running the agent"


def test_prewarming_happens_before_the_stores_are_opened():
    """Catches: prewarming while holding a Postgres connection.

    An ingest may be writing to the same database, and a scoring pass
    contending there has already cost an in-flight ingest 36% of its rate.
    """
    tree = _do_run_tree()
    prewarm_line = min(
        call.lineno
        for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "prewarm"
    )
    connect_line = min(
        call.lineno
        for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "connect"
    )

    assert prewarm_line < connect_line


def test_the_counters_reach_the_report():
    """Catches: the stats existing and nothing rendering them.

    Without this the only proof the optimisation ran is a wall-clock
    difference, which is the kind of evidence this project has repeatedly
    found to be misleading.
    """
    tree = _do_run_tree()
    stats = [
        call
        for call in _calls(tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "stats"
    ]

    assert stats, "_do_run must fold embeddings.stats() into the reported cost"


def test_the_wrapper_satisfies_the_shape_the_toolset_requires():
    """Catches: a wrapper missing `embed_query`, which fails only at query 1.

    `PrecomputedEmbeddingProvider` records the same trap: a provider without
    `embed_query` raises `AttributeError` after a full ingest.
    """
    for name in ("embed", "embed_query", "model", "dimension"):
        assert hasattr(PrewarmedQueryEmbeddings, name), name

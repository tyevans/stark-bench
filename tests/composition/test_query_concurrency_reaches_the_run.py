"""`--query-concurrency` must reach `run_queries.run`.

Twice in this project a helper was correct and unreachable, and both times
the tests written to prevent it passed -- because the helper was correct.
Running the real call site needs Postgres, Neo4j and an endpoint, so the
syntax tree is the check that is both cheap and has teeth.
"""

from __future__ import annotations

import ast
from pathlib import Path

import stark_bench.composition.cli as cli_mod

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _call_to_run() -> ast.Call:
    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run"
            and any(kw.arg == "checkpoint" for kw in node.keywords)
        ):
            return node
    raise AssertionError("no call to run_queries.run found in cli.py")


def test_the_run_is_passed_a_concurrency() -> None:
    assert any(kw.arg == "concurrency" for kw in _call_to_run().keywords)


def test_the_concurrency_comes_from_the_config_not_a_literal() -> None:
    """A hardcoded 4 would ignore the flag while looking correct."""
    kw = next(k for k in _call_to_run().keywords if k.arg == "concurrency")
    assert isinstance(kw.value, ast.Attribute)
    assert kw.value.attr == "query_concurrency"


def test_the_flag_exists_and_defaults_to_one() -> None:
    """Defaulting to anything else would silently change the wall time of
    every rerun of a previously-recorded arm."""
    assert '"--query-concurrency"' in _SRC
    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args
            and getattr(node.args[0], "value", "") == "--query-concurrency"
        ):
            default = next(k for k in node.keywords if k.arg == "default")
            assert default.value.value == 1
            return
    raise AssertionError("--query-concurrency not registered with argparse")


def test_the_flag_is_threaded_onto_the_config() -> None:
    """Parsed and then dropped is the same as absent."""
    assert "args.query_concurrency" in _SRC
    assert "query_concurrency=args.query_concurrency" in _SRC

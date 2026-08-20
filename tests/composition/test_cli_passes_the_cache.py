"""A perfect cache nobody passes is a cache that does nothing.

This repo has shipped that exact defect twice in one session, hours apart in
unrelated code, with a fully green suite both times: `_live_embeddings_for`
lost both `*_prefix=` arguments and 39 tests passed; `write_report` was
reverted to `ingest={}` and 45 passed. In both cases the helper was
exhaustively tested and nothing asserted the call site used it.

Running `_do_ingest` for real needs Postgres, Neo4j and a live endpoint, so
the call site is checked by reading the source -- the pattern from
`test_ingest_stats_reach_the_report.py`.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from stark_bench.composition import cli


def _call_named(source: str, name: str) -> ast.Call:
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            found = getattr(func, "id", None) or getattr(func, "attr", None)
            if found == name:
                return node
    raise AssertionError(f"no call to {name} found")


def _ingest_call() -> ast.Call:
    return _call_named(inspect.getsource(cli._do_ingest), "ingest_corpus")


def test_the_cache_reaches_the_ingest():
    passed = {kw.arg for kw in _ingest_call().keywords}
    assert "embedding_cache" in passed, (
        "_do_ingest builds an embedding cache and does not pass it to "
        "ingest_corpus -- the cache would be created, connected, and never read"
    )


def test_the_key_fields_reach_the_ingest():
    """A cache without the model and prefix mixes two arms' vectors.

    ADR 0002 and ADR 0043: `_table_for` separates arms on exactly these two
    fields, and the cache has to separate the same ones. Passing the cache
    but not the key fields is worse than no cache at all.
    """
    passed = {kw.arg for kw in _ingest_call().keywords}
    assert "cache_model" in passed
    assert "cache_document_prefix" in passed


def test_the_key_fields_come_from_the_config():
    """Hardcoding either field would defeat the separation silently."""
    by_name = {kw.arg: kw.value for kw in _ingest_call().keywords}
    for field, attribute in (
        ("cache_model", "embeddings"),
        ("cache_document_prefix", "document_prefix"),
    ):
        value = by_name[field]
        assert isinstance(value, ast.Attribute), f"{field} is not read from config"
        assert value.attr == attribute, f"{field} reads config.{value.attr}"


def test_a_no_cache_flag_exists():
    """The escape hatch, for the same reason --no-resume has one."""
    source = inspect.getsource(cli.main)
    assert '"--no-cache"' in source


def test_the_flag_reaches_do_ingest():
    call = _call_named(inspect.getsource(cli.main), "_do_ingest")
    passed = {kw.arg for kw in call.keywords}
    assert "use_cache" in passed, "--no-cache is parsed and never acted on"

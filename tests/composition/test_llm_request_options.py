"""`_llm_for` must ask for prompt caching AND stay non-reasoning.

The second is the one with teeth. `openai_compatible` put `NO_THINKING`
into `extra_body`, so building the chat model by hand -- which we now do, to
pass `cache_prompt` -- moves that field into our keeping. An `extra_body`
that omits it turns reasoning back on silently: no error, a slower run, and
non-deterministic extraction that CLAUDE.md records as two thinking-on runs
at temperature zero disagreeing about how many entities a document held.

Every accuracy number in this repository is a difference between two runs.
"""

from __future__ import annotations

import ast
from pathlib import Path


import stark_bench.composition.cli as cli_mod

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")


def _llm_for_source() -> str:
    return _SRC.split("def _llm_for")[1].split("\ndef ")[0]


def test_the_request_asks_for_prompt_caching() -> None:
    assert '"cache_prompt": True' in _llm_for_source()


def test_reasoning_stays_off() -> None:
    """The regression this file exists for: `extra_body` overwritten rather
    than extended."""
    src = _llm_for_source()
    assert "NO_THINKING" in src, "extra_body must carry the no-thinking flag"
    assert (
        "**dict(NO_THINKING)" in src or "**NO_THINKING" in src
    ), "NO_THINKING must be MERGED into extra_body, not replaced by it"


def test_temperature_is_zero() -> None:
    """`openai_compatible` defaulted it; hand-building loses that default."""
    assert "temperature=0.0" in _llm_for_source()


def test_the_model_id_is_the_effective_one() -> None:
    src = _llm_for_source()
    assert "effective_chat_model" in src


def test_no_thinking_is_actually_what_we_think_it_is() -> None:
    """Pins the imported constant. If redstring renamed the field or flipped
    the value, merging it would still pass every string check above."""
    from redstring.llm.adapters.langchain import NO_THINKING

    assert dict(NO_THINKING) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_is_built_as_one_dict_with_both_keys() -> None:
    """Structural, not textual: two separate `extra_body=` assignments would
    satisfy the substring checks while the last one wins."""
    tree = ast.parse(_SRC)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "extra_body":
            found.append(node)
    assert len(found) == 1, f"expected one extra_body, found {len(found)}"
    body = found[0].value
    assert isinstance(body, ast.Dict)
    keys = [k.value for k in body.keys if isinstance(k, ast.Constant)]
    assert "cache_prompt" in keys
    assert any(k is None for k in body.keys), "NO_THINKING must be splatted in"

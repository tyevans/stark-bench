"""Every LLM call this harness makes must be non-reasoning, and nothing said so.

CLAUDE.md records the measurement this rests on: at temperature zero, two
thinking-ON runs disagreed with each other about how many entities a document
held, while two thinking-OFF runs did not. Every accuracy number in this
repository is a difference between two runs, so a reasoning model that
disagrees with itself does not cost precision at the margin -- it makes the
comparison meaningless. It is also 9x slower per `extract` call, which is the
difference between a 2.4-hour phase and a 9-hour one.

The protection was entirely a library default. `LangChainLlmProvider.
openai_compatible` takes `thinking: bool = False` and `_llm_for` does not
pass it, so the harness was correct by inheritance -- and would flip silently
the day redstring changed its mind about the default, or the day someone
added a `thinking=True` while chasing extraction quality. `grep -rn thinking
src/stark_bench/` returned nothing before this file existed.

**This asserts on the constructed object, not on the signature.** The
distinction is the one CLAUDE.md's "the helper works, nobody calls it"
section was written about, twice in one session: an exhaustive test of a
helper cannot see a call site that stopped using it. `_llm_for` is the single
call site -- `cli.py` builds the toolset with `llm=_llm_for(config)` and
`RedstringToolset.extract` forwards to it, so `rerank`, `zero_shot` and
`deep` all reach the server through this one object -- and what goes on the
wire is `extra_body`, so `extra_body` is what is checked.

The server does NOT override this. The endpoint here is launched with
`--reasoning on`; measured against it, same prompt and temperature zero, our
request produced 0 reasoning characters and the server default produced 1503.
The request wins, which is why testing the request is testing the behaviour.
"""

from __future__ import annotations

import pytest

from stark_bench.adapters.config_file import load_config
from stark_bench.composition.cli import _llm_for
from stark_bench.domain.run_config import RunConfig

from pathlib import Path


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        name="test-reasoning",
        dataset="prime",
        split="test-0.1",
        chunker="whole-document",
        embeddings="qwen3-embedding-0.6b",
        dimension=1024,
        aggregation="max",
        agent="rerank",
        k=20,
        raw="",
    )


@pytest.fixture
def llm(config, monkeypatch):
    # `_llm_for` asks the endpoint whether the chat model exists. That is the
    # right thing in production and the wrong thing in a unit test -- it would
    # make this file fail when the machine is off, which is the fastest way to
    # get a test deleted.
    monkeypatch.setattr(
        "stark_bench.composition.cli.require_chat_model", lambda *a, **k: None
    )
    return _llm_for(config)


def test_the_request_carries_enable_thinking_false(llm):
    """The flag has to be on the wire, not merely in a default somewhere."""
    chat = llm._chat
    assert chat.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_temperature_is_zero(llm):
    """Sampling variety is a cost here with no compensating benefit.

    Two runs must agree about the same document. This is the same property
    `enable_thinking: False` protects, reached by the other route.
    """
    assert llm._chat.temperature == 0.0


def test_every_config_on_disk_builds_a_non_reasoning_provider(monkeypatch):
    """The per-config check, because `chat_model:` is overridable per config.

    A config naming its own chat model still goes through `_llm_for`, but a
    future one might not, and this is the cheap way to notice. It also means
    adding a config is enough to be covered -- there is no list here to
    forget to update.
    """
    monkeypatch.setattr(
        "stark_bench.composition.cli.require_chat_model", lambda *a, **k: None
    )
    configs = sorted(Path("config").glob("*.yaml"))
    assert configs, "no configs found -- this test would pass vacuously"
    for path in configs:
        chat = _llm_for(load_config(path))._chat
        assert chat.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False}
        }, f"{path.name} builds a reasoning provider"
        assert (
            chat.temperature == 0.0
        ), f"{path.name} does not extract deterministically"

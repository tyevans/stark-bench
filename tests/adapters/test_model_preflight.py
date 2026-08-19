"""The preflight refuses a renamed model, and tolerates a down endpoint.

Both halves matter. Without the first, a window change renames the served
model and every LLM run scores as plain retrieval while looking fine. Without
the second, a momentarily unreachable endpoint becomes a new way to lose runs
-- trading one failure for another rather than removing one.
"""

from __future__ import annotations

import json
import io
import pytest

from stark_bench.adapters import model_preflight
from stark_bench.adapters.model_preflight import (
    ModelUnavailableError,
    require_chat_model,
)


def _serving(*ids):
    payload = json.dumps({"data": [{"id": i} for i in ids]}).encode()

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return lambda url, timeout=None: _Response(payload)


def test_a_served_model_is_accepted(monkeypatch):
    monkeypatch.setattr(
        model_preflight.urllib.request, "urlopen", _serving("a", "wanted")
    )
    require_chat_model("http://x/v1/", "wanted")


def test_a_renamed_model_is_refused(monkeypatch):
    # The real incident: the window is part of the id, so raising it renames
    # the model and the old name simply stops existing.
    monkeypatch.setattr(
        model_preflight.urllib.request,
        "urlopen",
        _serving("qwen3.8-27b-64k-txt"),
    )
    with pytest.raises(ModelUnavailableError) as excinfo:
        require_chat_model("http://x/v1/", "qwen3.8-27b-32k-txt")
    # The message must name what *is* served, or the reader has to go and
    # ask the endpoint themselves to act on it.
    assert "qwen3.8-27b-64k-txt" in str(excinfo.value)


def test_an_unreachable_endpoint_does_not_block_the_run(monkeypatch):
    def _boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(model_preflight.urllib.request, "urlopen", _boom)
    require_chat_model("http://x/v1/", "anything")

"""Refuse to start a run whose chat model the endpoint does not serve.

Twice in one hour a context-window change renamed the served model --
`qwen3.8-27b-16k-txt` to `-32k-txt` to `-64k-txt`, because the window is in
the id -- and both times the harness kept going against a model that no
longer existed.

What makes that worth a preflight rather than a note in the README is the
shape of the failure. Every chat call 404s; `zero_shot` and `rerank` both
catch LLM errors by design and fall back to plain retrieval; and the run
finishes and writes a complete report. A reranker that never reranked scores
*exactly* `hybrid`, which is a plausible number sitting in the right part of
the table, and the honest conclusion it invites -- "reranking does not help
here" -- is wrong. The second occurrence was caught only because 14 failures
had already been logged when someone happened to look.

An embedding-model mismatch is not silent in the same way and is not checked
here: a missing embedder fails ingest immediately and loudly.
"""

from __future__ import annotations

import json
import urllib.request


class ModelUnavailableError(RuntimeError):
    """The configured chat model is not among those the endpoint serves."""


def require_chat_model(base_url: str, model: str, *, timeout: float = 10.0) -> None:
    """Raise unless `model` is served at `base_url`.

    A network failure here is *not* fatal: the endpoint being briefly
    unreachable is a different fact from it serving a different model, and
    refusing to start on the former would make the preflight a new source of
    lost runs rather than a guard against one.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            served = [m["id"] for m in json.load(response)["data"]]
    except Exception:
        return

    if model not in served:
        raise ModelUnavailableError(
            f"the endpoint at {base_url} does not serve {model!r}; it serves "
            f"{', '.join(sorted(served))}. The context window is part of the "
            f"model id here, so raising it renames the model -- update "
            f"DEFAULT_CHAT_MODEL. Refusing to start, because the LLM agents "
            f"catch extract failures and would score as plain retrieval."
        )

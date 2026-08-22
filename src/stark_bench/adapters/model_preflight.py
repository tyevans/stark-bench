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


#: A prompt no served context can hold. The server rejects it during
#: tokenization, before any prefill, so the probe costs ~1.4s and no GPU
#: work -- measured against the 27B at 65,536 tokens.
_OVERSIZED_TOKENS = 200_000


def chat_context_window(
    base_url: str, model: str, *, timeout: float = 30.0
) -> int | None:
    """The per-slot context the endpoint will actually accept, or `None`.

    ## Why this is probed rather than read from the model id

    The id carries a number and the number is not the answer. This endpoint
    serves `qwen3.8-27b-64k-txt` from a single `--ctx-size 65536` process
    divided by `-np`, so at `-np 4` each request gets **16,384** tokens
    while the id still says `64k`. On 2026-08-21 that cost a rerank arm 72
    of its first 79 LLM calls, and the arm carried on and wrote a report:
    `rerank` catches extract failures and falls back to retrieval order.

    The same run also showed why this must be recorded and not merely
    checked. `qwen-rel-whole` + `rerank40` scored **0.46323** with 280/280
    LLM calls succeeding, which is impossible at 16,384 -- so that number
    was taken at a lower `-np`, and nothing in its report says so. It reads
    as reproducible and is not.

    ## Why an oversized prompt rather than `/props`

    `/props` is not routed through llama-swap; only the `/v1/*` surface is
    reachable, the same limitation that made `/tokenize` unavailable when
    the chunk cap needed one. So the oracle is the server's own rejection,
    which cannot disagree with the server -- and it answers with `n_ctx`
    directly rather than requiring a binary search.

    `None` on any failure, for the reason `require_chat_model` returns on a
    network error: an unreachable endpoint is a different fact from a small
    context, and a probe that aborted runs would be a new way to lose them.
    A `None` in the report reads as "not measured", which is true.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "word " * _OVERSIZED_TOKENS}],
            "max_tokens": 1,
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
            # A 200 means the server took 200,000 tokens, which no model
            # here does. Refusing to guess is better than recording a
            # ceiling this probe did not establish.
            return None
    except urllib.error.HTTPError as error:
        try:
            body = json.load(error)
        except Exception:
            return None
        n_ctx = body.get("error", {}).get("n_ctx")
        return n_ctx if isinstance(n_ctx, int) else None
    except Exception:
        return None

"""Class docstrings must not be shipped to the model as schema descriptions.

Found by reading a real request body on 2026-08-20: pydantic serialises a
BaseModel's `__doc__` into the JSON schema's `description`, so 1,099
characters of this repo's own commentary about token budgets -- roughly 297
tokens -- were being sent on every rerank call. Against a lean 700-800 token
prompt that is a third of the input, spent explaining our reasoning to a
model that was not asked to care.

The engineering rationale now lives in `#:` comments above each class, which
pydantic does not read. The docstrings that remain are one line and
model-facing, because a schema description IS prompt text.

This is a budget, not a ban: a short description can genuinely help the
model, and `PairRelevances` keeps one on its field for exactly that reason.
"""

from __future__ import annotations

import json

import pytest

from stark_bench.agents.rerank import PairRelevances, Relevances, TerseRelevances

#: Generous enough for a real instruction, tight enough that a pasted-in
#: rationale block fails. The observed defect was 1,099 characters.
_MAX_DESCRIPTION_CHARS = 200


def _descriptions(schema: object) -> list[str]:
    found: list[str] = []
    if isinstance(schema, dict):
        value = schema.get("description")
        if isinstance(value, str):
            found.append(value)
        for item in schema.values():
            found.extend(_descriptions(item))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(_descriptions(item))
    return found


@pytest.mark.parametrize("model", [Relevances, TerseRelevances, PairRelevances])
def test_no_single_description_is_an_essay(model: type) -> None:
    for description in _descriptions(model.model_json_schema()):
        assert len(description) <= _MAX_DESCRIPTION_CHARS, (
            f"{model.__name__} ships a {len(description)}-char description "
            "to the model on every call"
        )


@pytest.mark.parametrize("model", [Relevances, TerseRelevances, PairRelevances])
def test_the_whole_schema_stays_small(model: type) -> None:
    """The schema is sent in full, not just its descriptions."""
    assert len(json.dumps(model.model_json_schema())) < 1_000


@pytest.mark.parametrize("model", [Relevances, TerseRelevances, PairRelevances])
def test_no_description_leaks_internal_commentary(model: type) -> None:
    """Catches the specific failure: a `#:` rationale pasted back as a
    docstring. These words belong in the repo, not in the prompt."""
    blob = " ".join(_descriptions(model.model_json_schema())).lower()
    for tell in ("tok/s", "wall time", "prefill", "measured", "budget", "--"):
        assert tell not in blob, f"{model.__name__} description leaks {tell!r}"

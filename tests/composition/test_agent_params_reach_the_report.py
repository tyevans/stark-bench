"""Two runs of one agent at different settings must not look identical.

An agent's parameters live in `composition/agent_registry.py`, which is
code. `config_verbatim` is the config FILE's bytes, so nothing in a report
recorded whether `rephrase` ran at `fetch=40` or `fetch=80` -- the two
differed only in the metric, which reads as an architecture result and is
not one.

That is the third instance of one shape in this repository, all closed the
same way: `retrieval_is_exact` for an index that lives in Postgres,
`chat_n_ctx` for a per-slot context that lives in a server flag, and now
this for a parameter that lives in the registry. Each is a basis the
artifact could not express.
"""

from __future__ import annotations

from pathlib import Path

import stark_bench.composition.cli as cli_module
from stark_bench.agents.decompose import DecomposeAgent
from stark_bench.composition.cli import _agent_params


def test_the_run_path_records_the_agents_parameters() -> None:
    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    assert 'cost["agent_params"] = _agent_params(agent)' in source, (
        "cli.py does not record the agent's parameters, so two runs of one "
        "agent at different settings are indistinguishable on disk"
    )


def test_the_parameters_that_change_a_number_are_captured() -> None:
    """Not a smoke test: these are the fields a reader compares runs on."""
    recorded = _agent_params(DecomposeAgent(rephrase=True))
    for field in ("fetch", "per_query_fetch", "sub_queries", "rank_all", "rephrase"):
        assert field in recorded, f"{field} missing from the recorded parameters"
    assert recorded["fetch"] == 80
    assert recorded["sub_queries"] == 2


def test_two_agents_differing_only_in_fetch_record_differently() -> None:
    """The whole point, stated as the comparison it exists to enable."""
    narrow = _agent_params(DecomposeAgent(fetch=40))
    wide = _agent_params(DecomposeAgent(fetch=80))
    assert narrow != wide
    assert narrow["fetch"] == 40 and wide["fetch"] == 80


def test_non_scalar_fields_are_left_out() -> None:
    """A prompt or a provider is not a parameter anyone compares runs on."""
    recorded = _agent_params(DecomposeAgent())
    assert all(
        isinstance(value, int | float | str | bool) for value in recorded.values()
    ), f"a non-scalar reached the report: {recorded}"


def test_an_agent_that_is_not_a_dataclass_records_nothing_rather_than_raising() -> None:
    """A run must not die because an agent was written differently."""

    class Plain:
        k = 20

    assert _agent_params(Plain()) == {}

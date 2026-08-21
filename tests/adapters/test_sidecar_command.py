"""Scoring uses a prebuilt 3.11 environment when one exists, and resolves
from PyPI when one does not.

B-SIDECAR-RESOLVE-1. `uv run --no-project --with stark-qa --with "numpy<2"`
re-resolves 166 packages on every scoring run. Warm that is ~114ms and
invisible; cold, or with PyPI degraded, it is a hard failure AFTER all
retrieval has been paid for. On 2026-08-19 a `deep` arm finished 280
queries in 46 minutes of shared GPU and then died on a 502 fetching
`anthropic` -- a transitive dependency of `stark-qa` that the sidecar never
imports.

The fallback is the point. Requiring the prebuilt environment, or adding
`--offline`, trades a rare failure for a certain one: the first run on any
machine and every CI checkout would fail instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stark_bench.adapters import stark_scorer


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(stark_scorer, "SIDECAR_VENV", tmp_path)
    return python


def test_a_prebuilt_environment_is_used_directly(fake_venv) -> None:
    assert stark_scorer._sidecar_command() == [str(fake_venv)]


def test_a_prebuilt_environment_touches_no_network(fake_venv) -> None:
    """The whole point: no `uv run --with`, so nothing resolves."""
    command = stark_scorer._sidecar_command()
    assert "uv" not in command
    assert not any(arg.startswith("--with") for arg in command)


def test_without_one_it_falls_back_to_resolving(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stark_scorer, "SIDECAR_VENV", tmp_path / "absent")
    command = stark_scorer._sidecar_command()
    assert command[0] == "uv"
    assert "stark-qa" in command


def test_the_fallback_still_pins_the_two_things_that_matter(
    monkeypatch, tmp_path
) -> None:
    """3.11 because `stark_qa` pulls `ogb` -> `rdkit`; `numpy<2` because
    rdkit crashes on NumPy 2.x. Losing either turns a working fallback into
    a confusing crash inside a subprocess."""
    monkeypatch.setattr(stark_scorer, "SIDECAR_VENV", tmp_path / "absent")
    command = stark_scorer._sidecar_command()
    assert "3.11" in command
    assert "numpy<2" in command


def test_a_venv_directory_without_an_interpreter_falls_back(
    monkeypatch, tmp_path
) -> None:
    """A half-built environment must not be trusted: `uv venv` creates the
    directory before installing anything into it."""
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(stark_scorer, "SIDECAR_VENV", tmp_path)
    assert stark_scorer._sidecar_command()[0] == "uv"


def test_the_command_is_spliced_into_the_subprocess_call() -> None:
    """Catches the helper being correct and unused -- twice a real defect in
    this repo, both times with green tests over the helper itself."""
    source = Path(stark_scorer.__file__).read_text(encoding="utf-8")
    assert "*_sidecar_command()" in source


def test_the_builder_script_exists_and_pins_setuptools() -> None:
    """`tdc.metadata` imports `pkg_resources`, removed in setuptools 81.
    uv's ephemeral environment ships it anyway, so the resolved path
    survives an import the prebuilt path fails on -- found on the first
    build."""
    script = (
        Path(stark_scorer.__file__).resolve().parents[3]
        / "scripts"
        / "build_sidecar_env.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "setuptools<81" in text
    assert "stark_qa.evaluator" in text, "the builder must verify the real import path"

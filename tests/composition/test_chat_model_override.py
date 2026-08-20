"""`--chat-model` must reach the provider, the filename, and the report.

Same three hazards `--split` had, and they are not hypothetical here: the
corpus is keyed on the config NAME, so swapping the chat model deliberately
reuses the same ingested store. Two runs therefore differ ONLY by a value
that `config_verbatim` cannot express, because that field is the config
file's own bytes.

If the filename did not carry the model, the second run would overwrite the
number it exists to be compared against.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import stark_bench.composition.cli as cli_mod
from stark_bench.composition.cli import predictions_path, report_path
from stark_bench.domain.run_config import RunConfig

_SRC = Path(cli_mod.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)

BASE = RunConfig(
    name="qwen-rel-whole",
    dataset="prime-rel",
    split="test-0.1",
    chunker="whole-document",
    embeddings="qwen3-embedding-0.6b",
    dimension=1024,
    aggregation="max",
    agent="rerank40title",
    k=20,
    raw="chat_model: qwen3.8-27b-64k-txt\n",
)


def test_the_effective_model_is_the_override() -> None:
    assert replace(BASE, chat_model_override="gemma").effective_chat_model == "gemma"


def test_without_an_override_the_config_value_wins() -> None:
    assert replace(BASE, chat_model="qwen").effective_chat_model == "qwen"


def test_an_overridden_run_cannot_overwrite_the_baseline() -> None:
    """The failure this prevents: two models, one filename, one number."""
    other = replace(BASE, chat_model_override="gemma-4-26b-qat")
    assert report_path(BASE) != report_path(other)
    assert predictions_path(BASE) != predictions_path(other)
    assert "gemma-4-26b-qat" in report_path(other).name


def test_the_baseline_filename_is_unchanged() -> None:
    """No override must leave existing result filenames exactly as they were,
    or every previously-scored arm becomes unfindable."""
    assert report_path(BASE).name == "qwen-rel-whole.rerank40title.json"


@pytest.mark.parametrize("model", ["org/model:v1", "a/b/c"])
def test_a_model_id_with_path_characters_is_filename_safe(model: str) -> None:
    name = report_path(replace(BASE, chat_model_override=model)).name
    assert "/" not in name and ":" not in name


def test_the_provider_reads_the_effective_model() -> None:
    """Catches the shipped-but-unused defect: the override is stored on the
    config and `_llm_for` keeps building the old model."""
    src = _SRC.split("def _llm_for")[1].split("\ndef ")[0]
    assert "effective_chat_model" in src
    assert "config.chat_model or" not in src


def test_the_report_records_the_model_that_ran() -> None:
    report_src = (
        Path(cli_mod.__file__).parent.parent / "adapters" / "report_file.py"
    ).read_text(encoding="utf-8")
    assert '"chat_model": config.effective_chat_model' in report_src


def test_the_flag_is_threaded_onto_the_config() -> None:
    """Parsed and dropped is the same as absent."""
    assert "chat_model_override=args.chat_model" in _SRC


def test_the_flag_defaults_to_none() -> None:
    """A default would silently change which model every run uses."""
    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args
            and getattr(node.args[0], "value", "") == "--chat-model"
        ):
            default = next(k for k in node.keywords if k.arg == "default")
            assert default.value.value is None
            return
    raise AssertionError("--chat-model not registered with argparse")

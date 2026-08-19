"""Resuming an ingest across a chunking change is worse than starting over.

Not merely stale: a chunk id derives from `(source, text)`, so a changed
chunker writes new ids and the old ones remain as live rows in the same
tenant, still returned by search. The corpus becomes a silent mixture of two
chunkings -- every count inflated, and the arm no longer measures the
granularity its config names.

So the guard refuses on anything short of a byte-identical match, and each
way of being short of one is tested, because the whole value of the guard is
in the cases where it says no.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from scripts.resume_is_safe import resume_is_safe
from stark_bench.application.ingest_corpus import ingest_corpus

CONFIG = "name: arm\nchunker: whole-document\n"


@pytest.fixture
def root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "config" / "arm.yaml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


def _report(root, **fields) -> None:
    (root / "results" / "arm.ingest.json").write_text(
        json.dumps(fields), encoding="utf-8"
    )


def test_an_identical_config_allows_resume(root):
    _report(root, nodes=129375, config_verbatim=CONFIG)
    assert resume_is_safe("arm", root) is True


def test_a_changed_chunker_refuses(root):
    """The case the guard exists for."""
    _report(
        root, nodes=129375, config_verbatim="name: arm\nchunker: sliding-1000-500\n"
    )
    assert resume_is_safe("arm", root) is False


def test_a_whitespace_difference_refuses(root):
    """Byte-identical, not equivalent -- the guard cannot parse intent."""
    _report(root, nodes=129375, config_verbatim=CONFIG + "\n")
    assert resume_is_safe("arm", root) is False


def test_a_report_predating_the_field_refuses(root):
    """No claim about what produced it, so it cannot vouch for the corpus.

    This is the case that would fail *open* under a `.get(...) == ...`
    written without care: `None == None` is true if the source were also
    missing, and a default of `""` would match an empty config file.
    """
    _report(root, nodes=129375)
    assert resume_is_safe("arm", root) is False


def test_a_missing_report_refuses(root):
    assert resume_is_safe("arm", root) is False


def test_a_missing_config_refuses(root):
    _report(root, config_verbatim=CONFIG)
    (root / "config" / "arm.yaml").unlink()
    assert resume_is_safe("arm", root) is False


def test_unreadable_json_refuses(root):
    (root / "results" / "arm.ingest.json").write_text("{not json", encoding="utf-8")
    assert resume_is_safe("arm", root) is False


async def test_the_ingest_report_actually_records_the_field():
    """The guard is inert unless `--ingest` writes what it reads.

    This was an AST grep of `cli.py` for the string `"config_verbatim"`,
    with a docstring explaining that writing a real report needed Postgres
    and an embedding endpoint. That stopped being true when the ingest
    became a use case over an injected engine: the producer now runs in a
    millisecond against a fake, so the guard can assert the actual bytes
    instead of the presence of a literal.

    The old form would also have passed on a `cli.py` that merely mentioned
    the name in a comment, and it failed the moment the field moved to a
    keyword argument -- wrong in both directions, which is what a
    structural stand-in buys you.
    """

    class Counts:
        nodes = edges = chunks = skipped = self_loops_dropped = 0

    async def engine(nodes, edges, /, **kwargs):
        return Counts()

    outcome = await ingest_corpus(
        engine=engine,
        nodes=iter(()),
        edges=iter(()),
        tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
        chunk_index=None,
        edges_ingested=False,
        config_verbatim=CONFIG,
    )

    assert outcome.as_dict()["config_verbatim"] == CONFIG, (
        "the ingest report must carry the config that produced it, "
        "or resume_is_safe has nothing to compare against"
    )

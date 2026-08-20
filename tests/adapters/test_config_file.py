from stark_bench.adapters.config_file import load_config


def test_a_config_round_trips_verbatim(tmp_path):
    """The resolved config is embedded verbatim in every results file.

    That is what makes a number re-runnable, so the raw text is retained and
    not reconstructed from parsed fields.
    """
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: vss-control\n"
        "dataset: prime\n"
        "split: test-0.1\n"
        "chunker: whole-document\n"
        "embeddings: precomputed-ada002\n"
        "dimension: 1536\n"
        "aggregation: max\n"
        "agent: dense\n"
        "k: 20\n"
    )
    config = load_config(path)
    assert config.name == "vss-control"
    assert config.dimension == 1536
    assert "whole-document" in config.raw


def test_the_cli_refuses_an_invocation_that_would_do_nothing():
    """Neither --ingest nor --run must fail, not exit 0 silently.

    A run queue passed `--agent dense` without `--run`. Four arms
    "completed" in a second each with rc=0 and empty logs, and the queue
    gated on the return code, so it accepted them and carried on. An exit
    status is the only thing a shell script can see.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stark_bench.composition.cli",
            "--config",
            "config/nomic-wholedoc.yaml",
            "--agent",
            "dense",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0, "a no-op invocation exited 0"
    assert "nothing to do" in result.stderr

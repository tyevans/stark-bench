from stark_bench.harness.config import load_config


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

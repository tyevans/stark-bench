import pytest

from stark_bench.domain.stark_ids import STARK_ID_KEY, entity_id_for, node_id_of


def test_the_same_node_maps_to_the_same_id_every_time():
    assert entity_id_for("prime", "4242") == entity_id_for("prime", "4242")


def test_the_dataset_is_part_of_the_key():
    """Both components decide the id.

    Written because a mapping that hashed `node_id` alone would pass every
    other test in this file: no fixture uses two datasets at once.
    """
    assert entity_id_for("prime", "4242") != entity_id_for("mag", "4242")


def test_different_nodes_in_one_dataset_differ():
    assert entity_id_for("prime", "1") != entity_id_for("prime", "2")


def test_the_reverse_map_is_a_field_read():
    class FakeEntity:
        external_ids = {STARK_ID_KEY: "4242"}

    assert node_id_of(FakeEntity()) == "4242"


def test_a_node_without_the_key_is_an_error_not_a_none():
    class FakeEntity:
        external_ids: dict[str, str] = {}

    with pytest.raises(KeyError):
        node_id_of(FakeEntity())

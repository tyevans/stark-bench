"""STaRK node ids to redstring entity ids, deterministically.

Deterministic means ingest is idempotent and resumable, and the forward map
needs no stored side-table that could drift from the data.

The reverse direction rides `Entity.external_ids`, a first-class field on the
domain type. No reader port offers lookup-by-external-id, and none is needed:
retrieval hands back whole `Entity` objects, so the reverse map is a field
read on an object we already hold.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid5

from redstring import EntityId

#: Fixed namespace. Changing it re-keys every entity, so it is a constant and
#: never a parameter.
NAMESPACE_STARK = UUID("6f2a1d54-7c3b-5e19-9a4f-2b8c0d1e3f57")

STARK_ID_KEY = "stark_node_id"


class HasExternalIds(Protocol):
    external_ids: dict[str, str]


def entity_id_for(dataset: str, node_id: str) -> EntityId:
    """Map one STaRK node to a redstring entity id.

    Both components are in the key. A map over `node_id` alone would collide
    the moment a second dataset is ingested into the same store.
    """
    return EntityId(uuid5(NAMESPACE_STARK, f"{dataset}:{node_id}"))


def node_id_of(entity: HasExternalIds) -> str:
    """Recover the STaRK node id. Raises `KeyError` if absent.

    Absent means the loader did not populate it, which is a bug worth failing
    on rather than a `None` that silently drops a result from the ranking.
    """
    return entity.external_ids[STARK_ID_KEY]

"""The progress query must be scoped to one arm's tenant.

`native-wholedoc`, `redstring-native` and `native-sliding1k` share a chunk
table and differ only by `tenant_id`, so an unscoped `count(*)` sums three
arms. That is not hypothetical: on 2026-08-19 it made an arm at 133,919
chunks read as 141,673 against an expected ~136,700 -- finished and
overshooting when it was neither.

`scripts/progress.py` exists so the correct query is the easy one. These
tests check the property that makes it correct, not that it can talk to a
database.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "progress.py"

SHARED_TABLE_ARMS = ["native-wholedoc", "redstring-native", "native-sliding1k"]


@pytest.fixture(scope="module")
def progress():
    spec = importlib.util.spec_from_file_location("progress", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_three_nemotron_arms_really_do_share_one_table(progress):
    """The premise. If this ever stops holding, the scoping stops mattering
    and these tests would pass while guarding nothing."""
    tables = {progress.query_for(name)[0] for name in SHARED_TABLE_ARMS}
    assert len(tables) == 1, f"expected one shared table, got {tables}"


def test_arms_sharing_a_table_have_distinct_tenants(progress):
    """The other half of the premise: same table, different tenant."""
    tenants = {progress.query_for(name)[1] for name in SHARED_TABLE_ARMS}
    assert len(tenants) == len(SHARED_TABLE_ARMS), f"tenants collide: {tenants}"


@pytest.mark.parametrize("name", SHARED_TABLE_ARMS)
def test_the_query_filters_by_tenant(progress, name):
    _, _, sql = progress.query_for(name)
    assert "where tenant_id" in sql.lower(), sql


def test_the_tenant_is_a_bound_parameter_not_interpolated(progress):
    """A tenant spliced into the SQL text would work and would be the wrong
    habit in a file whose whole subject is getting this query right."""
    _, tenant, sql = progress.query_for("native-wholedoc")
    assert tenant not in sql
    assert "$1" in sql

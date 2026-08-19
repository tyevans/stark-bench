"""The embeddings sidecar's `.npz` output must be readable by the harness.

Built against a small synthetic `.npz`, never the real ~800MB artifacts and
never network access — the same seam `test_export_contract.py` checks for
the JSONL artifacts.
"""

import numpy as np
import pytest

from stark_bench.adapters.stark_artifacts import (
    read_doc_embeddings,
    read_query_embeddings,
)


@pytest.fixture
def doc_npz(tmp_path):
    path = tmp_path / "doc_emb.npz"
    ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.savez(path, ids=ids, vectors=vectors)
    return path


@pytest.fixture
def query_npz(tmp_path):
    path = tmp_path / "query_emb.npz"
    ids = np.array([1, 2], dtype=np.int64)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.savez(path, ids=ids, vectors=vectors)
    return path


def test_read_doc_embeddings_keys_by_string_id(doc_npz):
    embeddings = read_doc_embeddings(doc_npz)

    assert set(embeddings) == {"10", "20", "30"}
    np.testing.assert_array_equal(embeddings["20"], [3.0, 4.0, 5.0])


def test_read_query_embeddings_keys_by_string_id(query_npz):
    embeddings = read_query_embeddings(query_npz)

    assert set(embeddings) == {"1", "2"}
    np.testing.assert_array_equal(embeddings["1"], [1.0, 0.0])


def test_vector_dtype_is_preserved_as_float(doc_npz):
    embeddings = read_doc_embeddings(doc_npz)

    assert embeddings["10"].dtype == np.float32

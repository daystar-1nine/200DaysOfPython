"""
Unit tests for VectorIndex: storage, similarity lookups, disk persistence, and metadata filtering.
"""
import pytest
import numpy as np
from app.index.vector_index import VectorIndex
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def test_index_initialization():
    index = VectorIndex(dimension=16)
    assert index.dimension == 16
    assert index.size == 0
    assert index.chunks == []


def test_add_chunks_dimension_check(sample_chunks):
    index = VectorIndex(dimension=4)
    # 5-dim vector when 4 was declared
    bad_vectors = np.ones((len(sample_chunks), 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        index.add_chunks(sample_chunks, bad_vectors)


def test_add_chunks_count_mismatch(sample_chunks):
    index = VectorIndex(dimension=4)
    bad_vectors = np.ones((len(sample_chunks) + 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Mismatch between chunks"):
        index.add_chunks(sample_chunks, bad_vectors)


def test_index_search_returns_sorted_results(populated_vector_index):
    # Search for vector identical to chunk 0 [1, 0, 0, 0]
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = populated_vector_index.search(query, top_k=3)
    assert len(results) == 3
    assert results[0].chunk.metadata.chunk_id == "doc_py_c0"
    assert abs(results[0].score - 1.0) < 1e-5
    # Scores must be in non-ascending order
    assert results[0].score >= results[1].score >= results[2].score


def test_index_search_k_larger_than_size(populated_vector_index):
    query = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    results = populated_vector_index.search(query, top_k=100)
    # Only 3 chunks in index
    assert len(results) == 3


def test_index_search_k_one(populated_vector_index):
    query = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    results = populated_vector_index.search(query, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.metadata.chunk_id == "doc_db_c0"


def test_index_search_empty_index():
    index = VectorIndex(dimension=4)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=5)
    assert results == []


def test_index_clear(populated_vector_index):
    assert populated_vector_index.size == 3
    populated_vector_index.clear()
    assert populated_vector_index.size == 0
    assert len(populated_vector_index.chunks) == 0


def test_index_save_and_load(populated_vector_index, tmp_path):
    save_dir = tmp_path / "saved_index"
    populated_vector_index.save(save_dir)

    assert (save_dir / "vectors.npz").exists()
    assert (save_dir / "metadata.json").exists()

    new_index = VectorIndex(dimension=4)
    new_index.load(save_dir)

    assert new_index.size == 3
    assert new_index.dimension == 4
    assert new_index.chunks[0].metadata.chunk_id == "doc_py_c0"
    assert new_index.chunks[2].metadata.chunk_id == "doc_db_c0"

    # Verify search on loaded index matches original
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    res = new_index.search(query, top_k=1)
    assert res[0].chunk.metadata.chunk_id == "doc_py_c0"


def test_index_filter_by_category(populated_vector_index):
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Filter only "dbms" category
    results = populated_vector_index.search(query, top_k=3, filter_category="dbms")
    assert len(results) == 1
    assert results[0].chunk.metadata.category == "dbms"


def test_index_filter_nonexistent_category(populated_vector_index):
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = populated_vector_index.search(query, top_k=3, filter_category="nonexistent_category")
    assert len(results) == 0


def test_index_get_chunk_by_id(populated_vector_index):
    chunk = populated_vector_index.get_chunk("doc_py_c1")
    assert chunk is not None
    assert chunk.metadata.chunk_id == "doc_py_c1"
    assert "Tuples" in chunk.text


def test_index_get_nonexistent_chunk_id(populated_vector_index):
    assert populated_vector_index.get_chunk("nonexistent_id_999") is None


def test_index_load_missing_files_error(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    index = VectorIndex(dimension=4)
    with pytest.raises(FileNotFoundError):
        index.load(empty_dir)


def test_index_incremental_add_chunks(sample_chunks):
    index = VectorIndex(dimension=4)
    v0 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    index.add_chunks([sample_chunks[0]], v0)
    assert index.size == 1

    v1 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    index.add_chunks([sample_chunks[1]], v1)
    assert index.size == 2
    assert index.chunks[1].metadata.chunk_id == "doc_py_c1"


def test_index_zero_query_vector(populated_vector_index):
    zero_q = np.zeros(4, dtype=np.float32)
    results = populated_vector_index.search(zero_q, top_k=2)
    assert len(results) == 2
    # Dot product with zero vector is 0
    assert results[0].score == 0.0

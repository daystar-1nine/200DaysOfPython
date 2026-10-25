"""
Unit tests for Reciprocal Rank Fusion (RRF) and Hybrid search.
"""
import pytest
from app.retrieval.rrf import reciprocal_rank_fusion, ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever, min_max_normalize
from app.index.vector_index import ScoredChunk
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def make_chunk(cid: str, doc_id: str = "d1", text: str = "sample") -> Chunk:
    return Chunk(text=text, metadata=ChunkMetadata(cid, doc_id, 0, "T", "cat", 0, 10, 2, 2))


def test_min_max_normalize_basic():
    scores = {"a": 10.0, "b": 20.0, "c": 30.0}
    norm = min_max_normalize(scores)
    assert norm["a"] == 0.0
    assert norm["b"] == 0.5
    assert norm["c"] == 1.0


def test_min_max_normalize_identical():
    scores = {"a": 5.0, "b": 5.0}
    norm = min_max_normalize(scores)
    assert norm["a"] == 1.0
    assert norm["b"] == 1.0


def test_min_max_normalize_empty():
    assert min_max_normalize({}) == {}


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_invalid_k0():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k0=-1)


def test_rrf_single_list():
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    list1 = [ScoredChunk(c1, 10.0), ScoredChunk(c2, 5.0)]
    fused = reciprocal_rank_fusion([list1], k0=60)
    assert len(fused) == 2
    assert fused[0].chunk.metadata.chunk_id == "c1"
    assert fused[0].score == pytest.approx(1.0 / 61.0)
    assert fused[1].score == pytest.approx(1.0 / 62.0)


def test_rrf_two_lists_deduplication():
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    c3 = make_chunk("c3")

    # List 1: c1 (rank 1), c2 (rank 2)
    # List 2: c2 (rank 1), c3 (rank 2)
    l1 = [ScoredChunk(c1, 1.0), ScoredChunk(c2, 0.8)]
    l2 = [ScoredChunk(c2, 0.9), ScoredChunk(c3, 0.5)]

    fused = reciprocal_rank_fusion([l1, l2], k0=60)
    
    # c2 is present in both lists: 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.0325
    # c1 is only in list 1: 1/61 = ~0.01639
    # c3 is only in list 2: 1/62 = ~0.016129
    assert len(fused) == 3
    assert fused[0].chunk.metadata.chunk_id == "c2"
    assert fused[0].score > fused[1].score


def test_rrf_weights():
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")

    l1 = [ScoredChunk(c1, 1.0)]
    l2 = [ScoredChunk(c2, 1.0)]

    # Give 3x weight to list 2
    fused = reciprocal_rank_fusion([l1, l2], k0=60, weights=[1.0, 3.0])
    assert fused[0].chunk.metadata.chunk_id == "c2"
    assert fused[0].score == pytest.approx(3.0 / 61.0)
    assert fused[1].score == pytest.approx(1.0 / 61.0)


def test_rrf_top_k_truncation():
    c1, c2, c3 = make_chunk("c1"), make_chunk("c2"), make_chunk("c3")
    l1 = [ScoredChunk(c1, 1.0), ScoredChunk(c2, 0.9), ScoredChunk(c3, 0.8)]
    fused = reciprocal_rank_fusion([l1], k0=60, top_k=2)
    assert len(fused) == 2


def test_reciprocal_rank_fusion_class_wrapper():
    rrf = ReciprocalRankFusion(k0=50)
    c1 = make_chunk("c1")
    fused = rrf.fuse([[ScoredChunk(c1, 1.0)]])
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1.0 / 51.0)


def test_hybrid_retriever_initialization_invalid_alpha(bm25_retriever, dense_retriever):
    with pytest.raises(ValueError):
        HybridRetriever(bm25_retriever, dense_retriever, alpha=-0.1)
    with pytest.raises(ValueError):
        HybridRetriever(bm25_retriever, dense_retriever, alpha=1.5)


def test_hybrid_search_weighted(hybrid_retriever):
    results = hybrid_retriever.search_weighted("Python list memory", top_k=3, alpha=0.5)
    assert len(results) > 0
    top_doc = results[0].chunk.metadata.document_id
    assert "doc_python_list" in top_doc


def test_hybrid_search_rrf(hybrid_retriever):
    results = hybrid_retriever.search_rrf("SQL INNER JOIN", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.metadata.document_id == "doc_sql_join"


def test_hybrid_search_dispatch_methods(hybrid_retriever):
    res_rrf = hybrid_retriever.search("HTTP 404 error", top_k=2, method="rrf")
    res_weighted = hybrid_retriever.search("HTTP 404 error", top_k=2, method="weighted")
    assert len(res_rrf) > 0
    assert len(res_weighted) > 0
    assert res_rrf[0].chunk.metadata.document_id == "doc_http_error"
    assert res_weighted[0].chunk.metadata.document_id == "doc_http_error"

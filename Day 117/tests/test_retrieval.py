"""
Unit tests for Keyword, Semantic, and Hybrid retrieval engines and result formatting.
"""
import pytest
import numpy as np
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.top_k import format_retrieval_results, QueryResult
from app.embeddings.embedder import DenseTFIDFEmbedder
from app.index.vector_index import VectorIndex


@pytest.fixture
def retrieval_environment(sample_chunks):
    texts = [c.text for c in sample_chunks]
    embedder = DenseTFIDFEmbedder(dim=8)
    embedder.fit(texts)
    vectors = embedder.embed_batch(texts)

    index = VectorIndex(dimension=8)
    index.add_chunks(sample_chunks, vectors)

    kw = KeywordRetriever()
    kw.index_chunks(sample_chunks)

    sem = SemanticRetriever(embedder, index)
    hyb = HybridRetriever(kw, sem, alpha=0.5)

    return {"chunks": sample_chunks, "kw": kw, "sem": sem, "hyb": hyb}


def test_keyword_retriever_exact_match(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("Python lists mutable collections", top_k=2)
    assert len(results) > 0
    chunk, score = results[0]
    assert chunk.metadata.chunk_id == "doc_py_c0"
    assert score > 0.0


def test_keyword_retriever_dbms_query(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("ACID relational databases integrity", top_k=2)
    assert len(results) > 0
    chunk, score = results[0]
    assert chunk.metadata.chunk_id == "doc_db_c0"


def test_keyword_retriever_empty_query(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("", top_k=3)
    assert results == []


def test_keyword_retriever_completely_unseen_query(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("astrophysics quantum dark matter", top_k=3)
    assert results == []


def test_keyword_retriever_case_insensitivity(retrieval_environment):
    kw = retrieval_environment["kw"]
    res_lower = kw.retrieve("python lists", top_k=1)
    res_upper = kw.retrieve("PYTHON LISTS", top_k=1)
    assert len(res_lower) == 1 and len(res_upper) == 1
    assert res_lower[0][0].metadata.chunk_id == res_upper[0][0].metadata.chunk_id


def test_semantic_retriever_returns_results(retrieval_environment):
    sem = retrieval_environment["sem"]
    results = sem.retrieve("What data structure is mutable in Python?", top_k=2)
    assert len(results) == 2
    assert results[0][0].metadata.category == "python"


def test_semantic_retriever_score_threshold(retrieval_environment):
    sem = retrieval_environment["sem"]
    results = sem.retrieve("unrelated question about space aliens", top_k=2, min_score=0.95)
    assert len(results) == 0


def test_semantic_retriever_empty_query(retrieval_environment):
    sem = retrieval_environment["sem"]
    results = sem.retrieve("", top_k=3)
    assert isinstance(results, list)


def test_hybrid_retriever_balanced(retrieval_environment):
    hyb = retrieval_environment["hyb"]
    results = hyb.retrieve("ACID properties in relational databases", top_k=2)
    assert len(results) > 0
    chunk, score = results[0]
    assert chunk.metadata.chunk_id == "doc_db_c0"


def test_hybrid_retriever_pure_keyword_alpha_zero(retrieval_environment):
    kw = retrieval_environment["kw"]
    sem = retrieval_environment["sem"]
    hyb_kw = HybridRetriever(kw, sem, alpha=0.0)
    res = hyb_kw.retrieve("Python lists mutable", top_k=1)
    assert len(res) == 1
    assert res[0][0].metadata.chunk_id == "doc_py_c0"


def test_hybrid_retriever_pure_semantic_alpha_one(retrieval_environment):
    kw = retrieval_environment["kw"]
    sem = retrieval_environment["sem"]
    hyb_sem = HybridRetriever(kw, sem, alpha=1.0)
    res = hyb_sem.retrieve("immutable dictionary keys", top_k=1)
    assert len(res) == 1
    assert res[0][0].metadata.chunk_id == "doc_py_c1"


def test_hybrid_retriever_invalid_alpha(retrieval_environment):
    kw = retrieval_environment["kw"]
    sem = retrieval_environment["sem"]
    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever(kw, sem, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever(kw, sem, alpha=1.5)


def test_reciprocal_rank_fusion_logic(sample_chunks):
    # Construct two ranked lists of tuples
    r_list1 = [(sample_chunks[0], 0.9)]
    r_list2 = [(sample_chunks[0], 0.8), (sample_chunks[1], 0.7)]

    fused = reciprocal_rank_fusion([r_list1, r_list2], k_constant=60)
    assert len(fused) == 2
    # Chunk 0 appeared in both list 1 and 2, so it must be top ranked
    assert fused[0][0].metadata.chunk_id == "doc_py_c0"
    assert fused[1][0].metadata.chunk_id == "doc_py_c1"
    assert fused[0][1] > fused[1][1]


def test_format_retrieval_results(sample_chunks):
    raw = [
        (sample_chunks[0], 0.8523),
        (sample_chunks[1], 0.6124)
    ]
    formatted = format_retrieval_results(raw, method="hybrid")
    assert len(formatted) == 2
    assert formatted[0].rank == 1
    assert abs(formatted[0].score - 0.8523) < 1e-4
    assert formatted[0].chunk.metadata.chunk_id == "doc_py_c0"
    assert formatted[1].rank == 2


def test_format_retrieval_results_empty():
    assert format_retrieval_results([]) == []


def test_query_result_to_dict(sample_chunks):
    qr = QueryResult(chunk=sample_chunks[0], score=0.75, rank=1, retrieval_method="hybrid")
    d = qr.to_dict()
    assert d["score"] == 0.75
    assert d["rank"] == 1
    assert d["chunk_id"] == "doc_py_c0"
    assert d["category"] == "python"


def test_keyword_retriever_token_overlap_scoring(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("Python lists mutable collections dynamic resizing", top_k=2)
    assert len(results) >= 1
    assert results[0][0].metadata.chunk_id == "doc_py_c0"
    if len(results) > 1:
        assert results[0][1] > results[1][1]


def test_retriever_top_k_larger_than_corpus(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("Python databases", top_k=50)
    assert len(results) <= 3


def test_retriever_zero_top_k(retrieval_environment):
    kw = retrieval_environment["kw"]
    results = kw.retrieve("Python", top_k=0)
    assert results == []


def test_hybrid_retriever_category_filter(retrieval_environment):
    hyb = retrieval_environment["hyb"]
    res = hyb.retrieve("properties", top_k=3, category_filter="dbms")
    for chunk, _ in res:
        assert chunk.metadata.category == "dbms"


def test_hybrid_retriever_custom_alpha_override(retrieval_environment):
    hyb = retrieval_environment["hyb"]
    res = hyb.retrieve("Python lists", top_k=1, alpha=0.0)
    assert len(res) == 1
    assert res[0][0].metadata.chunk_id == "doc_py_c0"

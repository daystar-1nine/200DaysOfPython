"""
Unit tests for Okapi BM25 and Dense vector retrieval engines.
"""
import pytest
import numpy as np

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.index.vector_index import VectorIndex, ScoredChunk
from app.embeddings.embedder import DenseTFIDFEmbedder
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def test_bm25_initialization_defaults():
    retriever = BM25Retriever()
    assert retriever.k1 == 1.5
    assert retriever.b == 0.75
    assert retriever.epsilon == 0.25
    assert retriever.corpus_size == 0
    assert not retriever._is_indexed


def test_bm25_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1.0)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)
    with pytest.raises(ValueError):
        BM25Retriever(b=-0.1)


def test_bm25_empty_index_search():
    retriever = BM25Retriever()
    retriever.index([])
    results = retriever.search("python")
    assert results == []


def test_bm25_indexing_and_scoring(sample_chunks, bm25_retriever):
    assert bm25_retriever.corpus_size == len(sample_chunks)
    assert bm25_retriever.avgdl > 0
    assert len(bm25_retriever.idf) > 0


def test_bm25_exact_keyword_matching(bm25_retriever):
    results = bm25_retriever.search("HTTP 404 error", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top.chunk.metadata.document_id == "doc_http_error"
    assert top.score > 0.0


def test_bm25_sql_inner_join_query(bm25_retriever):
    results = bm25_retriever.search("SQL INNER JOIN table ON", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.metadata.document_id == "doc_sql_join"


def test_bm25_term_frequency_saturation():
    c1 = Chunk(text="python python python python python", metadata=ChunkMetadata("c1", "d1", 0, "T", "cat", 0, 30, 5, 5))
    c2 = Chunk(text="python", metadata=ChunkMetadata("c2", "d2", 0, "T", "cat", 0, 6, 1, 1))
    
    retriever = BM25Retriever(k1=1.5, b=0.0)  # disable length norm to isolate TF saturation
    retriever.index([c1, c2])

    score1 = retriever.score_chunk(["python"], 0)
    score2 = retriever.score_chunk(["python"], 1)
    
    # 5x term frequency does NOT produce 5x score due to asymptotic saturation
    assert score1 > score2
    assert score1 < 5.0 * score2


def test_bm25_document_length_normalization():
    # Long document containing one mention vs short document containing one mention
    short_doc = Chunk(text="gradient descent optimization", metadata=ChunkMetadata("s", "ds", 0, "T", "cat", 0, 20, 3, 3))
    long_doc = Chunk(
        text="gradient descent and a lot of extra words that inflate document length and dilute relevance considerably",
        metadata=ChunkMetadata("l", "dl", 0, "T", "cat", 0, 100, 16, 16)
    )
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index([short_doc, long_doc])

    score_short = retriever.score_chunk(["gradient"], 0)
    score_long = retriever.score_chunk(["gradient"], 1)
    assert score_short > score_long


def test_bm25_unmatched_query_zero_score(bm25_retriever):
    results = bm25_retriever.search("completely non-existent term xyz987")
    assert results == []


def test_dense_retriever_search(dense_retriever):
    results = dense_retriever.search("neural network training loss", top_k=2)
    assert len(results) > 0
    top_doc = results[0].chunk.metadata.document_id
    assert top_doc in {"doc_overfitting", "doc_gradient_descent"}


def test_dense_retriever_empty_query(dense_retriever):
    assert dense_retriever.search("") == []
    assert dense_retriever.search("   ") == []


def test_dense_retriever_category_filter(dense_retriever):
    results = dense_retriever.search("syntax iteration", top_k=5, category="python")
    assert len(results) > 0
    for sc in results:
        assert sc.chunk.metadata.category == "python"


def test_dense_retriever_predicate_filter(dense_retriever):
    results = dense_retriever.search(
        "gradient optimization",
        top_k=5,
        filter_fn=lambda c: "machine_learning" in c.metadata.category
    )
    assert len(results) > 0
    for sc in results:
        assert sc.chunk.metadata.category == "machine_learning"


def test_bm25_get_top_terms(bm25_retriever):
    terms = bm25_retriever.get_top_terms("Python list comprehension memory", top_n=2)
    assert len(terms) <= 2
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in terms)

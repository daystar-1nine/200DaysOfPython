"""
Pytest configuration and shared fixtures for Day 118 Advanced RAG test suite.
"""
import pytest
import numpy as np
from pathlib import Path

from app.ingestion.metadata import ChunkMetadata, DocumentMetadata
from app.chunking.chunker import Chunk
from app.embeddings.embedder import DenseTFIDFEmbedder
from app.index.vector_index import VectorIndex, ScoredChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.context.compressor import ContextCompressor


@pytest.fixture
def sample_chunks():
    """Provides a synthetic diverse collection of text chunks."""
    chunks = [
        Chunk(
            text="Python list comprehension provides a concise syntax for constructing lists eagerly in memory.",
            metadata=ChunkMetadata(
                chunk_id="chunk_py_1",
                document_id="doc_python_list",
                chunk_index=0,
                title="List Comprehensions",
                category="python",
                start_char=0,
                end_char=88,
                token_count=13,
                word_count=13,
                tags=["python", "syntax", "comprehension"]
            )
        ),
        Chunk(
            text="Generator expressions use parentheses and yield items lazily on demand with O(1) memory footprint.",
            metadata=ChunkMetadata(
                chunk_id="chunk_py_2",
                document_id="doc_python_list",
                chunk_index=1,
                title="List Comprehensions",
                category="python",
                start_char=89,
                end_char=187,
                token_count=15,
                word_count=15,
                tags=["python", "memory", "iterators"]
            )
        ),
        Chunk(
            text="The SQL INNER JOIN clause selects records that have matching keys in both joined tables under the ON condition.",
            metadata=ChunkMetadata(
                chunk_id="chunk_sql_1",
                document_id="doc_sql_join",
                chunk_index=0,
                title="SQL Joins",
                category="dbms",
                start_char=0,
                end_char=109,
                token_count=19,
                word_count=19,
                tags=["dbms", "sql", "inner join"]
            )
        ),
        Chunk(
            text="HTTP 404 error indicates the requested resource URI cannot be located by the web server in its routing table.",
            metadata=ChunkMetadata(
                chunk_id="chunk_http_1",
                document_id="doc_http_error",
                chunk_index=0,
                title="HTTP Errors",
                category="systems",
                start_char=0,
                end_char=110,
                token_count=18,
                word_count=18,
                tags=["systems", "http", "http 404 error"]
            )
        ),
        Chunk(
            text="Gradient descent updates parameters in the direction of the negative gradient scaled by learning rate eta.",
            metadata=ChunkMetadata(
                chunk_id="chunk_ml_1",
                document_id="doc_gradient_descent",
                chunk_index=0,
                title="Gradient Descent",
                category="machine_learning",
                start_char=0,
                end_char=107,
                token_count=16,
                word_count=16,
                tags=["machine_learning", "gradient descent", "learning rate"]
            )
        ),
        Chunk(
            text="Overfitting occurs when a neural network memorizes training data noise leading to high validation loss.",
            metadata=ChunkMetadata(
                chunk_id="chunk_ml_2",
                document_id="doc_overfitting",
                chunk_index=0,
                title="Overfitting Regularization",
                category="machine_learning",
                start_char=0,
                end_char=103,
                token_count=15,
                word_count=15,
                tags=["machine_learning", "overfitting", "regularization"]
            )
        )
    ]
    return chunks


@pytest.fixture
def bm25_retriever(sample_chunks):
    """Provides an indexed BM25 retriever."""
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index(sample_chunks)
    return retriever


@pytest.fixture
def dense_retriever(sample_chunks):
    """Provides an indexed dense vector retriever."""
    embedder = DenseTFIDFEmbedder(dim=16)
    texts = [c.text for c in sample_chunks]
    embedder.fit(texts)
    embs = embedder.embed_batch(texts)
    
    index = VectorIndex(dimension=16)
    index.add_chunks(sample_chunks, embs)
    return DenseRetriever(embedder=embedder, index=index)


@pytest.fixture
def hybrid_retriever(bm25_retriever, dense_retriever):
    """Provides a configured hybrid retriever."""
    return HybridRetriever(bm25_retriever=bm25_retriever, dense_retriever=dense_retriever, alpha=0.5, rrf_k0=60)

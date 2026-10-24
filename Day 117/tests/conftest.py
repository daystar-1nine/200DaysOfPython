"""
Shared pytest fixtures and test configurations for Day 117 tests.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure app package is importable
day117_dir = Path(__file__).resolve().parent.parent
if str(day117_dir) not in sys.path:
    sys.path.insert(0, str(day117_dir))

from app.config import RAGConfig
from app.ingestion.loader import Document
from app.ingestion.metadata import DocumentMetadata, ChunkMetadata
from app.chunking.chunker import Chunk, TextChunker
from app.embeddings.embedder import DenseTFIDFEmbedder
from app.index.vector_index import VectorIndex


@pytest.fixture
def sample_config(tmp_path):
    cfg = RAGConfig(
        chunk_size=150,
        chunk_overlap=30,
        embedding_dim=16,
        top_k=2,
        base_dir=tmp_path
    )
    return cfg


@pytest.fixture
def sample_documents():
    return [
        Document(
            content=(
                "# Python Programming\n\n"
                "Python lists are ordered, mutable sequences supporting append, pop, and dynamic resizing. "
                "Tuples are immutable sequences that can be used as dictionary keys when elements are hashable."
            ),
            metadata=DocumentMetadata(
                document_id="doc_py",
                title="Python Programming",
                category="python",
                source_path="raw/python.md",
                file_type=".md",
                char_count=220,
                word_count=25
            )
        ),
        Document(
            content=(
                "# Relational Databases\n\n"
                "Relational databases guarantee transactional integrity using ACID properties. "
                "Durability is implemented using Write-Ahead Logging (WAL) before changes commit to disk."
            ),
            metadata=DocumentMetadata(
                document_id="doc_db",
                title="Relational Databases",
                category="dbms",
                source_path="raw/dbms.md",
                file_type=".md",
                char_count=210,
                word_count=22
            )
        )
    ]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            text="Python lists are mutable ordered collections with dynamic resizing.",
            metadata=ChunkMetadata(
                chunk_id="doc_py_c0",
                document_id="doc_py",
                chunk_index=0,
                title="Python Programming",
                category="python",
                start_char=0,
                end_char=68,
                token_count=10,
                word_count=10,
                source_file="python.md"
            )
        ),
        Chunk(
            text="Tuples are immutable sequences that can be used as dictionary keys.",
            metadata=ChunkMetadata(
                chunk_id="doc_py_c1",
                document_id="doc_py",
                chunk_index=1,
                title="Python Programming",
                category="python",
                start_char=69,
                end_char=137,
                token_count=10,
                word_count=10,
                source_file="python.md"
            )
        ),
        Chunk(
            text="Relational databases guarantee transactional integrity using ACID properties.",
            metadata=ChunkMetadata(
                chunk_id="doc_db_c0",
                document_id="doc_db",
                chunk_index=0,
                title="Relational Databases",
                category="dbms",
                start_char=0,
                end_char=77,
                token_count=9,
                word_count=9,
                source_file="dbms.md"
            )
        )
    ]


@pytest.fixture
def populated_vector_index(sample_chunks):
    index = VectorIndex(dimension=4)
    # 4-dim normalized orthogonal vectors for distinct chunks
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)
    index.add_chunks(sample_chunks, vectors)
    return index

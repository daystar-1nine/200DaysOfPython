"""
End-to-End integration tests for Advanced RAG Pipeline, Grounded Generation, and Citations.
"""
import pytest
from pathlib import Path

from app.config import AdvancedRAGConfig
from app.pipeline import AdvancedRAGPipeline
from app.ingestion.loader import Document
from app.ingestion.metadata import DocumentMetadata
from app.citations.citation import extract_citations_from_scored_chunks, format_citations_block
from app.generation.generator import AdvancedRAGGenerator, UNSUPPORTED_RESPONSE


@pytest.fixture
def mini_pipeline():
    config = AdvancedRAGConfig()
    pipeline = AdvancedRAGPipeline(config)
    docs = [
        Document(
            metadata=DocumentMetadata(
                document_id="doc_py",
                title="Python Comprehensions",
                category="python",
                source_path="py.md",
                file_type="md",
                tags=["python", "syntax"]
            ),
            content="Python list comprehension uses [expr for item in iterable if condition] to allocate memory eagerly."
        ),
        Document(
            metadata=DocumentMetadata(
                document_id="doc_http",
                title="HTTP Status",
                category="systems",
                source_path="http.md",
                file_type="md",
                tags=["systems", "networking"]
            ),
            content="An HTTP 404 error indicates the requested resource URI was not found on the web server."
        )
    ]
    pipeline.ingest_documents(docs)
    return pipeline


def test_pipeline_ingestion(mini_pipeline):
    assert len(mini_pipeline.documents) == 2
    assert len(mini_pipeline.chunks) >= 2
    assert mini_pipeline._is_indexed


def test_pipeline_retrieve_bm25(mini_pipeline):
    results = mini_pipeline.retrieve("HTTP 404 error", top_k=1, method="bm25")
    assert len(results) == 1
    assert results[0].chunk.metadata.document_id == "doc_http"


def test_pipeline_retrieve_dense(mini_pipeline):
    results = mini_pipeline.retrieve("web server missing resource", top_k=1, method="dense")
    assert len(results) == 1
    assert results[0].chunk.metadata.document_id == "doc_http"


def test_pipeline_retrieve_hybrid_rrf(mini_pipeline):
    results = mini_pipeline.retrieve("Python list memory eager", top_k=1, method="hybrid_rrf")
    assert len(results) == 1
    assert results[0].chunk.metadata.document_id == "doc_py"


def test_pipeline_answer_grounded_success(mini_pipeline):
    response = mini_pipeline.answer_query("What does HTTP 404 error mean?", top_k=2)
    assert response.is_supported is True
    assert "404" in response.answer
    assert len(response.citations) > 0
    assert "doc_http" in response.citations[0].document_id


def test_pipeline_answer_unsupported_query_rejection(mini_pipeline):
    response = mini_pipeline.answer_query("What is the speed of light in vacuum?", top_k=2)
    assert response.is_supported is False
    assert response.answer == UNSUPPORTED_RESPONSE
    assert response.citations == []


def test_citations_formatting():
    from app.index.vector_index import ScoredChunk
    from app.chunking.chunker import Chunk
    from app.ingestion.metadata import ChunkMetadata

    c = Chunk(
        text="A test snippet text for verifying citation formatting.",
        metadata=ChunkMetadata("c1", "doc_test", 0, "My Title", "category", 0, 50, 10, 10)
    )
    sc = ScoredChunk(c, 0.9542)
    citations = extract_citations_from_scored_chunks([sc])
    assert len(citations) == 1
    assert citations[0].title == "My Title"
    
    block = format_citations_block(citations)
    assert "[Source: My Title - Chunk 0]" in block
    assert "0.9542" in block

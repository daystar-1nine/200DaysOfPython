"""
Supplementary tests for advanced RAG components, index serialization, and grounding evaluation.
Ensures comprehensive unit coverage across all submodules.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.config import AdvancedRAGConfig
from app.embeddings.embedder import DenseTFIDFEmbedder, SparseTFIDFEmbedder, get_embedder
from app.index.vector_index import VectorIndex, ScoredChunk
from app.chunking.chunker import Chunk, TextChunker, chunk_text
from app.ingestion.metadata import ChunkMetadata, DocumentMetadata
from app.ingestion.cleaner import clean_text, tokenize_words, extract_sentences, normalize_for_search
from app.evaluation.grounding import compute_citation_precision, evaluate_faithfulness
from app.evaluation.error_analysis import diagnose_retrieval_error, ErrorAnalyzer, QueryErrorDiagnosis
from app.citations.citation import Citation


def test_clean_text_unicode_and_whitespace():
    raw = "  Hello \r\n\r\n World \xa0 with   spaces \n\n\n\n\n and blank lines.  "
    cleaned = clean_text(raw)
    assert "Hello" in cleaned
    assert "World" in cleaned
    assert "\xa0" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_tokenize_words_technical_symbols():
    text = "SQLSTATE 23505 and HTTP 404 error with TF-IDF."
    tokens = tokenize_words(text, lowercase=True)
    assert "sqlstate" in tokens
    assert "23505" in tokens
    assert "404" in tokens
    assert "tf-idf" in tokens


def test_extract_sentences_punctuation():
    text = "First sentence here! Second sentence is here? Third sentence is here."
    s = extract_sentences(text)
    assert len(s) == 3
    assert s[0] == "First sentence here!"
    assert s[1] == "Second sentence is here?"
    assert s[2] == "Third sentence is here."


def test_normalize_for_search():
    text = "Hello, World! Here is SQL-JOIN & TF-IDF."
    norm = normalize_for_search(text)
    assert "hello" in norm
    assert "world" in norm
    assert "!" not in norm


def test_text_chunker_invalid_boundaries():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=-5)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=100)


def test_text_chunker_strategies():
    text = "Sentence one about python. Sentence two about machine learning. Sentence three about databases."
    c_sent = chunk_text(text, chunk_size=50, overlap=10, strategy="sentence")
    c_word = chunk_text(text, chunk_size=10, overlap=2, strategy="word")
    c_char = chunk_text(text, chunk_size=40, overlap=10, strategy="character")
    assert len(c_sent) > 0
    assert len(c_word) > 0
    assert len(c_char) > 0


def test_sparse_tfidf_embedder():
    embedder = SparseTFIDFEmbedder()
    corpus = [
        "python list comprehension memory allocation",
        "sql inner join database query table",
        "gradient descent learning rate optimization"
    ]
    embedder.fit(corpus)
    vec = embedder.embed_text("python memory")
    assert isinstance(vec, np.ndarray)
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_get_embedder_factory():
    dense = get_embedder("dense_tfidf", dim=32)
    assert isinstance(dense, DenseTFIDFEmbedder)
    sparse = get_embedder("sparse_tfidf")
    assert isinstance(sparse, SparseTFIDFEmbedder)
    with pytest.raises(ValueError):
        get_embedder("unsupported_type")


def test_vector_index_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        index = VectorIndex(dimension=4)
        c1 = Chunk("Text 1", ChunkMetadata("c1", "d1", 0, "T1", "cat1", 0, 6, 2, 2))
        c2 = Chunk("Text 2", ChunkMetadata("c2", "d2", 0, "T2", "cat2", 0, 6, 2, 2))
        embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        index.add_chunks([c1, c2], embs)
        
        # Save to disk
        index.save(tmp_path)
        assert (tmp_path / "chunks.json").exists()
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "index_meta.json").exists()

        # Load from disk
        loaded = VectorIndex.load(tmp_path)
        assert len(loaded) == 2
        assert loaded.dimension == 4
        assert loaded.get_chunk("c1").text == "Text 1"
        assert loaded.get_chunk("c2").text == "Text 2"


def test_compute_citation_precision():
    context = "An HTTP 404 error indicates the requested resource URI was not found."
    c1 = Citation("c1", "d1", "Title", "cat", 0, "An HTTP 404 error indicates", 0.9)
    c2 = Citation("c2", "d2", "Title", "cat", 0, "Completely bogus hallucinated snippet", 0.9)

    # 1 valid, 1 bogus
    precision = compute_citation_precision([c1, c2], context)
    assert precision == 0.5

    # Both valid
    assert compute_citation_precision([c1], context) == 1.0


def test_evaluate_faithfulness():
    context = "Python manages memory using an internal private heap managed by PyMalloc."
    supported_ans = "Python manages memory through PyMalloc."
    unsupported_ans = "Java uses an advanced garbage collector named ZGC with concurrent marks."

    score_supp = evaluate_faithfulness(supported_ans, context)
    score_unsupp = evaluate_faithfulness(unsupported_ans, context)
    assert score_supp > 0.6
    assert score_unsupp < 0.2


def test_diagnose_retrieval_error_modes():
    # 1. Correct retrieval
    diag1 = diagnose_retrieval_error(
        "query", "q1", "lexical",
        target_doc_ids={"doc_1"},
        retrieved_doc_ids=["doc_1", "doc_2"],
        bm25_top_ids=["doc_1"],
        dense_top_ids=["doc_1"],
        k=5
    )
    assert diag1.error_mode == "correct_retrieval"

    # 2. Lexical mismatch (Dense found it, BM25 missed it)
    diag2 = diagnose_retrieval_error(
        "query", "q2", "semantic",
        target_doc_ids={"doc_target"},
        retrieved_doc_ids=["doc_other_1", "doc_other_2"],
        bm25_top_ids=["doc_other_1"],
        dense_top_ids=["doc_target"],
        k=5
    )
    assert diag2.error_mode == "lexical_mismatch"

    # 3. Semantic confusion (BM25 found it, Dense missed it)
    diag3 = diagnose_retrieval_error(
        "query", "q3", "lexical",
        target_doc_ids={"doc_target"},
        retrieved_doc_ids=["doc_other_1", "doc_other_2"],
        bm25_top_ids=["doc_target"],
        dense_top_ids=["doc_other_1"],
        k=5
    )
    assert diag3.error_mode == "semantic_confusion"

    # 4. Correct rejection
    diag4 = diagnose_retrieval_error(
        "unanswerable query", "q4", "unanswerable",
        target_doc_ids=set(),
        retrieved_doc_ids=[],
        bm25_top_ids=[],
        dense_top_ids=[],
        k=5
    )
    assert diag4.error_mode == "correct_rejection"


def test_error_analyzer_summary():
    analyzer = ErrorAnalyzer()
    diag = QueryErrorDiagnosis("q1", "test", "lexical", ["d1"], ["d1"], "correct_retrieval", "ok")
    analyzer.record(diag)
    dist = analyzer.summarize_distribution()
    assert dist["correct_retrieval"] == 1


def test_advanced_rag_config_directories():
    config = AdvancedRAGConfig()
    assert config.data_raw_dir.exists()
    assert config.data_processed_dir.exists()
    assert config.data_eval_dir.exists()
    assert config.charts_dir.exists()
    assert config.bm25_k1 == 1.5
    assert config.rrf_k0 == 60
    assert config.mmr_lambda == 0.7

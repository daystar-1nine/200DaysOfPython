"""
End-to-end integration tests for RAGPipeline, CLI execution, and RAGEvaluator.
"""
from pathlib import Path
import pytest
from app.config import RAGConfig
from app.pipeline import RAGPipeline
from app.evaluation.evaluator import RAGEvaluator


@pytest.fixture
def e2e_pipeline(tmp_path):
    # Create mini raw files
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    py_file = raw_dir / "python.md"
    py_file.write_text(
        "# Python Basics\n\nPython lists are mutable arrays that support append and pop operations. "
        "Python tuples are immutable sequences used as dictionary keys.",
        encoding="utf-8"
    )

    db_file = raw_dir / "dbms.md"
    db_file.write_text(
        "# Database Systems\n\nRelational databases guarantee ACID transaction safety. "
        "Write-Ahead Logging provides crash durability.",
        encoding="utf-8"
    )

    cfg = RAGConfig(
        chunk_size=120,
        chunk_overlap=20,
        embedding_dim=8,
        top_k=2,
        base_dir=tmp_path
    )
    pipeline = RAGPipeline(cfg)
    pipeline.ingest_directory(raw_dir)
    return pipeline


def test_e2e_pipeline_ingestion_chunk_count(e2e_pipeline):
    assert len(e2e_pipeline.chunks) >= 2
    assert e2e_pipeline._is_ready is True


def test_e2e_pipeline_query_hybrid(e2e_pipeline):
    resp = e2e_pipeline.query("What are Python lists?", top_k=2, method="hybrid")
    assert resp.is_supported is True
    assert "lists are mutable" in resp.answer.lower()
    assert len(resp.citations) >= 1
    assert resp.generation_time_ms > 0.0


def test_e2e_pipeline_query_keyword(e2e_pipeline):
    resp = e2e_pipeline.query("ACID relational databases", top_k=1, method="keyword")
    assert resp.is_supported is True
    assert "relational databases" in resp.answer.lower()


def test_e2e_pipeline_query_semantic(e2e_pipeline):
    resp = e2e_pipeline.query("Which sequences are immutable in Python?", top_k=2, method="semantic")
    assert resp.is_supported is True
    assert len(resp.retrieved_chunks) > 0


def test_e2e_pipeline_anti_hallucination_on_fake_query(e2e_pipeline):
    fake_query = "Who won the Martian Olympics 100m sprint in 2099?"
    resp = e2e_pipeline.query(fake_query, top_k=2, method="hybrid")
    assert resp.is_supported is False
    assert "I don't have enough information in the provided documents" in resp.answer


def test_e2e_pipeline_save_and_reload_index(e2e_pipeline, tmp_path):
    index_dir = tmp_path / "persisted_index"
    e2e_pipeline.save_index(index_dir)

    # Re-instantiate empty pipeline and load
    new_pipe = RAGPipeline(e2e_pipeline.config)
    new_pipe.load_index(index_dir)

    resp = new_pipe.query("Explain ACID properties", top_k=1, method="hybrid")
    assert resp.is_supported is True
    assert "ACID" in resp.context_used


def test_e2e_evaluator_single_item(e2e_pipeline):
    evaluator = RAGEvaluator(e2e_pipeline)
    item = {
        "id": "test_q1",
        "category": "python",
        "question": "What is the mutability of Python lists?",
        "relevant_docs": ["doc_python"],
        "supported": True
    }
    res = evaluator.evaluate_item(item, top_k=2, method="hybrid")
    assert res["question_id"] == "test_q1"
    assert res["precision_at_1"] in (0.0, 1.0)
    assert res["faithfulness"] >= 0.5


def test_e2e_evaluator_unsupported_item(e2e_pipeline):
    evaluator = RAGEvaluator(e2e_pipeline)
    item = {
        "id": "test_q2",
        "category": "general",
        "question": "What is the Hogwarts attendance rule?",
        "relevant_docs": [],
        "supported": False
    }
    res = evaluator.evaluate_item(item, top_k=2, method="hybrid")
    assert not res["is_hallucinated"]
    assert res["error_type"] == "None"


def test_e2e_evaluator_benchmark_run(e2e_pipeline):
    evaluator = RAGEvaluator(e2e_pipeline)
    dataset = [
        {"id": "q1", "category": "python", "question": "Python lists", "relevant_docs": ["doc_python"], "supported": True},
        {"id": "q2", "category": "general", "question": "Quantum antimatter warp", "relevant_docs": [], "supported": False}
    ]
    bench = evaluator.evaluate_benchmark(dataset, top_k=2, method="hybrid")
    s = bench["summary"]
    assert s["total_questions"] == 2
    assert s["mean_faithfulness"] > 0.5


def test_e2e_pipeline_category_filtered_query(e2e_pipeline):
    resp = e2e_pipeline.query("data integrity", top_k=2, method="hybrid", filter_category="dbms")
    for r in resp.retrieved_chunks:
        assert r.chunk.metadata.category == "dbms"


def test_e2e_pipeline_unindexed_query_raises():
    pipe = RAGPipeline()
    with pytest.raises(RuntimeError, match="must be ingested or loaded"):
        pipe.query("test query")


def test_e2e_pipeline_empty_query(e2e_pipeline):
    resp = e2e_pipeline.query("", top_k=2)
    assert not resp.is_supported

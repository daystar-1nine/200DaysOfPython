"""
Unit tests for context building, prompt grounding, citation formatting, and anti-hallucination guardrails.
"""
import pytest
from app.citations.citation import Citation, extract_citations_from_results, verify_citation_coverage
from app.generation.prompt import ContextBuilder, RAGPromptBuilder
from app.generation.generator import RAGGenerator, RAGResponse
from app.retrieval.top_k import QueryResult
from app.evaluation.grounding import evaluate_grounding, compute_token_overlap_f1


def test_citation_tag_formatting():
    c = Citation(
        source_id="chunk_42",
        document_id="doc_py",
        title="Python Basics",
        category="python",
        chunk_index=3,
        snippet="Lists are mutable sequences.",
        relevance_score=0.89
    )
    tag = c.format_citation_tag()
    assert tag == "[Source: Python Basics - Chunk 3]"


def test_citation_serialization():
    c = Citation(
        source_id="c1",
        document_id="d1",
        title="Title",
        category="cat",
        chunk_index=0,
        snippet="Snippet text",
        relevance_score=0.75
    )
    d = c.to_dict()
    assert d["source_id"] == "c1"
    assert d["citation_tag"] == "[Source: Title - Chunk 0]"

    restored = Citation.from_dict(d)
    assert restored.source_id == "c1"
    assert restored.chunk_index == 0


def test_context_builder_budget_enforcement(sample_chunks):
    # Each formatted chunk block is ~123 chars (header + text)
    # With max_chars=180, only 1 chunk fits (123 <= 180, 123 + 123 > 180)
    cb = ContextBuilder(max_chars=180)
    ctx, used_chunks = cb.build_context(sample_chunks)
    assert len(used_chunks) == 1
    assert len(ctx) <= 180


def test_context_builder_empty_chunks():
    cb = ContextBuilder(max_chars=500)
    ctx, used = cb.build_context([])
    assert ctx == ""
    assert used == []


def test_rag_prompt_builder_structure(sample_chunks):
    pb = RAGPromptBuilder()
    prompt, _, _ = pb.build_prompt("What is Python list mutability?", sample_chunks)
    assert "CONTEXT:" in prompt
    assert "USER QUESTION:" in prompt
    assert "What is Python list mutability?" in prompt
    assert "GROUNDED ANSWER:" in prompt
    assert "Python lists are mutable" in prompt


def test_extract_citations_from_results(sample_chunks):
    results = [
        QueryResult(chunk=sample_chunks[0], score=0.88, rank=1),
        QueryResult(chunk=sample_chunks[1], score=0.72, rank=2)
    ]
    citations = extract_citations_from_results(results)
    assert len(citations) == 2
    assert citations[0].source_id == "doc_py_c0"
    assert citations[1].source_id == "doc_py_c1"


def test_verify_citation_coverage_all_present():
    c1 = Citation("c0", "d1", "DocA", "cat", 0, "...", 0.9)
    c2 = Citation("c1", "d1", "DocB", "cat", 1, "...", 0.8)
    answer = f"According to {c1.format_citation_tag()} and {c2.format_citation_tag()}, this is factual."

    res = verify_citation_coverage(answer, [c1, c2])
    assert res["coverage_rate"] == 1.0
    assert len(res["missing_tags"]) == 0


def test_verify_citation_coverage_missing_tags():
    c1 = Citation("c0", "d1", "DocA", "cat", 0, "...", 0.9)
    c2 = Citation("c1", "d1", "DocB", "cat", 1, "...", 0.8)
    answer = f"Only one source is cited {c1.format_citation_tag()}."

    res = verify_citation_coverage(answer, [c1, c2])
    assert res["coverage_rate"] == 0.5
    assert len(res["missing_tags"]) == 1


def test_rag_generator_anti_hallucination_refusal(sample_chunks):
    gen = RAGGenerator(min_relevance_threshold=0.08)
    # Results with very low score (0.01) below threshold (0.08)
    low_res = [QueryResult(chunk=sample_chunks[0], score=0.01, rank=1)]
    resp = gen.generate("Who won the Martian Olympics 2099?", low_res)

    assert not resp.is_supported
    assert "I don't have enough information in the provided documents" in resp.answer
    assert resp.citations == []


def test_rag_generator_valid_grounded_answer(sample_chunks):
    gen = RAGGenerator(min_relevance_threshold=0.08)
    high_res = [QueryResult(chunk=sample_chunks[0], score=0.85, rank=1)]
    resp = gen.generate("What is a Python list?", high_res)

    assert resp.is_supported
    assert "Python lists are mutable" in resp.answer
    assert len(resp.citations) == 1


def test_compute_token_overlap_f1():
    f1 = compute_token_overlap_f1("quick brown fox", "quick brown fox jumps")
    assert f1 > 0.7
    f1_zero = compute_token_overlap_f1("apple orange", "car truck bicycle")
    assert f1_zero == 0.0


def test_evaluate_grounding_supported_answer():
    question = "What is a list in Python?"
    context = "Python lists are mutable ordered collections with dynamic resizing."
    answer = "Python lists are mutable ordered sequences."

    res = evaluate_grounding(question, answer, context, is_supported=True)
    assert not res["is_hallucinated"]
    assert res["faithfulness"] > 0.5
    assert res["context_relevance"] > 0.3


def test_evaluate_grounding_refusal_is_not_hallucination():
    question = "Who is the emperor of planet Saturn?"
    context = ""
    answer = "I don't have enough information in the provided documents to answer this question."

    res = evaluate_grounding(question, answer, context, is_supported=False)
    assert not res["is_hallucinated"]
    assert res["faithfulness"] == 1.0


def test_evaluate_grounding_unsupported_hallucinated():
    question = "Who is the emperor of planet Saturn?"
    context = ""
    answer = "The emperor of planet Saturn is Lord Kronos elected in 2045."

    res = evaluate_grounding(question, answer, context, is_supported=False)
    assert res["is_hallucinated"]


def test_rag_response_serialization():
    resp = RAGResponse(
        question="What is a list?",
        answer="Test answer",
        retrieved_chunks=[],
        context_used="Test context",
        citations=[],
        is_supported=True,
        generation_time_ms=12.5
    )
    d = resp.to_dict()
    assert d["answer"] == "Test answer"
    assert d["is_supported"] is True
    assert d["generation_time_ms"] == 12.5


def test_rag_generator_without_citations(sample_chunks):
    gen = RAGGenerator(min_relevance_threshold=0.05, include_citations=False)
    high_res = [QueryResult(chunk=sample_chunks[0], score=0.85, rank=1)]
    resp = gen.generate("What is a Python list?", high_res)
    assert "[Source:" not in resp.answer

"""
Grounded Answer Generator for Advanced RAG with strict anti-hallucination guardrails and citations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re

from app.index.vector_index import ScoredChunk
from app.citations.citation import (
    Citation,
    extract_citations_from_scored_chunks,
    format_citations_block
)
from app.generation.prompt import AdvancedRAGPromptBuilder
from app.context.compressor import CompressedContext
from app.ingestion.cleaner import tokenize_words


UNSUPPORTED_RESPONSE = (
    "I don't have enough information in the provided documents to answer this question."
)


@dataclass
class RAGResponse:
    """Complete RAG response bundle including answer, citations, and grounding metadata."""
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[ScoredChunk]
    is_supported: bool
    context_used: str
    compression_ratio: float
    generation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "is_supported": self.is_supported,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunk_ids": [sc.chunk.metadata.chunk_id for sc in self.retrieved_chunks],
            "compression_ratio": round(self.compression_ratio, 4),
            "generation_time_ms": round(self.generation_time_ms, 2)
        }


class AdvancedRAGGenerator:
    """
    RAG Answer Generation Engine.
    Enforces strict grounding, detects unsupported queries, and attaches source citations.
    """
    def __init__(
        self,
        prompt_builder: Optional[AdvancedRAGPromptBuilder] = None,
        min_relevance_threshold: float = 0.05,
        include_citations: bool = True
    ):
        self.prompt_builder = prompt_builder or AdvancedRAGPromptBuilder()
        self.min_relevance_threshold = min_relevance_threshold
        self.include_citations = include_citations

    def check_context_support(
        self,
        question: str,
        scored_chunks: List[ScoredChunk]
    ) -> bool:
        """
        Determines whether the retrieved chunks provide sufficient evidence to answer the question.
        """
        if not scored_chunks:
            return False

        max_score = max(sc.score for sc in scored_chunks)
        if max_score < self.min_relevance_threshold:
            return False

        q_tokens = set(tokenize_words(question))
        stop_words = {"what", "is", "the", "how", "to", "in", "a", "an", "and", "or", "of", "can", "you", "tell", "me"}
        meaningful_q_tokens = q_tokens - stop_words

        if not meaningful_q_tokens:
            return True

        corpus_text = " ".join(sc.chunk.text for sc in scored_chunks).lower()
        matched = sum(1 for t in meaningful_q_tokens if t in corpus_text)
        return (matched / len(meaningful_q_tokens)) >= 0.25

    def generate(
        self,
        question: str,
        compressed_context: CompressedContext
    ) -> RAGResponse:
        """
        Generates a grounded response based on compressed context.
        """
        start_time = time.perf_counter()
        used_chunks = compressed_context.chunks_used

        is_supported = self.check_context_support(question, used_chunks)

        if not is_supported or not compressed_context.text.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                question=question,
                answer=UNSUPPORTED_RESPONSE,
                citations=[],
                retrieved_chunks=used_chunks,
                is_supported=False,
                context_used=compressed_context.text,
                compression_ratio=compressed_context.compression_ratio,
                generation_time_ms=elapsed_ms
            )

        # Formulate grounded synthesis
        citations = extract_citations_from_scored_chunks(used_chunks)
        
        # Build answer synthesis from highest-ranked evidence chunk
        primary_chunk = used_chunks[0].chunk
        primary_text = primary_chunk.text.strip()
        lines = [line.strip() for line in primary_text.split("\n") if line.strip() and not line.strip().startswith("#")]
        core_facts = lines[:3] if lines else [primary_text[:200]]

        synthesis_body = " ".join(core_facts)
        # Ensure answer ends with proper punctuation
        if not synthesis_body.endswith((".", "!", "?")):
            synthesis_body += "."

        citation_tag = citations[0].format_citation_tag() if citations else ""
        answer = f"Based on the provided documentation: {synthesis_body} {citation_tag}".strip()

        if self.include_citations and citations:
            citations_block = format_citations_block(citations)
            answer = f"{answer}\n{citations_block}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=used_chunks,
            is_supported=True,
            context_used=compressed_context.text,
            compression_ratio=compressed_context.compression_ratio,
            generation_time_ms=elapsed_ms
        )

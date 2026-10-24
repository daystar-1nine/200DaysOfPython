"""
Grounded RAG answer generator with strict anti-hallucination guardrails and citations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re

from app.retrieval.top_k import QueryResult
from app.citations.citation import (
    Citation,
    extract_citations_from_results,
    format_citations_block
)
from app.generation.prompt import RAGPromptBuilder, ContextBuilder
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
    retrieved_chunks: List[QueryResult]
    is_supported: bool
    context_used: str
    generation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "is_supported": self.is_supported,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunk_ids": [r.chunk.metadata.chunk_id for r in self.retrieved_chunks],
            "generation_time_ms": round(self.generation_time_ms, 2)
        }


class RAGGenerator:
    """
    RAG Answer Generation Engine.
    Enforces strict grounding, detects unsupported queries, and attaches source citations.
    """
    def __init__(
        self,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        min_relevance_threshold: float = 0.08,
        include_citations: bool = True
    ):
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.min_relevance_threshold = min_relevance_threshold
        self.include_citations = include_citations

    def _assess_context_support(
        self,
        question: str,
        results: List[QueryResult]
    ) -> bool:
        """
        Determines whether the retrieved context contains substantive evidence
        relevant to the user's question, guarding against hallucination.
        """
        if not results:
            return False

        # Check maximum retrieval score
        max_score = max(r.score for r in results)
        if max_score < self.min_relevance_threshold:
            return False

        q_tokens = set(tokenize_words(question))
        # Filter out common stop words
        stops = {"what", "is", "the", "in", "of", "and", "a", "an", "to", "how", "does", "do", "explain", "why", "are"}
        content_tokens = q_tokens - stops
        if not content_tokens:
            return True

        # Check word overlap in top retrieved chunks
        top_chunks_text = " ".join(r.chunk.text for r in results[:2])
        top_chunk_tokens = set(tokenize_words(top_chunks_text))
        overlap = content_tokens & top_chunk_tokens
        overlap_ratio = len(overlap) / len(content_tokens)

        # If zero keywords match, context definitely cannot support answer
        if overlap_ratio == 0.0:
            return False

        # Require reasonable keyword overlap and relevance score
        return overlap_ratio >= 0.20 or (max_score >= 0.60 and overlap_ratio >= 0.10)

    def _extract_grounded_answer(
        self,
        question: str,
        results: List[QueryResult]
    ) -> str:
        """
        Extracts the most relevant explanatory sentences directly from the top retrieved chunks.
        """
        top_chunks = results[:min(len(results), 2)]
        all_sentences = []

        q_words = set(tokenize_words(question))
        stops = {"what", "is", "the", "in", "of", "and", "a", "an", "to", "how", "does", "explain"}
        key_words = q_words - stops

        for r in top_chunks:
            # Split into sentences
            s_list = re.split(r"(?<=[.!?])\s+", r.chunk.text.strip())
            for s in s_list:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_tokens = set(tokenize_words(s_clean))
                # Score sentence by overlap with question keywords
                overlap = len(key_words & s_tokens)
                all_sentences.append((s_clean, overlap, r.chunk.metadata))

        if not all_sentences:
            return top_chunks[0].chunk.text.strip()

        # Sort sentences by keyword overlap, then preserve order
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentences = [s[0] for s in all_sentences[:3] if s[1] > 0]

        if not best_sentences:
            # Fall back to first 2 sentences of top chunk
            s_list = re.split(r"(?<=[.!?])\s+", top_chunks[0].chunk.text.strip())
            best_sentences = s_list[:2]

        return " ".join(best_sentences).strip()

    def generate(
        self,
        question: str,
        retrieved_results: List[QueryResult]
    ) -> RAGResponse:
        """
        Generates grounded RAG response for a user query given retrieved context.
        """
        start_time = time.perf_counter()

        prompt, context_str, included_results = self.prompt_builder.build_prompt(
            question, retrieved_results
        )

        is_supported = self._assess_context_support(question, included_results)

        if not is_supported or not included_results:
            answer = UNSUPPORTED_RESPONSE
            citations: List[Citation] = []
        else:
            base_answer = self._extract_grounded_answer(question, included_results)
            citations = extract_citations_from_results(included_results)

            if self.include_citations and citations:
                answer = f"{base_answer}\n{format_citations_block(citations)}"
            else:
                answer = base_answer

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=included_results,
            is_supported=is_supported,
            context_used=context_str,
            generation_time_ms=elapsed_ms
        )

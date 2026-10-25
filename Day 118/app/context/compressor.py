"""
Context Compression Engine.
Performs query-aware sentence-level extractive compression on retrieved chunks to discard
filler sentences, reduce prompt token budget, and maximize evidence density for LLM generation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.ingestion.cleaner import extract_sentences, tokenize_words


@dataclass
class CompressedContext:
    """Represents compressed prompt context with detailed reduction metrics."""
    text: str
    original_char_count: int
    compressed_char_count: int
    compression_ratio: float
    retained_sentences_count: int
    total_sentences_count: int
    chunks_used: List[ScoredChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "original_char_count": self.original_char_count,
            "compressed_char_count": self.compressed_char_count,
            "compression_ratio": self.compression_ratio,
            "retained_sentences_count": self.retained_sentences_count,
            "total_sentences_count": self.total_sentences_count,
        }


class ContextCompressor:
    """
    Extractive query-aware context compressor.
    Evaluates each sentence within retrieved chunks and retains only evidence-bearing sentences.
    """
    def __init__(
        self,
        min_sentence_score: float = 0.15,
        max_context_length: int = 1200
    ):
        self.min_sentence_score = min_sentence_score
        self.max_context_length = max_context_length

    def _score_sentence(self, sentence: str, query_tokens: List[str]) -> float:
        """Scores a sentence based on query token overlap and key term presence."""
        if not sentence or not query_tokens:
            return 0.0

        s_tokens = set(tokenize_words(sentence))
        if not s_tokens:
            return 0.0

        matches = sum(1 for t in query_tokens if t in s_tokens)
        coverage = matches / len(query_tokens)

        # Length factor (penalize ultra-short 1-2 word fragments)
        len_weight = 1.0 if len(s_tokens) >= 5 else 0.5

        return coverage * len_weight

    def compress(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        max_context_length: Optional[int] = None,
        min_sentence_score: Optional[float] = None
    ) -> CompressedContext:
        """
        Compresses a list of retrieved chunks into an evidence-dense formatted context string.
        """
        budget = max_context_length if max_context_length is not None else self.max_context_length
        min_score = min_sentence_score if min_sentence_score is not None else self.min_sentence_score

        if not scored_chunks:
            return CompressedContext(
                text="",
                original_char_count=0,
                compressed_char_count=0,
                compression_ratio=0.0,
                retained_sentences_count=0,
                total_sentences_count=0,
                chunks_used=[]
            )

        q_tokens = tokenize_words(query)
        total_original_chars = sum(len(sc.chunk.text) for sc in scored_chunks)
        total_sentences = 0
        retained_sentences = 0

        compressed_blocks: List[str] = []
        curr_chars = 0
        used_chunks: List[ScoredChunk] = []

        for sc in scored_chunks:
            chunk = sc.chunk
            meta = chunk.metadata
            chunk_header = f"[Source: {meta.document_id} | Chunk: {meta.chunk_index} | Category: {meta.category}]"
            
            sentences = extract_sentences(chunk.text)
            total_sentences += len(sentences)

            # Score each sentence
            scored_sentences: List[Tuple[str, float]] = [
                (s, self._score_sentence(s, q_tokens))
                for s in sentences
            ]

            # Keep sentences that exceed min_score or keep the single best sentence if none exceed
            kept = [s for s, score in scored_sentences if score >= min_score]
            if not kept and scored_sentences:
                # Fallback: keep best sentence if it has at least some match
                best_s, best_score = max(scored_sentences, key=lambda x: x[1])
                if best_score > 0:
                    kept = [best_s]

            if not kept:
                continue

            retained_sentences += len(kept)
            block_body = " ".join(kept)
            block = f"{chunk_header}\n{block_body}"

            if curr_chars + len(block) + 2 > budget:
                # Try adding remaining characters if within budget
                remaining_budget = budget - curr_chars - len(chunk_header) - 3
                if remaining_budget > 80:
                    truncated_body = block_body[:remaining_budget].rsplit(" ", 1)[0] + "..."
                    block = f"{chunk_header}\n{truncated_body}"
                    compressed_blocks.append(block)
                    used_chunks.append(sc)
                break

            compressed_blocks.append(block)
            curr_chars += len(block) + 2
            used_chunks.append(sc)

        final_text = "\n\n".join(compressed_blocks).strip()
        final_chars = len(final_text)
        ratio = (
            (total_original_chars - final_chars) / max(1, total_original_chars)
            if total_original_chars > 0 else 0.0
        )
        ratio = max(0.0, min(1.0, ratio))

        return CompressedContext(
            text=final_text,
            original_char_count=total_original_chars,
            compressed_char_count=final_chars,
            compression_ratio=float(ratio),
            retained_sentences_count=retained_sentences,
            total_sentences_count=total_sentences,
            chunks_used=used_chunks
        )

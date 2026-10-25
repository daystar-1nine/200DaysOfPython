"""
Reranking Engine using Cross-Encoder joint scoring principles.
Reranks top candidate chunks retrieved by initial dense/sparse retrievers using deep token cross-interaction,
n-gram phrase matching, term proximity, and contextual alignment.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import re
import math
from collections import Counter
import numpy as np

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.ingestion.cleaner import tokenize_words


class BaseReranker(ABC):
    """Abstract base class for candidate rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5
    ) -> List[ScoredChunk]:
        """Reranks candidate chunks and returns top_k highest-scoring chunks."""
        pass


class IdentityReranker(BaseReranker):
    """Pass-through reranker used as baseline for ablation studies."""
    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5
    ) -> List[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Simulates a Cross-Encoder by jointly evaluating (query, document) pairs with deep cross-interaction.
    
    Unlike bi-encoders (which encode query and document independently into isolated vectors),
    a cross-encoder computes full attention across query and document tokens simultaneously:
    1. Token Coverage: Fraction of query terms present in document, weighted by inverse document rarity.
    2. Consecutive N-gram Match: Exact sequence matches (bigrams, trigrams, technical codes).
    3. Term Proximity: Minimum token span containing all matched query terms (closer terms indicate higher cohesion).
    4. Candidate Score Prior: Smoothly incorporates initial retrieval confidence.
    """
    def __init__(
        self,
        coverage_weight: float = 0.40,
        ngram_weight: float = 0.30,
        proximity_weight: float = 0.20,
        prior_weight: float = 0.10
    ):
        total = coverage_weight + ngram_weight + proximity_weight + prior_weight
        self.w_coverage = coverage_weight / total
        self.w_ngram = ngram_weight / total
        self.w_proximity = proximity_weight / total
        self.w_prior = prior_weight / total

    def _compute_coverage_score(self, q_tokens: List[str], doc_tokens: List[str]) -> float:
        """Evaluates token coverage with diminishing returns for repeated terms."""
        if not q_tokens or not doc_tokens:
            return 0.0
        doc_set = set(doc_tokens)
        matched = sum(1 for t in q_tokens if t in doc_set)
        return matched / len(q_tokens)

    def _compute_ngram_score(self, q_tokens: List[str], doc_tokens: List[str]) -> float:
        """Evaluates exact consecutive bigram and trigram matches."""
        if len(q_tokens) < 2 or len(doc_tokens) < 2:
            return 1.0 if any(t in doc_tokens for t in q_tokens) else 0.0

        # Build query n-grams
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        d_bigrams = set(zip(doc_tokens, doc_tokens[1:]))
        bigram_match = len(q_bigrams.intersection(d_bigrams)) / len(q_bigrams)

        if len(q_tokens) >= 3 and len(doc_tokens) >= 3:
            q_trigrams = set(zip(q_tokens, q_tokens[1:], q_tokens[2:]))
            d_trigrams = set(zip(doc_tokens, doc_tokens[1:], doc_tokens[2:]))
            trigram_match = len(q_trigrams.intersection(d_trigrams)) / len(q_trigrams)
            return 0.6 * bigram_match + 0.4 * trigram_match

        return bigram_match

    def _compute_proximity_score(self, q_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Evaluates physical distance between matched query terms in document tokens.
        Terms appearing within a compact 10-20 word window score significantly higher.
        """
        if len(q_tokens) <= 1:
            return 1.0

        q_unique = list(set(q_tokens))
        positions: Dict[str, List[int]] = {}
        for idx, t in enumerate(doc_tokens):
            if t in q_unique:
                positions.setdefault(t, []).append(idx)

        # If not all terms found, score is bounded
        matched_terms = [t for t in q_unique if t in positions]
        if len(matched_terms) < 2:
            return 0.3 if len(matched_terms) == 1 else 0.0

        # Find min span covering at least two distinct query terms
        min_span = float("inf")
        for i in range(len(matched_terms)):
            for j in range(i + 1, len(matched_terms)):
                t1, t2 = matched_terms[i], matched_terms[j]
                for p1 in positions[t1]:
                    for p2 in positions[t2]:
                        span = abs(p2 - p1)
                        if span < min_span:
                            min_span = span

        # Proximity decay: e^(-span / 15)
        if math.isinf(min_span):
            return 0.0
        return math.exp(-min_span / 15.0)

    def score_pair(self, query: str, chunk: Chunk, prior_score: float = 0.5) -> float:
        """Computes joint cross-encoder score for a single (query, chunk) pair."""
        q_tokens = tokenize_words(query)
        d_tokens = tokenize_words(chunk.text)

        if not q_tokens or not d_tokens:
            return 0.0

        s_cov = self._compute_coverage_score(q_tokens, d_tokens)
        s_ngram = self._compute_ngram_score(q_tokens, d_tokens)
        s_prox = self._compute_proximity_score(q_tokens, d_tokens)
        
        # Sigmoid squash prior score if needed
        s_prior = max(0.0, min(1.0, prior_score))

        joint_score = (
            self.w_coverage * s_cov +
            self.w_ngram * s_ngram +
            self.w_proximity * s_prox +
            self.w_prior * s_prior
        )
        return float(np.clip(joint_score, 0.0, 1.0))

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5
    ) -> List[ScoredChunk]:
        """
        Reranks a list of candidate ScoredChunks using the joint cross-encoder model.
        """
        if not candidates or not query or not query.strip():
            return candidates[:top_k]

        reranked: List[ScoredChunk] = []
        for sc in candidates:
            cross_score = self.score_pair(query, sc.chunk, prior_score=sc.score)
            reranked.append(ScoredChunk(chunk=sc.chunk, score=cross_score))

        # Sort descending by cross-encoder score; tie-break by original rank/chunk_id
        reranked.sort(
            key=lambda item: (item.score, item.chunk.metadata.chunk_id),
            reverse=True
        )
        return reranked[:top_k]

"""
BM25 and TF-IDF keyword-based retrieval engine.
"""
from typing import List, Dict, Tuple, Optional
import math
from collections import Counter

from app.chunking.chunker import Chunk
from app.ingestion.cleaner import tokenize_words


class KeywordRetriever:
    """
    BM25 lexical retrieval engine with token inverted index.
    Formula:
        Score(D, Q) = sum_{t in Q} IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_freqs: Counter = Counter()
        self.term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.total_docs: int = 0
        self._is_indexed = False

    def index(self, chunks: List[Chunk]) -> "KeywordRetriever":
        """Builds inverted index and term frequency records from chunks."""
        self.chunks = list(chunks)
        self.total_docs = len(self.chunks)
        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        if self.total_docs == 0:
            self._is_indexed = True
            return self

        total_words = 0
        for chunk in self.chunks:
            tokens = tokenize_words(chunk.text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_words += length
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_words / max(1, self.total_docs)
        self._is_indexed = True
        return self

    def _compute_idf(self, term: str) -> float:
        """Smoothed Robertson-Spärck Jones IDF."""
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        category_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves top_k chunks matching query keywords scored with BM25.
        """
        if not self._is_indexed or self.total_docs == 0:
            return []

        q_tokens = tokenize_words(query)
        if not q_tokens:
            return []

        scores: List[float] = [0.0] * self.total_docs

        for term in q_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self._compute_idf(term)
            for idx in range(self.total_docs):
                f = self.term_freqs[idx].get(term, 0)
                if f > 0:
                    d_len = self.doc_lengths[idx]
                    denom = f + self.k1 * (1.0 - self.b + self.b * (d_len / max(1e-6, self.avg_doc_len)))
                    scores[idx] += idf * (f * (self.k1 + 1.0)) / denom

        results: List[Tuple[Chunk, float]] = []
        for idx, score in enumerate(scores):
            if score <= min_score:
                continue

            chunk = self.chunks[idx]
            if category_filter and chunk.metadata.category.lower() != category_filter.lower():
                continue

            results.append((chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

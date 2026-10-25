"""
Okapi BM25 Keyword Retrieval Engine.
Implements the Robertson-Spärck Jones Okapi BM25 probabilistic relevance algorithm with term frequency saturation,
document length normalization, and inverse document frequency (IDF) smoothing.
"""
from typing import List, Dict, Optional, Callable, Tuple
import math
from collections import Counter
import numpy as np

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.ingestion.cleaner import tokenize_words


class BM25Retriever:
    """
    Pure Python & NumPy Okapi BM25 implementation.
    
    Formula:
        BM25(D, Q) = sum_{q in Q} IDF(q) * [ f(q, D) * (k1 + 1) ] / [ f(q, D) + k1 * (1 - b + b * (|D| / avgdl)) ]
    where:
        IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )
    """
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ):
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be in [0.0, 1.0], got {b}")

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer or tokenize_words
        
        self.chunks: List[Chunk] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.corpus_size: int = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Counter] = []
        self._is_indexed: bool = False

    def index(self, chunks: List[Chunk]) -> "BM25Retriever":
        """Indexes a collection of chunks for BM25 retrieval."""
        if not chunks:
            self.chunks = []
            self.doc_len = []
            self.avgdl = 0.0
            self.corpus_size = 0
            self.doc_freqs = {}
            self.idf = {}
            self.doc_term_freqs = []
            self._is_indexed = True
            return self

        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        self.doc_term_freqs = []
        self.doc_len = []
        df: Dict[str, int] = Counter()

        for chunk in chunks:
            tokens = self.tokenizer(chunk.text)
            self.doc_len.append(len(tokens))
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            for term in term_freq.keys():
                df[term] += 1

        self.doc_freqs = dict(df)
        self.avgdl = float(sum(self.doc_len)) / max(1, self.corpus_size)

        # Calculate smoothed Robertson-Spärck Jones IDF
        # Standard: ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )
        # Adding 1 inside the log ensures IDF is strictly non-negative
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            idf_val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = max(self.epsilon, idf_val)

        self._is_indexed = True
        return self

    def score_chunk(self, query_terms: List[str], chunk_idx: int) -> float:
        """Computes BM25 score for a specific chunk index and tokenized query."""
        if not self._is_indexed or self.corpus_size == 0:
            return 0.0

        d_len = self.doc_len[chunk_idx]
        tf_dict = self.doc_term_freqs[chunk_idx]
        len_norm = 1.0 - self.b + self.b * (d_len / max(1e-6, self.avgdl))
        score = 0.0

        for term in query_terms:
            if term not in tf_dict:
                continue
            tf = tf_dict[term]
            idf = self.idf.get(term, self.epsilon)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * len_norm
            score += idf * (numerator / denominator)

        return float(score)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
        min_score: float = 0.0
    ) -> List[ScoredChunk]:
        """
        Retrieves top-K chunks ranked by BM25 score.
        Supports predicate filtering and minimum score thresholding.
        """
        if not self._is_indexed or self.corpus_size == 0 or not query or not query.strip():
            return []

        query_terms = self.tokenizer(query)
        if not query_terms:
            return []

        scored: List[ScoredChunk] = []
        for idx, chunk in enumerate(self.chunks):
            if filter_fn is not None and not filter_fn(chunk):
                continue

            s = self.score_chunk(query_terms, idx)
            if s >= min_score:
                scored.append(ScoredChunk(chunk=chunk, score=s))

        # Sort descending by score, tie-break by chunk_id
        scored.sort(key=lambda sc: (sc.score, sc.chunk.metadata.chunk_id), reverse=True)
        return scored[:top_k]

    def get_top_terms(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Returns highest IDF terms in given text."""
        terms = set(self.tokenizer(text))
        scored = [(t, self.idf.get(t, self.epsilon)) for t in terms]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

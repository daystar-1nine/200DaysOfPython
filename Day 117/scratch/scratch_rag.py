"""
Day 117 — Coding Challenge 7: Complete Mini Semantic Search Engine (Without LLM)
Builds an end-to-end self-contained semantic search index:
1. Document ingestion and sentence chunking
2. Dense TF-IDF vocabulary embedding space
3. In-memory cosine vector index
4. Top-K semantic retrieval with formatted result snippets
"""
import re
import math
from typing import List, Dict, Any, Tuple


class MiniSemanticSearchEngine:
    """
    Lightweight, dependency-free semantic retrieval engine implementing
    TF-IDF sublinear vectorization and exact cosine nearest-neighbor search.
    """
    def __init__(self, chunk_size: int = 150):
        self.chunk_size = chunk_size
        self.chunks: List[str] = []
        self.doc_names: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
        self.vectors: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]

    def add_document(self, name: str, content: str):
        """Chunks a document into short passages and adds them to the corpus."""
        words = content.split()
        stride = max(1, self.chunk_size - 30)
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                self.chunks.append(chunk)
                self.doc_names.append(name)

    def fit_index(self):
        """Builds vocabulary, calculates IDFs, and precomputes unit-normalized chunk vectors."""
        if not self.chunks:
            return

        # Build vocabulary
        vocab = {}
        doc_freq = {}
        num_chunks = len(self.chunks)

        tokenized_corpus = []
        for c in self.chunks:
            tokens = self._tokenize(c)
            tokenized_corpus.append(tokens)
            seen = set(tokens)
            for t in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                if t not in vocab:
                    vocab[t] = len(vocab)

        self.vocabulary = vocab

        # Compute IDF
        self.idf = [0.0] * len(vocab)
        for term, tid in vocab.items():
            df = doc_freq.get(term, 0)
            self.idf[tid] = math.log((num_chunks + 1) / (df + 1)) + 1.0

        # Vectorize all chunks
        self.vectors = [self._embed_tokens(tokens) for tokens in tokenized_corpus]

    def _embed_tokens(self, tokens: List[str]) -> List[float]:
        """Encodes token stream into L2-normalized TF-IDF vector."""
        vec = [0.0] * len(self.vocabulary)
        if not tokens:
            return vec

        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1

        for t, count in counts.items():
            if t in self.vocabulary:
                tid = self.vocabulary[t]
                tf = 1.0 + math.log(count)
                vec[tid] = tf * self.idf[tid]

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            vec = [x / norm for x in vec]

        return vec

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Finds top-k nearest chunks using cosine similarity."""
        query_tokens = self._tokenize(query)
        q_vec = self._embed_tokens(query_tokens)

        # Dot product with pre-normalized vectors
        scores = []
        for idx, d_vec in enumerate(self.vectors):
            score = sum(q * d for q, d in zip(q_vec, d_vec))
            scores.append((idx, score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "rank": len(results) + 1,
                "document": self.doc_names[idx],
                "score": round(float(score), 4),
                "text": self.chunks[idx]
            })

        return results


if __name__ == "__main__":
    engine = MiniSemanticSearchEngine(chunk_size=40)

    corpus = {
        "Python Notes": "Python lists are mutable ordered collections with dynamic resizing, append, and pop operations.",
        "DBMS Notes": "Relational databases provide ACID properties: Atomicity, Consistency, Isolation, and Durability via WAL.",
        "NLP Notes": "TF-IDF weighting identifies distinctive keywords by penalizing ubiquitous words using inverse document frequency.",
        "Deep Learning Notes": "Transformers rely on scaled dot-product self-attention to process entire sequences in parallel without recurrence."
    }

    for name, content in corpus.items():
        engine.add_document(name, content)

    engine.fit_index()

    test_queries = [
        "What guarantees database transaction durability?",
        "How do transformers compute attention without recurrence?",
        "Python dynamic array lists append pop"
    ]

    print("=== Challenge 7: Standalone Semantic Search Engine Demo ===")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        res = engine.search(q, top_k=2)
        for r in res:
            print(f"  [Rank {r['rank']}] Doc: {r['document']} (Score: {r['score']}) -> \"{r['text'][:70]}...\"")

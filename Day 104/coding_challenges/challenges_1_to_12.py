"""
Complete Coding Challenges 1 to 12 for Day 104: Semantic Search & Document Embeddings.
"""

import math
from typing import Any, Dict, List, Set, Tuple
import numpy as np


# -------------------------------------------------------------
# Challenge 1: Mean Document Embeddings
# -------------------------------------------------------------
def challenge_1_mean_document_embedding(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    dim: int = 64
) -> np.ndarray:
    """Implement mean document embeddings."""
    valid = [embeddings[t] for t in tokens if t in embeddings]
    if not valid:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)


# -------------------------------------------------------------
# Challenge 2: Weighted Document Embeddings
# -------------------------------------------------------------
def challenge_2_weighted_document_embedding(
    tokens: List[str],
    embeddings: Dict[str, np.ndarray],
    weights: Dict[str, float],
    dim: int = 64
) -> np.ndarray:
    """Implement weighted document embeddings."""
    valid_vecs = []
    valid_weights = []
    for t in tokens:
        if t in embeddings:
            valid_vecs.append(embeddings[t])
            valid_weights.append(float(weights.get(t, 1.0)))

    if not valid_vecs or sum(valid_weights) == 0.0:
        return np.zeros(dim, dtype=np.float32)

    v_mat = np.stack(valid_vecs, axis=0)
    w_vec = np.array(valid_weights, dtype=np.float32)[:, np.newaxis]
    return (np.sum(v_mat * w_vec, axis=0) / np.sum(w_vec)).astype(np.float32)


# -------------------------------------------------------------
# Challenge 3: Cosine Similarity
# -------------------------------------------------------------
def challenge_3_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Implement safe cosine similarity."""
    a_arr = np.asarray(a, dtype=np.float32).flatten()
    b_arr = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0 or np.isnan(denom):
        return 0.0
    return float(np.clip(np.dot(a_arr, b_arr) / denom, -1.0, 1.0))


# -------------------------------------------------------------
# Challenge 4: Top-K Ranking
# -------------------------------------------------------------
def challenge_4_top_k_ranking(
    scores: List[float],
    document_ids: List[Any],
    top_k: int = 5
) -> List[Tuple[int, Any, float]]:
    """Implement Top-K ranking. Returns [(rank, doc_id, score), ...]."""
    ranked_indices = np.argsort(-np.array(scores))[:top_k]
    return [(rank, document_ids[idx], float(scores[idx])) for rank, idx in enumerate(ranked_indices, start=1)]


# -------------------------------------------------------------
# Challenge 5: Precision@K
# -------------------------------------------------------------
def challenge_5_precision_at_k(
    retrieved_ids: List[Any],
    relevant_ids: List[Any],
    k: int
) -> float:
    """Implement Precision@K."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for did in top_k if did in rel_set)
    return float(hits / k)


# -------------------------------------------------------------
# Challenge 6: Recall@K
# -------------------------------------------------------------
def challenge_6_recall_at_k(
    retrieved_ids: List[Any],
    relevant_ids: List[Any],
    k: int
) -> float:
    """Implement Recall@K."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for did in top_k if did in rel_set)
    return float(hits / len(rel_set))


# -------------------------------------------------------------
# Challenge 7: Reciprocal Rank
# -------------------------------------------------------------
def challenge_7_reciprocal_rank(
    retrieved_ids: List[Any],
    relevant_ids: List[Any]
) -> float:
    """Implement Reciprocal Rank (RR)."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in rel_set:
            return float(1.0 / rank)
    return 0.0


# -------------------------------------------------------------
# Challenge 8: Mean Reciprocal Rank (MRR)
# -------------------------------------------------------------
def challenge_8_mrr(rr_scores: List[float]) -> float:
    """Implement Mean Reciprocal Rank (MRR)."""
    if not rr_scores:
        return 0.0
    return float(sum(rr_scores) / len(rr_scores))


# -------------------------------------------------------------
# Challenge 9: TF-IDF Search from Scratch
# -------------------------------------------------------------
class Challenge9TfidfSearch:
    """TF-IDF Search implementation from scratch."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        self.doc_ids: List[Any] = []

    def fit_index(self, corpus: List[Dict[str, Any]]):
        self.doc_ids = [d["id"] for d in corpus]
        tokenized_docs = [d["text"].lower().split() for d in corpus]
        n_docs = len(tokenized_docs)

        # Build vocabulary
        all_words = sorted(list({w for doc in tokenized_docs for w in doc}))
        self.vocab = {w: i for i, w in enumerate(all_words)}
        v_size = len(self.vocab)

        # Compute IDF
        df = np.zeros(v_size, dtype=np.float32)
        for doc in tokenized_docs:
            unique_words = set(doc)
            for w in unique_words:
                df[self.vocab[w]] += 1.0

        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # Compute TF-IDF for documents
        doc_matrix = np.zeros((n_docs, v_size), dtype=np.float32)
        for i, doc in enumerate(tokenized_docs):
            for w in doc:
                doc_matrix[i, self.vocab[w]] += 1.0
            # sublinear tf safely
            log_tf = np.zeros_like(doc_matrix[i])
            pos_mask = doc_matrix[i] > 0
            log_tf[pos_mask] = 1.0 + np.log(doc_matrix[i][pos_mask])
            doc_matrix[i] = log_tf * self.idf
            # L2 normalize
            norm = np.linalg.norm(doc_matrix[i])
            if norm > 0:
                doc_matrix[i] /= norm

        self.doc_vectors = doc_matrix

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Any, float]]:
        q_tokens = query.lower().split()
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w in q_tokens:
            if w in self.vocab:
                q_vec[self.vocab[w]] += 1.0
        log_q = np.zeros_like(q_vec)
        pos_mask = q_vec > 0
        log_q[pos_mask] = 1.0 + np.log(q_vec[pos_mask])
        q_vec = log_q * self.idf
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        scores = [challenge_3_cosine_similarity(q_vec, d_vec) for d_vec in self.doc_vectors]
        ranked = challenge_4_top_k_ranking(scores, self.doc_ids, top_k=top_k)
        return [(did, score) for _, did, score in ranked]


# -------------------------------------------------------------
# Challenge 10: Embedding Search
# -------------------------------------------------------------
class Challenge10EmbeddingSearch:
    """Embedding-based search engine."""

    def __init__(self, embeddings: Dict[str, np.ndarray], dim: int = 3):
        self.embeddings = embeddings
        self.dim = dim
        self.doc_vectors: List[np.ndarray] = []
        self.doc_ids: List[Any] = []

    def index(self, corpus: List[Dict[str, Any]]):
        self.doc_ids = [d["id"] for d in corpus]
        self.doc_vectors = [
            challenge_1_mean_document_embedding(d["text"].lower().split(), self.embeddings, dim=self.dim)
            for d in corpus
        ]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Any, float]]:
        q_vec = challenge_1_mean_document_embedding(query.lower().split(), self.embeddings, dim=self.dim)
        scores = [challenge_3_cosine_similarity(q_vec, d_vec) for d_vec in self.doc_vectors]
        ranked = challenge_4_top_k_ranking(scores, self.doc_ids, top_k=top_k)
        return [(did, score) for _, did, score in ranked]


# -------------------------------------------------------------
# Challenge 11: Compare Both Systems on 20 Queries
# -------------------------------------------------------------
def challenge_11_compare_systems(
    tfidf_engine: Challenge9TfidfSearch,
    emb_engine: Challenge10EmbeddingSearch,
    test_queries: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compare TF-IDF and Embedding retrieval across queries."""
    tfidf_rrs = []
    emb_rrs = []

    for q in test_queries:
        q_text = q["query"]
        relevant = q["relevant_ids"]

        t_res = [did for did, _ in tfidf_engine.search(q_text, top_k=5)]
        e_res = [did for did, _ in emb_engine.search(q_text, top_k=5)]

        tfidf_rrs.append(challenge_7_reciprocal_rank(t_res, relevant))
        emb_rrs.append(challenge_7_reciprocal_rank(e_res, relevant))

    return {
        "TF-IDF MRR": challenge_8_mrr(tfidf_rrs),
        "Embedding MRR": challenge_8_mrr(emb_rrs)
    }


# -------------------------------------------------------------
# Challenge 12: Hybrid Retrieval Score
# -------------------------------------------------------------
def challenge_12_hybrid_score(
    tfidf_scores: List[float],
    emb_scores: List[float],
    alpha: float = 0.5
) -> List[float]:
    """Compute normalized hybrid retrieval score.
    
    Formula: alpha * norm(tfidf) + (1 - alpha) * norm(emb)
    """
    t_arr = np.array(tfidf_scores, dtype=np.float32)
    e_arr = np.array(emb_scores, dtype=np.float32)

    def min_max(x):
        denom = np.max(x) - np.min(x)
        return np.zeros_like(x) if denom == 0 else (x - np.min(x)) / denom

    t_norm = min_max(t_arr)
    e_norm = min_max(e_arr)

    return (alpha * t_norm + (1.0 - alpha) * e_norm).tolist()


if __name__ == "__main__":
    print("Verifying all 12 coding challenges...")

    # Mini test corpus
    corpus = [
        {"id": 1, "text": "python machine learning algorithms"},
        {"id": 2, "text": "deep neural network training with backpropagation"},
        {"id": 3, "text": "vehicle maintenance and car repair diagnostics"}
    ]

    mock_embeddings = {
        "python": np.array([0.9, 0.1, 0.0]),
        "machine": np.array([0.8, 0.2, 0.0]),
        "learning": np.array([0.85, 0.15, 0.0]),
        "algorithms": np.array([0.7, 0.3, 0.0]),
        "deep": np.array([0.1, 0.9, 0.0]),
        "neural": np.array([0.1, 0.95, 0.0]),
        "network": np.array([0.15, 0.85, 0.0]),
        "training": np.array([0.2, 0.8, 0.0]),
        "backpropagation": np.array([0.05, 0.95, 0.0]),
        "vehicle": np.array([0.0, 0.1, 0.9]),
        "maintenance": np.array([0.0, 0.15, 0.85]),
        "car": np.array([0.0, 0.1, 0.92]),
        "repair": np.array([0.0, 0.2, 0.88]),
        "diagnostics": np.array([0.0, 0.25, 0.8])
    }

    # 1. Mean embedding
    v_mean = challenge_1_mean_document_embedding(["python", "machine"], mock_embeddings, dim=3)
    assert v_mean.shape == (3,)

    # 2. Weighted embedding
    v_weighted = challenge_2_weighted_document_embedding(["python", "machine"], mock_embeddings, {"python": 2.0, "machine": 1.0}, dim=3)
    assert v_weighted.shape == (3,)

    # 3. Cosine similarity
    sim = challenge_3_cosine_similarity(v_mean, v_weighted)
    assert 0.9 <= sim <= 1.0

    # 4. Top-k ranking
    top_k = challenge_4_top_k_ranking([0.1, 0.9, 0.5], [1, 2, 3], top_k=2)
    assert top_k[0][1] == 2

    # 5. Precision@K
    assert challenge_5_precision_at_k([1, 2, 3], [2, 3], 2) == 0.5

    # 6. Recall@K
    assert challenge_6_recall_at_k([1, 2, 3], [2, 3], 2) == 0.5

    # 7. RR
    assert challenge_7_reciprocal_rank([1, 2, 3], [2]) == 0.5

    # 8. MRR
    assert challenge_8_mrr([1.0, 0.5]) == 0.75

    # 9. TF-IDF Search
    t_search = Challenge9TfidfSearch()
    t_search.fit_index(corpus)
    t_res = t_search.search("python algorithms")
    assert t_res[0][0] == 1

    # 10. Embedding Search
    e_search = Challenge10EmbeddingSearch(mock_embeddings, dim=3)
    e_search.index(corpus)
    e_res = e_search.search("car repair")
    assert e_res[0][0] == 3

    # 11. Comparison on queries
    comp = challenge_11_compare_systems(
        t_search, e_search,
        [
            {"query": "python algorithms", "relevant_ids": [1]},
            {"query": "neural training", "relevant_ids": [2]},
            {"query": "car repair", "relevant_ids": [3]}
        ]
    )
    assert "TF-IDF MRR" in comp and "Embedding MRR" in comp

    # 12. Hybrid score
    hyb = challenge_12_hybrid_score([0.2, 0.8], [0.5, 0.9], alpha=0.5)
    assert len(hyb) == 2

    print("All 12 coding challenges verified successfully!")

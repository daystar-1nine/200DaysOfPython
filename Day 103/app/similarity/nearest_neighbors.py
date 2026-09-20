"""Pairwise similarity and nearest-neighbor search utilities."""
import numpy as np
import pandas as pd
from typing import List, Dict
from app.similarity.cosine import cosine_similarity
from app.data.vocabulary import Vocabulary

def compute_similarity_matrix(W: np.ndarray, words: List[str], vocab: Vocabulary) -> pd.DataFrame:
    """Computes pairwise cosine similarity matrix for a subset of words."""
    n = len(words)
    matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        idx_i = vocab.get_id(words[i])
        vec_i = W[idx_i] if idx_i is not None else np.zeros(W.shape[1])
        for j in range(n):
            idx_j = vocab.get_id(words[j])
            vec_j = W[idx_j] if idx_j is not None else np.zeros(W.shape[1])
            matrix[i, j] = cosine_similarity(vec_i, vec_j)
            
    return pd.DataFrame(matrix, index=words, columns=words)

def extract_all_nearest_neighbors(W: np.ndarray, vocab: Vocabulary, top_k: int = 3) -> pd.DataFrame:
    """Extracts top-K nearest neighbors for every word in the vocabulary."""
    records = []
    for word, idx in vocab.word2idx.items():
        vec = W[idx]
        scores = []
        for other_w, other_id in vocab.word2idx.items():
            if idx != other_id:
                sim = cosine_similarity(vec, W[other_id])
                scores.append((other_w, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_words = [f"{w} ({s:.3f})" for w, s in scores[:top_k]]
        records.append({
            "word": word,
            "top_neighbors": ", ".join(top_words)
        })
    return pd.DataFrame(records)

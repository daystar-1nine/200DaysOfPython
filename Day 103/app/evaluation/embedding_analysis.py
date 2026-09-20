"""Statistical analysis of learned embedding geometry."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.similarity.cosine import cosine_similarity
from app.data.vocabulary import Vocabulary

def analyze_embeddings(W: np.ndarray, vocab: Vocabulary) -> Dict[str, Any]:
    """Computes norm statistics and semantic pair distances."""
    norms = np.linalg.norm(W, axis=1)
    
    # Pre-defined semantic pair tests
    semantic_pairs = [
        ("cat", "dog"),
        ("cat", "kitten"),
        ("dog", "puppy"),
        ("python", "java"),
        ("cat", "python"),
        ("dog", "java"),
        ("milk", "water"),
    ]
    
    pair_similarities = {}
    for w1, w2 in semantic_pairs:
        if w1 in vocab and w2 in vocab:
            sim = cosine_similarity(W[vocab.get_id(w1)], W[vocab.get_id(w2)])
            pair_similarities[f"{w1} <-> {w2}"] = float(sim)
            
    return {
        "vocab_size": len(vocab),
        "embedding_dim": W.shape[1],
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "semantic_pairs": pair_similarities
    }

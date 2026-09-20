"""
Similarity analysis module for Day 104 Semantic Search Engine.
Analyzes vector distributions, intra-category cohesion, and inter-category separation.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np


class SimilarityAnalyzer:
    """Analyzes geometric properties and cohesion of document representations."""

    @staticmethod
    def compute_pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
        """Compute all-pairs cosine similarity matrix.
        
        Args:
            matrix: 2D numpy array of shape (N, D).
            
        Returns:
            2D symmetric matrix of shape (N, N) with values in [-1, 1].
        """
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        zero_mask = (norms == 0.0) | np.isnan(norms)
        safe_norms = np.where(zero_mask, 1.0, norms)

        normalized = matrix / safe_norms
        normalized[zero_mask.flatten()] = 0.0

        sim_matrix = np.dot(normalized, normalized.T)
        return np.clip(sim_matrix, -1.0, 1.0).astype(np.float32)

    @classmethod
    def analyze_category_cohesion(
        cls,
        doc_matrix: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate mean intra-category and inter-category similarities.
        
        Args:
            doc_matrix: 2D array of document embeddings.
            documents: List of document dicts with 'category'.
            
        Returns:
            Dictionary with category-level intra-similarity and inter-similarity.
        """
        sim_mat = cls.compute_pairwise_similarity(doc_matrix)
        n = len(documents)
        cat_indices = defaultdict(list)
        for i, d in enumerate(documents):
            cat_indices[d.get("category", "Unknown")].append(i)

        results: Dict[str, Dict[str, float]] = {}

        for cat, indices in cat_indices.items():
            intra_sims: List[float] = []
            inter_sims: List[float] = []
            idx_set = set(indices)

            for i in indices:
                for j in range(n):
                    if i == j:
                        continue
                    score = float(sim_mat[i, j])
                    if j in idx_set:
                        intra_sims.append(score)
                    else:
                        inter_sims.append(score)

            results[cat] = {
                "intra_similarity": float(np.mean(intra_sims)) if intra_sims else 0.0,
                "inter_similarity": float(np.mean(inter_sims)) if inter_sims else 0.0,
                "cohesion_ratio": (
                    float(np.mean(intra_sims) / np.mean(inter_sims))
                    if inter_sims and np.mean(inter_sims) > 0
                    else 1.0
                )
            }

        return results

"""
Embedding visualization module for Day 105: Neural NLP & Text Classification.
Projects high-dimensional learned word representations to 2D using PCA.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class EmbeddingVisualizer:
    """Extracts and projects learned embedding weights for semantic inspection."""

    @staticmethod
    def project_pca(
        embedding_matrix: np.ndarray,
        words: List[str],
        n_components: int = 2,
        seed: int = 42
    ) -> pd.DataFrame:
        """Reduce embedding vectors to 2D coordinates using PCA.
        
        Args:
            embedding_matrix: Array of shape (n_words, embedding_dim).
            words: List of word labels corresponding to rows.
            n_components: Target PCA dimensions (default 2).
            seed: Random state.
            
        Returns:
            DataFrame with 'word', 'pc1', 'pc2' coordinates.
        """
        assert len(embedding_matrix) == len(words), "Length mismatch between embeddings and words"
        if len(words) < n_components:
            raise ValueError(f"Need at least {n_components} words for PCA projection.")

        pca = PCA(n_components=n_components, random_state=seed)
        coords = pca.fit_transform(embedding_matrix)

        return pd.DataFrame({
            "word": words,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1]
        })

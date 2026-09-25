"""
Embedding projection and geometric analysis module for Day 106.
Extracts learned representations and projects them via PCA for exploratory analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from ..preprocessing.vocabulary import Vocabulary


class EmbeddingAnalyzer:
    """Extracts and analyzes word embeddings learned by the recurrent neural network."""

    @staticmethod
    def extract_weights(model: nn.Module) -> np.ndarray:
        """Extract embedding weights as a NumPy matrix of shape (vocab_size, embedding_dim)."""
        if hasattr(model, "embedding"):
            return model.embedding.weight.detach().cpu().numpy()
        raise AttributeError("Model does not have an 'embedding' attribute.")

    @classmethod
    def project_pca(
        cls,
        embedding_matrix: np.ndarray,
        vocabulary: Vocabulary,
        selected_words: Optional[List[str]] = None,
        n_components: int = 2,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Project selected word vectors to 2D using PCA.
        
        Args:
            embedding_matrix: (V, D) array.
            vocabulary: Fitted Vocabulary instance.
            selected_words: Words to project. If None, top frequent words are chosen.
            n_components: Dimensionality of projection.
            random_seed: Reproducibility seed.
            
        Returns:
            DataFrame with 'word', 'pca_1', 'pca_2', and token frequency.
        """
        if selected_words is None:
            selected_words = [
                "win", "free", "prize", "cash", "urgent", "claim", "call", "reward",
                "hello", "meeting", "thanks", "later", "lunch", "tonight", "work", "home"
            ]

        valid_words = [w for w in selected_words if w in vocabulary.word2idx]
        if len(valid_words) < 2:
            valid_words = [vocabulary.id_to_token(i) for i in range(min(15, len(vocabulary)))]

        indices = [vocabulary.token_to_id(w) for w in valid_words]
        sub_matrix = embedding_matrix[indices]

        pca = PCA(n_components=n_components, random_state=random_seed)
        coords = pca.fit_transform(sub_matrix)

        rows = []
        for i, word in enumerate(valid_words):
            rows.append({
                "word": word,
                "pca_1": float(coords[i, 0]),
                "pca_2": float(coords[i, 1]),
                "count": vocabulary.word_counts.get(word, 0),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_embeddings(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Persist 2D projected embeddings to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

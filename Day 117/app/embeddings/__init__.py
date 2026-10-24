"""
Embedding models and vector similarity computation package.
"""
from app.embeddings.similarity import (
    cosine_similarity,
    batch_cosine_similarity,
    euclidean_distance,
    dot_product_similarity
)
from app.embeddings.embedder import (
    BaseEmbedder,
    DenseTFIDFEmbedder,
    SparseTFIDFEmbedder,
    get_embedder
)

__all__ = [
    "cosine_similarity",
    "batch_cosine_similarity",
    "euclidean_distance",
    "dot_product_similarity",
    "BaseEmbedder",
    "DenseTFIDFEmbedder",
    "SparseTFIDFEmbedder",
    "get_embedder"
]

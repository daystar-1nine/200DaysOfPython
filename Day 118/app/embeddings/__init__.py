"""
Embeddings package for dense and sparse vector spaces.
"""
from app.embeddings.similarity import (
    cosine_similarity,
    cosine_similarity_pair,
    cosine_similarity_matrix,
    batch_cosine_similarity,
    normalize_vector,
    euclidean_distance,
    dot_product_similarity,
)
from app.embeddings.embedder import (
    BaseEmbedder,
    DenseTFIDFEmbedder,
    SparseTFIDFEmbedder,
    get_embedder,
)

__all__ = [
    "cosine_similarity",
    "cosine_similarity_pair",
    "cosine_similarity_matrix",
    "batch_cosine_similarity",
    "normalize_vector",
    "euclidean_distance",
    "dot_product_similarity",
    "BaseEmbedder",
    "DenseTFIDFEmbedder",
    "SparseTFIDFEmbedder",
    "get_embedder",
]

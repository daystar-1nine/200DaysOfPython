"""
Document representations module: TF-IDF, pooling strategies, and document embeddings.
"""

from .tfidf import TfidfRepresentation
from .pooling import mean_pooling, weighted_pooling
from .document_embedding import DocumentEmbedder

__all__ = ["TfidfRepresentation", "mean_pooling", "weighted_pooling", "DocumentEmbedder"]

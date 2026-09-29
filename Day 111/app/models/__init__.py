"""
BERT model architectures and classification heads.
"""
from app.models.classification_head import BertClassificationHead
from app.models.embeddings import BertEmbeddings

BertEmbeddingLayer = BertEmbeddings

__all__ = ["BertClassificationHead", "BertEmbeddings", "BertEmbeddingLayer", "BertSpamClassifier"]

from app.models.bert_classifier import BertSpamClassifier

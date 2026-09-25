"""
Baseline classifiers for Day 106: RNNs & Sequential Text Learning benchmark.
Includes:
  1. Dummy Classifier (majority class)
  2. TF-IDF + Logistic Regression
  3. TF-IDF + Linear SVM (Calibrated)
  4. Day 105 Neural Model (Trainable Embedding + Masked Average Pooling)
"""

import time
from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn


class BaselineModels:
    """Factory and wrapper for classical and previous-day baseline models."""

    @staticmethod
    def build_dummy_classifier() -> DummyClassifier:
        """Construct a DummyClassifier predicting the most frequent class."""
        return DummyClassifier(strategy="most_frequent")

    @staticmethod
    def build_tfidf_logreg(random_seed: int = 42) -> "TfidfBaselineWrapper":
        """Construct TF-IDF + Logistic Regression."""
        return TfidfBaselineWrapper(model_type="logistic_regression", random_seed=random_seed)

    @staticmethod
    def build_tfidf_svm(random_seed: int = 42) -> "TfidfBaselineWrapper":
        """Construct TF-IDF + Linear SVM with probability calibration."""
        return TfidfBaselineWrapper(model_type="linear_svm", random_seed=random_seed)


class TfidfBaselineWrapper:
    """Wrapper for TF-IDF feature extraction and Scikit-Learn linear models."""

    def __init__(self, model_type: str = "logistic_regression", random_seed: int = 42):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        self.training_time: float = 0.0

        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
        elif self.model_type == "linear_svm":
            base_svm = LinearSVC(random_state=random_seed, max_iter=2000, dual="auto")
            self.classifier = CalibratedClassifierCV(estimator=base_svm, cv=3)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, texts: List[str], labels: Union[List[int], np.ndarray]) -> "TfidfBaselineWrapper":
        start = time.perf_counter()
        X = self.vectorizer.fit_transform(texts)
        y = np.asarray(labels, dtype=int)
        self.classifier.fit(X, y)
        self.training_time = time.perf_counter() - start
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)[:, 1]

    def evaluate(self, texts: List[str], labels: Union[List[int], np.ndarray]) -> Dict[str, float]:
        y_true = np.asarray(labels, dtype=int)
        preds = self.predict(texts)
        probs = self.predict_proba(texts)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
            "training_time": float(self.training_time),
            "num_features": int(len(self.vectorizer.vocabulary_)),
        }


class Day105PoolingClassifier(nn.Module):
    """Day 105 Architecture: Trainable Embedding + Masked Global Average Pooling + Dense Head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout_rate: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T), mask: (B, T)
        embedded = self.embedding(input_ids)  # (B, T, D)
        mask_expanded = mask.unsqueeze(-1)    # (B, T, 1)
        masked_emb = embedded * mask_expanded
        summed = masked_emb.sum(dim=1)        # (B, D)
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = summed / valid_lengths       # (B, D)
        h = self.fc1(pooled)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

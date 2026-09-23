"""
Classical NLP baseline models using TF-IDF and Scikit-Learn classifiers.
Provides benchmarks for Day 105 comparison.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class TfidfBaseline:
    """Classical NLP baseline wrapper combining TF-IDF with Logistic Regression or Linear SVM."""

    def __init__(self, model_type: str = "logistic_regression", random_seed: int = 42):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        self.training_time: float = 0.0

        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
        elif self.model_type == "linear_svm":
            # CalibratedClassifierCV allows predict_proba for ROC-AUC on SVM
            base_svm = LinearSVC(random_state=random_seed, max_iter=2000, dual="auto")
            self.classifier = CalibratedClassifierCV(estimator=base_svm, cv=3)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'logistic_regression' or 'linear_svm'.")

    def fit(self, texts: List[str], labels: Union[List[int], np.ndarray]) -> "TfidfBaseline":
        """Fit TF-IDF vectorizer and classifier on training data."""
        start = time.perf_counter()
        X = self.vectorizer.fit_transform(texts)
        y = np.asarray(labels, dtype=int)
        self.classifier.fit(X, y)
        self.training_time = time.perf_counter() - start
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probability estimates for class 1 (spam)."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)[:, 1]

    def evaluate(self, texts: List[str], labels: Union[List[int], np.ndarray]) -> Dict[str, float]:
        """Compute standard classification evaluation metrics."""
        y_true = np.asarray(labels, dtype=int)
        preds = self.predict(texts)
        probs = self.predict_proba(texts)

        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "training_time": float(self.training_time),
            "num_features": int(len(self.vectorizer.vocabulary_))
        }

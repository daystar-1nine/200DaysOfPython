import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import time

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64, dense_dim: int = 32, dropout_rate: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        embedded = self.embedding(x)
        _, h_n = self.rnn(embedded)
        out = self.dropout(h_n[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64, dense_dim: int = 32, dropout_rate: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        out = self.dropout(h_n[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TfidfLogisticRegressionBaseline:
    """
    Classical NLP baseline: TF-IDF + Logistic Regression.
    """
    def __init__(self, max_features: int = 5000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])
        self.training_time_sec = 0.0

    def fit(self, texts, labels):
        start = time.time()
        self.pipeline.fit(texts, labels)
        self.training_time_sec = time.time() - start
        return self

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)[:, 1]

    def count_parameters(self) -> int:
        # Number of TF-IDF features + 1 intercept
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return clf.coef_.size + clf.intercept_.size
        return 0

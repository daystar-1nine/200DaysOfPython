"""
Baseline recurrent models (SimpleRNN, LSTM, GRU) and TF-IDF baseline for benchmarking.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class RecurrentBaselineClassifier(nn.Module):
    """
    Unified PyTorch wrapper for SimpleRNN, LSTM, and GRU baselines.
    """
    def __init__(
        self,
        cell_type: str,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        pad_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        if self.cell_type == "lstm":
            _, (h_n, _) = self.rnn(embeds)
            last_hidden = h_n[-1]
        else:
            _, h_n = self.rnn(embeds)
            last_hidden = h_n[-1]

        probs = self.classifier(last_hidden)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TfidfBaseline:
    """TF-IDF + Logistic Regression baseline."""
    def __init__(self, max_features: int = 1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(class_weight="balanced", random_state=42)

    def fit(self, texts_train, y_train):
        x_train = self.vectorizer.fit_transform(texts_train)
        self.model.fit(x_train, y_train)

    def predict_proba(self, texts):
        x = self.vectorizer.transform(texts)
        return self.model.predict_proba(x)[:, 1]

    def count_parameters(self) -> int:
        return int(len(self.vectorizer.vocabulary_) + 1)

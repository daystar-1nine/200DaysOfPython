"""
Tests for baseline models in Day 106: RNNs & Sequential Text Learning.
"""

import numpy as np
import pytest
import torch
from app.models.baseline import BaselineModels, TfidfBaselineWrapper, Day105PoolingClassifier


def test_dummy_classifier():
    dummy = BaselineModels.build_dummy_classifier()
    X = ["spam text", "ham text", "ham text 2"]
    y = [1, 0, 0]
    dummy.fit(X, y)
    preds = dummy.predict(["any message", "another message"])
    assert len(preds) == 2
    assert all(p == 0 for p in preds)  # Majority class is 0


def test_tfidf_logreg_fit_predict():
    train_texts = [
        "urgent win free prize claim now",
        "hello are we meeting for coffee today",
        "lottery cash payout winner claim immediately",
        "sounds good see you at noon",
    ]
    train_y = [1, 0, 1, 0]

    model = BaselineModels.build_tfidf_logreg()
    model.fit(train_texts, train_y)

    test_texts = ["urgent lottery winner claim cash", "see you at coffee"]
    preds = model.predict(test_texts)
    probs = model.predict_proba(test_texts)

    assert preds[0] == 1
    assert preds[1] == 0
    assert probs[0] > 0.5
    assert probs[1] < 0.5


def test_tfidf_svm_calibrated_probabilities():
    train_texts = [
        "urgent win free prize cash lottery winner",
        "hello friend are you free for lunch",
        "claim your millions now free bonus",
        "see you at five for coffee",
        "urgent cash prize winner claim jackpot",
        "let us review the code tonight",
    ]
    train_y = [1, 0, 1, 0, 1, 0]

    model = BaselineModels.build_tfidf_svm()
    model.fit(train_texts, train_y)

    test_texts = ["urgent jackpot winner cash prize", "see you tonight friend"]
    probs = model.predict_proba(test_texts)

    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert probs[0] > probs[1]


def test_day105_pooling_classifier():
    model = Day105PoolingClassifier(vocab_size=50, embedding_dim=16, hidden_dim=16)
    x = torch.randint(0, 50, (3, 8), dtype=torch.long)
    mask = torch.ones((3, 8), dtype=torch.float32)

    logits = model(x, mask)
    assert logits.shape == (3,)
    assert model.count_parameters() > 500

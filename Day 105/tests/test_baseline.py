"""
Tests for baseline models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
"""

import numpy as np
import pytest
from app.models.baseline import TfidfBaseline


def test_baseline_logreg_fit_predict():
    train_texts = [
        "win free cash now call prize",
        "hello friend how are you",
        "claim your reward win immediately",
        "are we meeting for coffee today"
    ]
    train_labels = [1, 0, 1, 0]
    
    test_texts = [
        "win cash prize",
        "coffee meeting today"
    ]
    
    clf = TfidfBaseline(model_type="logistic_regression")
    clf.fit(train_texts, train_labels)
    
    preds = clf.predict(test_texts)
    probs = clf.predict_proba(test_texts)
    
    assert len(preds) == 2
    assert preds[0] == 1
    assert preds[1] == 0
    assert probs[0] > 0.5
    assert probs[1] < 0.5


def test_baseline_svm_calibrated_probabilities():
    train_texts = [
        "urgent lottery winner claim your jackpot prize cash reward win win",
        "hey are you free tonight to hang out friend coffee lunch",
        "win free entry to our contest now prize cash cash claim",
        "sounds good see you at five hello friend how are you",
        "claim your millions today winner cash prize win reward",
        "let us grab lunch tomorrow noon friend meeting coffee"
    ]
    train_labels = [1, 0, 1, 0, 1, 0]
    
    test_texts = [
        "urgent jackpot lottery winner win prize cash reward",
        "see you tonight friend lunch coffee"
    ]
    
    clf = TfidfBaseline(model_type="linear_svm")
    clf.fit(train_texts, train_labels)
    
    preds = clf.predict(test_texts)
    probs = clf.predict_proba(test_texts)
    
    assert len(preds) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert probs[0] > probs[1]


def test_baseline_evaluate_metrics():
    train_texts = ["spam prize", "ham message", "spam bonus", "ham chat"]
    train_labels = [1, 0, 1, 0]
    clf = TfidfBaseline(model_type="logistic_regression")
    clf.fit(train_texts, train_labels)
    metrics = clf.evaluate(train_texts, train_labels)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] >= 0.5

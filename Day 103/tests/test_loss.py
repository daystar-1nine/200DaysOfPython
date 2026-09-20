"""Tests for negative sampling loss calculation."""
import pytest
import numpy as np
from app.training.loss import compute_loss

def test_loss_perfect_prediction():
    # pos_prob ~ 1.0, neg_probs ~ 0.0 -> loss ~ 0
    loss = compute_loss(pos_prob=0.9999, neg_probs=np.array([0.0001, 0.0001]))
    assert loss < 0.01

def test_loss_poor_prediction():
    # pos_prob ~ 0.0, neg_probs ~ 1.0 -> loss should be very high
    loss = compute_loss(pos_prob=0.001, neg_probs=np.array([0.999]))
    assert loss > 5.0

def test_loss_non_negative():
    loss = compute_loss(pos_prob=0.5, neg_probs=np.array([0.5, 0.5]))
    assert loss >= 0.0

def test_loss_monotonic():
    l1 = compute_loss(0.8, np.array([0.2]))
    l2 = compute_loss(0.4, np.array([0.2]))
    assert l2 > l1

def test_loss_multiple_negatives():
    l_one = compute_loss(0.8, np.array([0.3]))
    l_two = compute_loss(0.8, np.array([0.3, 0.3]))
    assert l_two > l_one

def test_loss_handles_extreme_probabilities():
    loss = compute_loss(pos_prob=1.0, neg_probs=np.array([0.0]))
    assert loss >= 0.0
    assert not np.isnan(loss)

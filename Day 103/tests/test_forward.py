"""Tests for forward activation and sigmoid."""
import pytest
import numpy as np
from app.training.forward import sigmoid, forward_step

def test_sigmoid_zero():
    assert pytest.approx(sigmoid(0.0)) == 0.5

def test_sigmoid_large_positive():
    assert pytest.approx(sigmoid(100.0), abs=1e-4) == 1.0

def test_sigmoid_large_negative():
    assert pytest.approx(sigmoid(-100.0), abs=1e-4) == 0.0

def test_sigmoid_array():
    arr = np.array([-1.0, 0.0, 1.0])
    res = sigmoid(arr)
    assert res.shape == (3,)
    assert res[0] < res[1] < res[2]

def test_forward_step_shapes():
    target = np.array([1.0, 0.0])
    context = np.array([1.0, 0.0])
    negs = np.array([[0.0, 1.0], [-1.0, 0.0]])
    pos_p, neg_p = forward_step(target, context, negs)
    assert isinstance(pos_p, float)
    assert neg_p.shape == (2,)
    assert 0.0 <= pos_p <= 1.0
    assert np.all((neg_p >= 0.0) & (neg_p <= 1.0))

def test_forward_step_values():
    target = np.array([1.0, 0.0])
    context = np.array([1.0, 0.0]) # dot=1.0 -> sigmoid(1.0) ~ 0.731
    negs = np.array([[0.0, 0.0]])   # dot=0.0 -> sigmoid(0.0) = 0.5
    pos_p, neg_p = forward_step(target, context, negs)
    assert pytest.approx(pos_p, abs=1e-3) == 0.731
    assert pytest.approx(neg_p[0], abs=1e-3) == 0.5


import pytest
import numpy as np
from app.metrics.iou import calculate_iou_mask
from app.metrics.dice import dice_score

def test_iou_perfect():
    mask = [[0, 1], [1, 0]]
    assert calculate_iou_mask(mask, mask) == 1.0

def test_iou_zero():
    t = [[1, 1], [1, 1]]
    p = [[0, 0], [0, 0]]
    assert calculate_iou_mask(t, p) == 0.0

def test_iou_partial():
    t = [[1, 0], [0, 0]]
    p = [[1, 1], [0, 0]]
    assert calculate_iou_mask(t, p) == 0.5

def test_dice_perfect():
    mask = np.array([[0, 1], [1, 0]])
    assert dice_score(mask, mask) > 0.99
    
def test_dice_zero():
    t = np.array([[1, 0], [0, 0]])
    p = np.array([[0, 1], [1, 1]])
    assert dice_score(t, p) < 0.01

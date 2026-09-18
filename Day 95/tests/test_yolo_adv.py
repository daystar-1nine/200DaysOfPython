
import pytest
from app.data.validator import validate_bbox
from app.evaluation.metrics import evaluate_thresholds
from app.detection.iou import calculate_iou

def test_validate_bbox_valid():
    assert validate_bbox(0.5, 0.5, 0.2, 0.2) == True

def test_validate_bbox_invalid_x():
    assert validate_bbox(1.5, 0.5, 0.2, 0.2) == False

def test_validate_bbox_invalid_w():
    assert validate_bbox(0.5, 0.5, -0.2, 0.2) == False

def test_iou_perfect():
    assert calculate_iou([10,10,20,20], [10,10,20,20]) == 1.0

def test_threshold_sweep():
    res = evaluate_thresholds([], [], [0.1, 0.5, 0.9])
    assert len(res) == 3

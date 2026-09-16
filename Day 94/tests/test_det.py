
import pytest
from app.detection.iou import calculate_iou
from app.detection.nms import nms

def test_iou_perfect_overlap():
    box_a = [100, 100, 300, 300]
    box_b = [100, 100, 300, 300]
    assert calculate_iou(box_a, box_b) == 1.0

def test_iou_no_overlap():
    box_a = [100, 100, 200, 200]
    box_b = [300, 300, 400, 400]
    assert calculate_iou(box_a, box_b) == 0.0

def test_nms_filters_overlap():
    boxes = [
        [100, 100, 200, 200],
        [110, 110, 210, 210], # High overlap
        [300, 300, 400, 400]  # No overlap
    ]
    scores = [0.9, 0.8, 0.85]
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    assert len(keep) == 2
    assert keep == [0, 2] # Keeps highest score 0.9, and the non-overlapping 0.85

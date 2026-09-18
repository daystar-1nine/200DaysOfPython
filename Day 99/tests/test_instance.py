
import pytest
import numpy as np
import cv2
from app.metrics.iou import mask_iou
from app.shape.features import extract_shape_features
from app.morphology.clean import clean_mask

def test_mask_iou():
    m1 = np.array([[1, 0], [0, 0]])
    m2 = np.array([[1, 1], [0, 0]])
    assert mask_iou(m1, m2) == 0.5

def test_shape_features_circle():
    mask = np.zeros((100, 100))
    cv2.circle(mask, (50, 50), 10, 1, -1)
    
    f = extract_shape_features(mask)
    assert f['area'] > 200
    assert f['aspect_ratio'] == 1.0 # Bounding rect for circle is square
    assert f['cx'] == 50.0
    assert f['cy'] == 50.0

def test_morphology_cleaning():
    mask = np.zeros((10, 10))
    mask[2:5, 2:5] = 1
    mask[8, 8] = 1 # Noise pixel
    
    cleaned = clean_mask(mask, operation='opening', kernel_size=(3,3))
    assert cleaned[8, 8] == 0 # Noise removed
    assert cleaned[3, 3] == 1 # Core shape retained

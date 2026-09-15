import numpy as np
from app.evaluation.confusion_matrix import compute_confusion_matrix

def test_binary_confusion_matrix_keys():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["TP"] == 1
    assert cm["FN"] == 1
    assert cm["TN"] == 1
    assert cm["FP"] == 1

def test_confusion_matrix_sum():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["matrix"].sum() == len(y_true)

def test_confusion_matrix_dataframe():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["dataframe"].shape == (2, 2)

import numpy as np
from app.evaluation.confusion_matrix import compute_confusion_matrix

def test_binary_confusion_matrix():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm["TP"] == 1
    assert cm["FN"] == 1
    assert cm["TN"] == 1
    assert cm["FP"] == 1

def test_multiclass_confusion_matrix_shape():
    labels = ["Low", "Medium", "High"]
    y_true = np.array(["Low", "Medium", "High", "Low"])
    y_pred = np.array(["Low", "Low", "High", "Medium"])
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    assert cm["matrix"].shape == (3, 3)
    assert not cm["confused_pairs"].empty

def test_confusion_matrix_sum_matches_n():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm["matrix"].sum() == len(y_true)

def test_normalized_confusion_matrix_row_sums():
    labels = ["Low", "Medium", "High"]
    y_true = np.array(["Low", "Medium", "High", "Low", "Medium", "High"])
    y_pred = np.array(["Low", "Low", "High", "Medium", "Medium", "Medium"])
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm["normalized_matrix"].sum(axis=1)
    assert np.allclose(row_sums, 1.0)

def test_confused_pairs_columns():
    labels = ["Low", "Medium", "High"]
    y_true = np.array(["Low", "Medium", "High", "Low"])
    y_pred = np.array(["Medium", "Low", "High", "Low"])
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    assert set(cm["confused_pairs"].columns) == {"True_Class", "Predicted_Class", "Count", "Rate"}

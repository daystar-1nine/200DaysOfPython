
import numpy as np

def calculate_iou_mask(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true) > 0.5
    y_pred = np.array(y_pred) > threshold
    
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

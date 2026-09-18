
import numpy as np

def mask_iou(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    intersection = (pred & true).sum()
    union = (pred | true).sum()
    if union == 0: return 1.0
    return intersection / union

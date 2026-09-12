# Task 4: Threshold Experiment
# Observe how changing threshold affects predictions

import numpy as np
from sklearn.metrics import precision_score, recall_score

def experiment_thresholds(y_true, probs, thresholds):
    for t in thresholds:
        y_pred = (np.array(probs) >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        print(f"Threshold: {t:.2f} -> Precision: {p:.2f}, Recall: {r:.2f}")

if __name__ == "__main__":
    y_t = [0, 1, 1, 0, 0, 1]
    probs = [0.1, 0.8, 0.4, 0.2, 0.6, 0.9]
    experiment_thresholds(y_t, probs, [0.3, 0.5, 0.7])

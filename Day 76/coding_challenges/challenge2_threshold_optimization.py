# Challenge 2: Threshold Optimization
# Write a function that finds the optimal threshold to maximize F1 score

import numpy as np
from sklearn.metrics import f1_score

def optimize_threshold(y_true, probabilities):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_t = 0.5
    
    for t in thresholds:
        preds = (np.array(probabilities) >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    return best_t, best_f1

if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    probs = [0.1, 0.4, 0.8, 0.2, 0.9, 0.3, 0.6, 0.7, 0.45, 0.1]
    
    t, f1 = optimize_threshold(y_true, probs)
    print(f"Optimal Threshold: {t:.2f} with F1: {f1:.2f}")

# Task 3: Confusion Matrix Explain
# Generate and explain a confusion matrix

from sklearn.metrics import confusion_matrix
import numpy as np

def explain_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(cm)
    print(f"True Positives (Correctly identified positive class): {tp}")
    print(f"True Negatives (Correctly identified negative class): {tn}")
    print(f"False Positives (Incorrectly identified as positive): {fp}")
    print(f"False Negatives (Incorrectly identified as negative): {fn}")

if __name__ == "__main__":
    y_t = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
    y_p = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    explain_cm(y_t, y_p)

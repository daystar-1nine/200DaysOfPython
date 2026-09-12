# Task 6: ROC Curve
# Compute and print ROC curve data points

from sklearn.metrics import roc_curve, roc_auc_score

def compute_roc():
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    probs = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7]
    
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    
    print(f"AUC: {auc:.2f}")
    for f, t, th in zip(fpr, tpr, thresholds):
        print(f"Threshold: {th:.2f} -> FPR: {f:.2f}, TPR: {t:.2f}")

if __name__ == "__main__":
    compute_roc()

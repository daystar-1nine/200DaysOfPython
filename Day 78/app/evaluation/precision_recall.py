from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_pr(y_true, y_prob) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": ap
    }

# Task 2: Metrics Calculation
# Calculate Precision, Recall, and F1 given true and predicted labels

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return p, r, f1

if __name__ == "__main__":
    y_t = [0, 1, 1, 0, 1, 1, 0]
    y_p = [0, 1, 0, 0, 1, 1, 1]
    p, r, f1 = calculate_metrics(y_t, y_p)
    print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

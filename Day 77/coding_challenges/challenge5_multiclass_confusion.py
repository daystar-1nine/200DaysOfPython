"""Challenge 5: Multi-Class Confusion Matrix Deep-Dive."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def solve():
    labels = ["Low", "Medium", "High"]
    X, y_num = make_classification(n_samples=1200, n_features=8, n_informative=4, n_classes=3, weights=[0.55, 0.30, 0.15], random_state=42)
    y = np.array([labels[idx] for idx in y_num])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    preds = model.predict(X_test)
    
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Multi-Class Confusion Matrix:")
    print(cm_df)
    
    # Calculate per-class metrics
    precs = precision_score(y_test, preds, labels=labels, average=None)
    recs = recall_score(y_test, preds, labels=labels, average=None)
    
    lowest_rec_cls = labels[int(np.argmin(recs))]
    lowest_prec_cls = labels[int(np.argmin(precs))]
    
    # Most confused pair
    confused = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                confused.append((labels[i], labels[j], cm[i, j]))
    most_confused = sorted(confused, key=lambda x: x[2], reverse=True)[0]
    
    print(f"\nMost Confused Pair: True '{most_confused[0]}' misclassified as '{most_confused[1]}' ({most_confused[2]} times)")
    print(f"Class with Lowest Recall:    '{lowest_rec_cls}' ({min(recs):.4f})")
    print(f"Class with Lowest Precision: '{lowest_prec_cls}' ({min(precs):.4f})")

if __name__ == "__main__":
    solve()

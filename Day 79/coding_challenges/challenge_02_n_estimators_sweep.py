"""Challenge 2: n_estimators Sweep & Out-of-Bag Error Plateau Analysis."""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def sweep_n_estimators():
    X, y = make_classification(n_samples=800, n_features=12, random_state=42)
    tree_counts = [10, 30, 50, 100, 150]
    scores = {}
    for n in tree_counts:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
        rf.fit(X, y)
        scores[n] = rf.oob_score_
        print(f"Trees: {n:3d} -> OOB Accuracy: {rf.oob_score_:.4f}")
    return scores

if __name__ == '__main__':
    res = sweep_n_estimators()
    assert res[100] > res[10]
    print("Challenge 2 passed!")

"""Challenge 5: Hyperparameter Tuning of Random Forest using GridSearchCV."""
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_random_forest():
    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    param_grid = {
        'max_depth': [4, 8],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best Accuracy: {grid.best_score_:.4f}")
    return grid.best_score_

if __name__ == '__main__':
    score = tune_random_forest()
    assert score > 0.85
    print("Challenge 5 passed!")

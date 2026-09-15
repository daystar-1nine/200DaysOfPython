import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def tune_decision_tree(pipeline, X_train, y_train, param_grid=None, n_splits=5, scoring="f1", random_state=42):
    if param_grid is None:
        param_grid = {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 4, 5, 6, 8, 10],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10]
        }
        
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    
    results_df = pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, results_df

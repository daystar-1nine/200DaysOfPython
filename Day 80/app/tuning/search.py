from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def run_model_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """Run systematic GridSearchCV on a model pipeline."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)

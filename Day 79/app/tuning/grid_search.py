from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def run_rf_grid_search(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = 'f1',
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], float]:
    """Run GridSearchCV on Random Forest pipeline."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, float(grid_search.best_score_)

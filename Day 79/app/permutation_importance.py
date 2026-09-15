import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = 'f1'
) -> pd.DataFrame:
    """Compute permutation feature importance on unseen test set."""
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )
    
    df_perm = pd.DataFrame({
        'Feature': list(X_test.columns),
        'Mean_Importance': result.importances_mean,
        'Std_Importance': result.importances_std
    }).sort_values(by='Mean_Importance', ascending=False).reset_index(drop=True)
    
    return df_perm

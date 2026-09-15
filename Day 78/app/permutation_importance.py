import pandas as pd
from sklearn.inspection import permutation_importance

def compute_permutation_importance(pipeline, X_test, y_test, feature_names: list, scoring="f1", n_repeats=10, random_state=42) -> pd.DataFrame:
    r = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )
    
    # Preprocessor may output more or fewer features than raw test df
    # Map back to raw features if scoring on pipeline input
    feat_names = list(X_test.columns)
    df = pd.DataFrame({
        "Feature": feat_names,
        "Importance_Mean": r.importances_mean,
        "Importance_Std": r.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)
    return df

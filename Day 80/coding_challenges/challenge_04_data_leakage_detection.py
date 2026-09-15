"""Challenge 4: Data Leakage Detection and Audit Checklist."""
from typing import List, Dict
import pandas as pd
from sklearn.pipeline import Pipeline

def audit_data_leakage(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str
) -> Dict[str, bool]:
    """Audit pipeline and data splits against common data leakage vulnerabilities."""
    checks = {}
    
    # 1. Target not in feature columns
    checks['target_not_in_features'] = (target_column not in train_df.columns or target_column not in test_df.columns)
    
    # 2. Train and test indices are completely disjoint
    train_idx = set(train_df.index)
    test_idx = set(test_df.index)
    checks['disjoint_train_test_indices'] = len(train_idx.intersection(test_idx)) == 0
    
    # 3. Pipeline contains encapsulation (preprocessing steps before classifier)
    step_names = [name for name, _ in pipeline.steps]
    checks['has_preprocessor_step'] = 'preprocessor' in step_names or len(step_names) > 1
    
    # 4. No future leakage indicators
    leakage_keywords = ['cancellation_date', 'churn_date', 'exit_reason', 'end_date']
    found_leakage_cols = [col for col in train_df.columns if any(k in col.lower() for k in leakage_keywords)]
    checks['no_forbidden_future_columns'] = len(found_leakage_cols) == 0
    
    return checks

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([('preprocessor', StandardScaler()), ('classifier', LogisticRegression())])
    tr = pd.DataFrame({'feature_a': [1, 2, 3], 'feature_b': [4, 5, 6]}, index=[0, 1, 2])
    te = pd.DataFrame({'feature_a': [7, 8], 'feature_b': [9, 10]}, index=[3, 4])
    
    audit = audit_data_leakage(pipe, tr, te, target_column='Churn')
    print("Leakage Audit Checks:", audit)
    assert all(audit.values())
    print("Challenge 4 passed!")

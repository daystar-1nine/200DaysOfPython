from typing import Dict, Any

def get_logistic_param_grid() -> Dict[str, Any]:
    return {
        'classifier__C': [0.1, 1.0, 10.0]
    }

def get_decision_tree_param_grid() -> Dict[str, Any]:
    return {
        'classifier__max_depth': [4, 6, 8],
        'classifier__min_samples_split': [5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

def get_random_forest_param_grid() -> Dict[str, Any]:
    return {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [8, 12],
        'classifier__min_samples_split': [2, 5]
    }

from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = 'sqrt',
    criterion: str = 'gini',
    bootstrap: bool = True,
    oob_score: bool = True,
    random_state: int = 42,
    class_weight: Optional[str] = 'balanced',
    n_jobs: int = -1
) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

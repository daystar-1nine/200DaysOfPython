from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Optional

def create_decision_tree_pipeline(
    preprocessor: ColumnTransformer,
    max_depth: Optional[int] = 6,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    criterion: str = 'gini',
    class_weight: Optional[str] = 'balanced',
    random_state: int = 42
) -> Pipeline:
    """Create Decision Tree pipeline."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

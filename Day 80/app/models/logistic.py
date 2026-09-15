from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Optional

def create_logistic_pipeline(
    preprocessor: ColumnTransformer,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs',
    class_weight: Optional[str] = 'balanced',
    random_state: int = 42
) -> Pipeline:
    """Create Logistic Regression pipeline."""
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

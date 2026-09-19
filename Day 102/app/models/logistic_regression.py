"""Logistic Regression model factory."""
from sklearn.linear_model import LogisticRegression

def get_logistic_model(C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> LogisticRegression:
    """Returns configured LogisticRegression."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state
    )

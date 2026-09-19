"""Baseline classification models."""
from sklearn.dummy import DummyClassifier

def get_baseline_model(strategy: str = "most_frequent", random_state: int = 42) -> DummyClassifier:
    """Returns a DummyClassifier baseline."""
    return DummyClassifier(strategy=strategy, random_state=random_state)

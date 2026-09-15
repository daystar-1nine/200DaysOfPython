from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_baseline_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create a DummyClassifier baseline (most frequent class)."""
    model = DummyClassifier(strategy="most_frequent")
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

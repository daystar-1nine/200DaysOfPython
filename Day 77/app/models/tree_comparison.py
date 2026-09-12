from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_random_forest_model(preprocessor, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])
    return pipeline

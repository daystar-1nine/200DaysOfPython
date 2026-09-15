from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_logistic_baseline_model(preprocessor, random_state: int = 42) -> Pipeline:
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

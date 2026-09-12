from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

def build_balanced_logistic_model(preprocessor, multi_class: str = "auto", C: float = 1.0, random_state: int = 42) -> Pipeline:
    base_clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )
    if multi_class == "ovr":
        clf = OneVsRestClassifier(base_clf)
    else:
        clf = base_clf
        
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

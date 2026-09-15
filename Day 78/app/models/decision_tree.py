from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def build_decision_tree_model(
    preprocessor,
    criterion: str = "gini",
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features=None,
    class_weight=None,
    random_state: int = 42
) -> Pipeline:
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

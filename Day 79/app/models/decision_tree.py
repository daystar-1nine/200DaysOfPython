from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_decision_tree_pipeline(
    preprocessor: ColumnTransformer, 
    max_depth: int = 6, 
    min_samples_split: int = 10, 
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> Pipeline:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight='balanced'
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_logistic_pipeline(preprocessor: ColumnTransformer, random_state: int = 42) -> Pipeline:
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from app.preprocessing import build_preprocessor
from app.config import AppConfig

class LassoRegressionModel:
    def __init__(self, alpha=1.0):
        config = AppConfig()
        self.pipeline = Pipeline([
            ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
            ('regressor', Lasso(alpha=alpha, max_iter=5000, random_state=42))
        ])
        
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
        
    def predict(self, X):
        return self.pipeline.predict(X)
        
    def score(self, X, y):
        return self.pipeline.score(X, y)

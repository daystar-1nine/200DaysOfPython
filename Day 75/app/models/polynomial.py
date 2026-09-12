from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from app.preprocessing import build_preprocessor
from app.config import AppConfig

class PolynomialRegressionModel:
    def __init__(self, degree=2, include_bias=False):
        config = AppConfig()
        self.pipeline = Pipeline([
            ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
            ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ('regressor', LinearRegression())
        ])
        
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
        
    def predict(self, X):
        return self.pipeline.predict(X)
        
    def score(self, X, y):
        return self.pipeline.score(X, y)

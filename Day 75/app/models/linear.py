from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from app.preprocessing import build_preprocessor
from app.config import AppConfig

class LinearRegressionModel:
    def __init__(self):
        config = AppConfig()
        self.pipeline = Pipeline([
            ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
            ('regressor', LinearRegression())
        ])
        
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
        
    def predict(self, X):
        return self.pipeline.predict(X)
        
    def score(self, X, y):
        return self.pipeline.score(X, y)
        
    @property
    def coef_(self):
        return self.pipeline.named_steps['regressor'].coef_
        
    @property
    def intercept_(self):
        return self.pipeline.named_steps['regressor'].intercept_

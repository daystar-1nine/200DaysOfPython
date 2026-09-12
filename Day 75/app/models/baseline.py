import numpy as np

class BaselineRegressor:
    def __init__(self):
        self.mean_ = None
        
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
        
    def predict(self, X):
        return np.full((X.shape[0],), self.mean_)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v if v != 0 else 0.0

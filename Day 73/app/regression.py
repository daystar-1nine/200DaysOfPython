"""
Day 73 - Simple Linear Regression Wrapper
Wraps scikit-learn LinearRegression and provides parameter extraction and equation strings.
"""
from typing import Union, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression

class SimpleLinearRegressor:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.is_fitted: bool = False
        
    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X_arr = np.asarray(X, dtype=float).reshape(-1, 1)
        y_arr = np.asarray(y, dtype=float).ravel()
        
        self.model.fit(X_arr, y_arr)
        self.slope = float(self.model.coef_[0])
        self.intercept = float(self.model.intercept_)
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        X_arr = np.asarray(X, dtype=float).reshape(-1, 1)
        return self.model.predict(X_arr)
        
    def get_equation(self, feature_name: str = "Advertising_Spend", target_name: str = "Sales") -> str:
        sign = "+" if self.slope >= 0 else "-"
        return f"{target_name} = {self.intercept:,.2f} {sign} {abs(self.slope):.4f} * {feature_name}"
        
    def get_params(self) -> Dict[str, float]:
        return {
            "slope": self.slope,
            "intercept": self.intercept
        }

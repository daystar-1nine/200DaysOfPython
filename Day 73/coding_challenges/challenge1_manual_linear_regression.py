"""
Day 73 - Challenge 1: Full Object-Oriented Manual Linear Regression Engine
Implements a complete custom Scikit-Learn API compliant regressor and benchmarks against LinearRegression.
"""
from typing import Union, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression

class ManualLinearRegression:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.slope_: float = 0.0
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False
        
    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        X_arr = np.asarray(X, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have identical lengths.")
        if len(X_arr) < 2:
            raise ValueError("At least 2 samples required to fit linear regression.")
            
        mean_x = np.mean(X_arr)
        mean_y = np.mean(y_arr)
        
        denom = np.sum((X_arr - mean_x) ** 2)
        if np.isclose(denom, 0.0):
            raise ValueError("Predictor X has zero variance (cannot fit slope).")
            
        numer = np.sum((X_arr - mean_x) * (y_arr - mean_y))
        self.slope_ = float(numer / denom)
        
        if self.fit_intercept:
            self.intercept_ = float(mean_y - self.slope_ * mean_x)
        else:
            self.intercept_ = 0.0
            
        self.is_fitted_ = True
        return self
        
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float).ravel()
        return self.intercept_ + self.slope_ * X_arr
        
    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        y_arr = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))
        
    def summary(self) -> Dict[str, Any]:
        return {
            "slope": round(self.slope_, 4),
            "intercept": round(self.intercept_, 4),
            "equation": f"y_hat = {self.intercept_:.2f} + {self.slope_:.4f} * X"
        }

def main():
    np.random.seed(42)
    n = 150
    X = np.random.uniform(10.0, 100.0, size=n)
    y = 45.0 + 2.35 * X + np.random.normal(0, 12.0, size=n)
    
    # 1. Custom regressor
    custom_model = ManualLinearRegression(fit_intercept=True)
    custom_model.fit(X, y)
    y_pred_custom = custom_model.predict(X)
    r2_custom = custom_model.score(X, y)
    
    # 2. Scikit-learn regressor
    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X.reshape(-1, 1), y)
    y_pred_sk = sk_model.predict(X.reshape(-1, 1))
    r2_sk = sk_model.score(X.reshape(-1, 1), y)
    
    print("=" * 65)
    print("DAY 73 - CHALLENGE 1: OBJECT-ORIENTED MANUAL LINEAR REGRESSION")
    print("=" * 65)
    print(f"{'Parameter':<18} | {'Manual Model':<20} | {'Scikit-Learn':<20}")
    print("-" * 65)
    print(f"{'Slope (b1)':<18} | {custom_model.slope_:<20.6f} | {sk_model.coef_[0]:<20.6f}")
    print(f"{'Intercept (b0)':<18} | {custom_model.intercept_:<20.6f} | {sk_model.intercept_:<20.6f}")
    print(f"{'R2 Score':<18} | {r2_custom:<20.6f} | {r2_sk:<20.6f}")
    print("-" * 65)
    print(f"Fitted Equation: {custom_model.summary()['equation']}")
    
    assert np.isclose(custom_model.slope_, sk_model.coef_[0]), "Slope mismatch!"
    assert np.isclose(custom_model.intercept_, sk_model.intercept_), "Intercept mismatch!"
    assert np.isclose(r2_custom, r2_sk), "R2 mismatch!"
    assert np.allclose(y_pred_custom, y_pred_sk), "Predictions mismatch!"
    print("Verification: PASSED (Exact numerical parity achieved)")

if __name__ == "__main__":
    main()

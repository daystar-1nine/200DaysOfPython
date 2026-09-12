"""
Day 73 - Task 2: Prediction
Trains a linear regression model and generates predictions for single and batch input points.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y_train = np.array([2, 4, 5, 8, 10], dtype=float)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    test_points = np.array([[6], [7], [10]], dtype=float)
    predictions = model.predict(test_points)
    
    print("=" * 60)
    print("DAY 73 - TASK 2: LINEAR REGRESSION PREDICTIONS")
    print("=" * 60)
    print(f"Learned Equation: y_hat = {model.intercept_:.2f} + {model.coef_[0]:.2f} * X")
    print("-" * 60)
    for x_val, pred in zip(test_points.flatten(), predictions):
        print(f"  Input X = {x_val:>2.0f} --> Predicted y_hat = {pred:.2f}")
        
    # Extrapolation note
    pred_large = model.predict([[35]])[0]
    print(f"\n  Extrapolated Input X = 35 --> Predicted y_hat = {pred_large:.2f}")
    print("  [Advisory] X = 35 is far beyond training support [1, 5] (Extrapolation warning).")
    
    assert len(predictions) == 3
    assert predictions[0] < predictions[1] < predictions[2]
    print("Verification: PASSED")

if __name__ == "__main__":
    main()

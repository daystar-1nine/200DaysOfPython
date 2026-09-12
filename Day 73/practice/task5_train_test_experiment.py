"""
Day 73 - Task 5: Train/Test Split Experiment
Generates 200 synthetic observations, trains on 80%, and benchmarks generalization on 20% test data.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    np.random.seed(42)
    n = 200
    X = np.random.uniform(10.0, 100.0, size=(n, 1))
    noise = np.random.normal(0, 15.0, size=(n, 1))
    y = (25.0 + 1.85 * X + noise).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Train evaluation
    y_pred_train = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    # Test evaluation
    y_pred_test = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("=" * 60)
    print("DAY 73 - TASK 5: TRAIN / TEST GENERALIZATION BENCHMARK")
    print("=" * 60)
    print(f"Total Observations: {n} (Train = {len(X_train)}, Test = {len(X_test)})")
    print(f"Fitted Line: y_hat = {model.intercept_:.2f} + {model.coef_[0]:.2f} * X")
    print("-" * 60)
    print(f"{'Metric':<10} | {'Training Set (80%)':<20} | {'Test Set (20%)':<20}")
    print("-" * 60)
    print(f"{'MAE':<10} | {train_mae:<20.4f} | {test_mae:<20.4f}")
    print(f"{'RMSE':<10} | {train_rmse:<20.4f} | {test_rmse:<20.4f}")
    print(f"{'R2':<10} | {train_r2:<20.4f} | {test_r2:<20.4f}")
    print("-" * 60)
    
    assert train_r2 > 0.85, "Expected strong training R2!"
    assert test_r2 > 0.85, "Expected strong test generalization R2!"
    assert abs(train_r2 - test_r2) < 0.08, "Train and test performance should be balanced!"
    print("Conclusion: Model exhibits high generalization with balanced train and test metrics.")

if __name__ == "__main__":
    main()

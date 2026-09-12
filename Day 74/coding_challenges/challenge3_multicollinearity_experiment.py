"""
Day 74 - Challenge 3: Multicollinearity Experiment
Simulates synthetic multicollinearity to observe coefficient instability and VIF inflation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X1 = np.random.uniform(10, 100, n)
    X2 = X1 * 0.98 + np.random.normal(0, 1.0, n) # severe multicollinearity
    y = 50 + 2.0 * X1 + 0.0 * X2 + np.random.normal(0, 5, n)
    
    df_both = pd.DataFrame({"X1": X1, "X2": X2})
    m_both = LinearRegression().fit(df_both, y)
    
    df_one = pd.DataFrame({"X1": X1})
    m_one = LinearRegression().fit(df_one, y)
    
    vif_X = add_constant(df_both)
    vif1 = variance_inflation_factor(vif_X.values, 1)
    vif2 = variance_inflation_factor(vif_X.values, 2)
    
    print("=== Multicollinearity Simulation ===")
    print(f"Correlation between X1 and X2: {np.corrcoef(X1, X2)[0, 1]:.4f}")
    print(f"VIF(X1): {vif1:.1f}, VIF(X2): {vif2:.1f}")
    print(f"Model with both: Coef(X1)={m_both.coef_[0]:.3f}, Coef(X2)={m_both.coef_[1]:.3f} (True beta1=2.0, beta2=0.0)")
    print(f"Model with X1 only: Coef(X1)={m_one.coef_[0]:.3f}")

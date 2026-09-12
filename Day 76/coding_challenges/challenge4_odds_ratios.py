# Challenge 4: Odds Ratios
# Extract coefficients from a trained logistic regression model and convert them to odds ratios

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def calculate_odds_ratios(X, y, feature_names):
    model = LogisticRegression()
    model.fit(X, y)
    
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Odds_Ratio': odds_ratios
    })
    
    return df.sort_values('Odds_Ratio', ascending=False)

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    y = np.array([0, 0, 1, 1])
    
    res = calculate_odds_ratios(X, y, ['Feature_A', 'Feature_B'])
    print(res)

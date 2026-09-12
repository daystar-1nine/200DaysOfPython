"""
Day 73 - Scenario Prediction Engine
Generates predictions for business budget scenarios and flags extrapolation risks.
"""
from typing import List, Union
import numpy as np
import pandas as pd

try:
    from app.regression import SimpleLinearRegressor
except ImportError:
    from regression import SimpleLinearRegressor

def generate_scenario_predictions(
    model: SimpleLinearRegressor,
    budgets: List[float],
    min_x: float,
    max_x: float
) -> pd.DataFrame:
    if len(budgets) == 0:
        return pd.DataFrame({
            "Advertising_Spend": np.array([], dtype=float),
            "Predicted_Sales": np.array([], dtype=float),
            "Is_Extrapolation": np.array([], dtype=bool)
        })
    b_arr = np.array(budgets, dtype=float)
    preds = model.predict(b_arr)
    
    is_extrapolated = [(val < min_x or val > max_x) for val in b_arr]
    
    df_preds = pd.DataFrame({
        "Advertising_Spend": b_arr,
        "Predicted_Sales": np.round(preds, 2),
        "Is_Extrapolation": is_extrapolated
    })
    
    return df_preds

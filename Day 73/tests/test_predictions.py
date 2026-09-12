"""
Unit tests for scenario predictions engine.
"""
import pytest
import numpy as np
import pandas as pd
from app.regression import SimpleLinearRegressor
from app.predictions import generate_scenario_predictions

def test_generate_scenario_predictions_counts(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    budgets = [15.0, 25.0, 35.0, 45.0]
    df_preds = generate_scenario_predictions(reg, budgets, min_x=10.0, max_x=50.0)
    assert len(df_preds) == 4
    assert not df_preds["Is_Extrapolation"].any()

def test_extrapolation_flag_detection(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    budgets = [5.0, 20.0, 100.0]
    df_preds = generate_scenario_predictions(reg, budgets, min_x=10.0, max_x=50.0)
    # 5.0 is below min_x (10), 100.0 is above max_x (50)
    assert df_preds.loc[0, "Is_Extrapolation"] == True
    assert df_preds.loc[1, "Is_Extrapolation"] == False
    assert df_preds.loc[2, "Is_Extrapolation"] == True

def test_prediction_monotonic_growth(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    budgets = [10.0, 20.0, 30.0, 40.0]
    df_preds = generate_scenario_predictions(reg, budgets, min_x=10.0, max_x=40.0)
    preds = df_preds["Predicted_Sales"].values
    assert (np.diff(preds) > 0).all()

def test_scenario_empty_budgets(clean_linear_arrays):
    X, y = clean_linear_arrays
    reg = SimpleLinearRegressor().fit(X, y)
    df_preds = generate_scenario_predictions(reg, [], min_x=10.0, max_x=50.0)
    assert len(df_preds) == 0

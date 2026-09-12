import pytest
import pandas as pd
import numpy as np

try:
    from app.predictions import generate_predictions_df
except ImportError:
    from predictions import generate_predictions_df

@pytest.fixture
def pred_data():
    actuals = np.array([100, 200, 300])
    preds = np.array([110, 195, 305])
    return actuals, preds

def test_predictions_returns_dataframe(pred_data):
    actuals, preds = pred_data
    df = generate_predictions_df(actuals, preds)
    assert isinstance(df, pd.DataFrame)

def test_predictions_has_required_columns(pred_data):
    actuals, preds = pred_data
    df = generate_predictions_df(actuals, preds)
    expected = ['Actual', 'Predicted', 'Residual', 'AbsError']
    for col in expected:
        assert col in df.columns

def test_predictions_residual_formula(pred_data):
    actuals, preds = pred_data
    df = generate_predictions_df(actuals, preds)
    expected_residuals = actuals - preds
    pd.testing.assert_series_equal(df['Residual'], pd.Series(expected_residuals, name='Residual'))

def test_predictions_length_matches_input(pred_data):
    actuals, preds = pred_data
    df = generate_predictions_df(actuals, preds)
    assert len(df) == len(actuals)

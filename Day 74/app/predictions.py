import pandas as pd
import numpy as np

def generate_predictions_df(actuals, preds) -> pd.DataFrame:
    """
    Constructs a DataFrame containing Actual, Predicted, Residual, AbsError, and PctError.
    """
    actuals = np.asarray(actuals)
    preds = np.asarray(preds)
    residuals = actuals - preds
    abs_error = np.abs(residuals)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_error = np.where(actuals != 0, (abs_error / actuals) * 100, 0.0)

    return pd.DataFrame({
        'Actual': actuals,
        'Predicted': preds,
        'Residual': residuals,
        'AbsError': abs_error,
        'PctError': pct_error
    })

def generate_predictions(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Returns DataFrame with columns: Actual, Predicted, Residual, AbsError, PctError
    """
    preds = model.predict(X_test)
    return generate_predictions_df(y_test, preds)

def predict_scenario(model, scenario_dict: dict, feature_names: list = None) -> float:
    """
    Takes a dict of feature values, returns predicted Sales
    """
    df = pd.DataFrame([scenario_dict])
    return float(model.predict(df)[0])

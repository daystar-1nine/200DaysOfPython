import pandas as pd
import numpy as np
from app.model import ChurnLogisticModel

def generate_predictions(model: ChurnLogisticModel, X_test: pd.DataFrame, y_test: pd.Series = None, threshold: float = 0.50) -> pd.DataFrame:
    probabilities = model.predict_proba(X_test)
    predicted_classes = (probabilities >= threshold).astype(int)
    
    df = pd.DataFrame({
        'Probability': probabilities,
        'Predicted_Class': predicted_classes
    })
    
    if y_test is not None:
        df['Actual'] = y_test.values
        
    return df

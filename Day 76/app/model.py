import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.preprocessing import build_preprocessor

class ChurnLogisticModel:
    def __init__(self, preprocessor=None, max_iter=1000, random_state=42):
        if preprocessor is None:
            preprocessor = build_preprocessor()
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=max_iter, random_state=random_state))
        ])
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
        
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(X, y)
        
    def get_coefficients(self, feature_names=None) -> pd.DataFrame:
        classifier = self.model.named_steps['classifier']
        coefs = classifier.coef_[0]
        
        if feature_names is None:
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = [f"Feature_{i}" for i in range(len(coefs))]
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                pass
                
        df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        })
        df['Odds_Ratio'] = np.exp(df['Coefficient'])
        return df

def train_logistic_model(X: pd.DataFrame, y: pd.Series, preprocessor=None) -> ChurnLogisticModel:
    model = ChurnLogisticModel(preprocessor=preprocessor)
    model.fit(X, y)
    return model

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional, List

def build_preprocessor(numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None) -> ColumnTransformer:
    num = numeric_features or []
    cat = categorical_features or []
    
    transformers = []
    if num:
        transformers.append(('num', StandardScaler(), num))
    if cat:
        transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat))
        
    return ColumnTransformer(transformers=transformers, remainder='passthrough')

def preprocess_data(df: pd.DataFrame, numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None) -> np.ndarray:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return preprocessor.fit_transform(df)

class RegressionPreprocessor:
    def __init__(self, numeric_features=None, categorical_features=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self
        
    def transform(self, X):
        return self.preprocessor.transform(X)
        
    def fit_transform(self, X, y=None):
        return self.preprocessor.fit_transform(X)

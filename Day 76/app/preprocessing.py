import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = []
    if categorical_features is None:
        categorical_features = []
        
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

class ChurnPreprocessor:
    def __init__(self, numeric_features=None, categorical_features=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.fit_transform(X)
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(X)

def preprocess_data(df: pd.DataFrame, numeric_features=None, categorical_features=None) -> np.ndarray:
    preprocessor = ChurnPreprocessor(numeric_features, categorical_features)
    return preprocessor.fit_transform(df)

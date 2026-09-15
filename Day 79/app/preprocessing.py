from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_tree_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Preprocessor for tree models: passthrough numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

def build_linear_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Preprocessor for linear models: standardize numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

def split_data(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

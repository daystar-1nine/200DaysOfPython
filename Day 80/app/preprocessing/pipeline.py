from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from app.preprocessing.scaler import create_numerical_scaler
from app.preprocessing.encoder import create_categorical_encoder

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """Build ColumnTransformer for numeric and categorical pipelines."""
    num_trans = create_numerical_scaler() if scale_numeric else 'passthrough'
    cat_trans = create_categorical_encoder()
    
    return ColumnTransformer(
        transformers=[
            ('num', num_trans, numeric_features),
            ('cat', cat_trans, categorical_features)
        ]
    )

def split_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split without data leakage."""
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

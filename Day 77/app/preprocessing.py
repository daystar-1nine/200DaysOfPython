import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from app.config import AppConfig

def build_preprocessor(numeric_features=None, categorical_features=None, config: AppConfig = None) -> ColumnTransformer:
    if config is None:
        config = AppConfig()
    if numeric_features is None:
        numeric_features = config.NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = config.CATEGORICAL_FEATURES
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer, numeric_features, categorical_features) -> list:
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_cols

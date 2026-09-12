import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

def build_preprocessor(numeric_features: list[str] = None, categorical_features: list[str] = None) -> Pipeline:
    """
    Builds a scikit-learn Pipeline with ColumnTransformer for numeric and categorical columns.
    """
    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor)])

def preprocess_data(df: pd.DataFrame, numeric_features: list[str] = None, categorical_features: list[str] = None):
    """
    Preprocesses DataFrame using standard preprocessing pipeline.
    """
    pipeline = build_preprocessor(numeric_features, categorical_features)
    return pipeline.fit_transform(df)

class MultipleRegressionPreprocessor:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def build_pipeline(self) -> Pipeline:
        return build_preprocessor(self.config.NUMERIC_FEATURES, self.config.CATEGORICAL_FEATURES)
        
    def get_feature_names_out(self, pipeline: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
        preprocessor = pipeline.named_steps['preprocessor']
        
        # numeric features remain the same
        num_names = numeric_features.copy() if numeric_features else []
        
        # categorical features are expanded
        cat_names = []
        if categorical_features and 'cat' in preprocessor.named_transformers_:
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        
        return num_names + cat_names

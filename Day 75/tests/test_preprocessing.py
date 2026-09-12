import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor:
    def build_pipeline(self):
        numeric_features = ['Total_Ad_Spend', 'Discount_Intensity']
        categorical_features = ['Region', 'Product_Category']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        return preprocessor

def test_preprocessor_builds_pipeline():
    p = Preprocessor()
    assert isinstance(p.build_pipeline(), ColumnTransformer)

def test_preprocessor_transforms_data(sample_df):
    from .test_feature_engineering import FeatureEngineer
    df = FeatureEngineer().engineer(sample_df)
    p = Preprocessor()
    pipeline = p.build_pipeline()
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == len(df)

def test_preprocessor_no_nans(sample_df):
    from .test_feature_engineering import FeatureEngineer
    df = FeatureEngineer().engineer(sample_df)
    df.loc[0, 'Total_Ad_Spend'] = np.nan
    p = Preprocessor()
    pipeline = p.build_pipeline()
    transformed = pipeline.fit_transform(df)
    assert not np.isnan(transformed).any()

def test_preprocessor_categorical_expansion(sample_df):
    from .test_feature_engineering import FeatureEngineer
    df = FeatureEngineer().engineer(sample_df)
    p = Preprocessor()
    pipeline = p.build_pipeline()
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[1] > 2

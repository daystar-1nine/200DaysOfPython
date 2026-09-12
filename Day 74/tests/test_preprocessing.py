import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

try:
    from app.preprocessing import build_preprocessor, preprocess_data
except ImportError:
    from preprocessing import build_preprocessor, preprocess_data

def test_preprocessor_builds_pipeline():
    cat_cols = ['Region', 'Category']
    num_cols = ['TV_Spend', 'Digital_Spend', 'Radio_Spend']
    pipeline = build_preprocessor(num_cols, cat_cols)
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'transform')

def test_preprocessor_transforms_data(sample_df):
    cat_cols = ['Region', 'Category']
    num_cols = ['TV_Spend', 'Digital_Spend']
    pipeline = build_preprocessor(num_cols, cat_cols)
    transformed = pipeline.fit_transform(sample_df)
    assert transformed.shape[0] == sample_df.shape[0]

def test_preprocessor_output_not_nan(sample_df):
    cat_cols = ['Region', 'Category']
    num_cols = ['TV_Spend', 'Digital_Spend']
    pipeline = build_preprocessor(num_cols, cat_cols)
    transformed = pipeline.fit_transform(sample_df)
    assert not np.isnan(transformed).any()

def test_preprocessor_categorical_encoded(sample_df):
    cat_cols = ['Region']
    num_cols = []
    pipeline = build_preprocessor(num_cols, cat_cols)
    transformed = pipeline.fit_transform(sample_df)
    # One-hot encoding should expand the 'Region' column
    assert transformed.shape[1] > 1

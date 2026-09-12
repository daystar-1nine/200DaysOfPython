import pytest
import numpy as np
import pandas as pd
from app.validation.stratified_split import stratified_train_test_split

def test_stratified_split_preserves_ratios(sample_df):
    target_col = "Churn"
    orig_ratio = sample_df[target_col].mean()
    X_train, X_test, y_train, y_test = stratified_train_test_split(sample_df, target_col, test_size=0.25, random_state=42)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(orig_ratio - train_ratio) < 0.05
    assert abs(orig_ratio - test_ratio) < 0.05

def test_stratified_split_shapes(sample_df):
    X_train, X_test, y_train, y_test = stratified_train_test_split(sample_df, "Risk_Level", test_size=0.20, random_state=42)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)
    assert len(X_test) == int(len(sample_df) * 0.20)

def test_stratified_split_drops_target_column(sample_df):
    target = "Churn"
    X_train, X_test, y_train, y_test = stratified_train_test_split(sample_df, target, test_size=0.20)
    assert target not in X_train.columns
    assert target not in X_test.columns

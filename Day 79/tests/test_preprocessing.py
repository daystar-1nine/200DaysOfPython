import pandas as pd
import numpy as np
from app.preprocessing import build_tree_preprocessor, build_linear_preprocessor, split_data

def test_tree_preprocessor(sample_config, split_sample_data):
    X_train, X_test, _, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    X_trans = preproc.fit_transform(X_train)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(X_train)
    assert X_trans.shape[1] >= len(sample_config.numeric_features)

def test_linear_preprocessor_scaling(sample_config, split_sample_data):
    X_train, X_test, _, _ = split_sample_data
    preproc = build_linear_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    X_trans = preproc.fit_transform(X_train)
    num_dim = len(sample_config.numeric_features)
    means = np.mean(X_trans[:, :num_dim], axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-2)

def test_split_data_shapes(engineered_sample_df, sample_config):
    feature_cols = sample_config.numeric_features + sample_config.categorical_features
    X_tr, X_te, y_tr, y_te = split_data(engineered_sample_df, feature_cols, sample_config.target_column, test_size=0.20)
    assert len(X_tr) + len(X_te) == len(engineered_sample_df)
    assert len(y_tr) == len(X_tr)

def test_split_stratification(engineered_sample_df, sample_config):
    feature_cols = sample_config.numeric_features + sample_config.categorical_features
    _, _, y_tr, y_te = split_data(engineered_sample_df, feature_cols, sample_config.target_column, test_size=0.25)
    tr_rate = np.mean(y_tr)
    te_rate = np.mean(y_te)
    assert abs(tr_rate - te_rate) < 0.05

def test_preprocessor_handles_unseen_categories(sample_config, split_sample_data):
    X_train, X_test, _, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    preproc.fit(X_train)
    X_unseen = X_test.copy()
    X_unseen.loc[X_unseen.index[0], 'Contract_Type'] = 'Unknown_Custom_Tier'
    res = preproc.transform(X_unseen)
    assert res.shape[0] == len(X_unseen)

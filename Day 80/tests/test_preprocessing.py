import numpy as np
from app.preprocessing.pipeline import build_preprocessor, split_dataset

def test_linear_preprocessor_scales(sample_config, sample_splits):
    X_tr, _, _, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=True)
    res = prep.fit_transform(X_tr)
    num_dim = len(sample_config.numeric_features)
    means = np.mean(res[:, :num_dim], axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-2)

def test_tree_preprocessor_passthrough(sample_config, sample_splits):
    X_tr, _, _, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    res = prep.fit_transform(X_tr)
    num_dim = len(sample_config.numeric_features)
    assert res.shape[1] >= num_dim

def test_split_dataset_shapes(sample_clean_df, sample_config):
    feature_cols = sample_config.numeric_features + sample_config.categorical_features
    X_tr, X_te, y_tr, y_te = split_dataset(sample_clean_df, feature_cols, sample_config.target_column, test_size=0.20)
    assert len(X_tr) + len(X_te) == len(sample_clean_df)
    assert len(y_tr) == len(X_tr)

def test_split_dataset_stratification(sample_clean_df, sample_config):
    feature_cols = sample_config.numeric_features + sample_config.categorical_features
    _, _, y_tr, y_te = split_dataset(sample_clean_df, feature_cols, sample_config.target_column, test_size=0.25)
    tr_rate = np.mean(y_tr)
    te_rate = np.mean(y_te)
    assert abs(tr_rate - te_rate) < 0.05

def test_unseen_category_handling(sample_config, sample_splits):
    X_tr, X_te, _, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    prep.fit(X_tr)
    X_mod = X_te.copy()
    X_mod.loc[X_mod.index[0], 'Contract_Type'] = 'SuperCustomContract'
    res = prep.transform(X_mod)
    assert res.shape[0] == len(X_mod)

import pandas as pd
from app.preprocessing import build_tree_preprocessor
from app.models.random_forest import create_random_forest_pipeline
from app.feature_importance import compute_mdi_importance, extract_feature_names
from app.permutation_importance import compute_permutation_importance

def test_compute_mdi_importance(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=30, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    df_mdi = compute_mdi_importance(pipe, sample_config.numeric_features, sample_config.categorical_features)
    assert isinstance(df_mdi, pd.DataFrame)
    assert 'Feature' in df_mdi.columns and 'Importance' in df_mdi.columns and 'Std' in df_mdi.columns
    assert (df_mdi['Importance'] >= 0.0).all()
    assert abs(df_mdi['Importance'].sum() - 1.0) < 1e-4

def test_extract_feature_names(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=20, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    names = extract_feature_names(pipe, sample_config.numeric_features, sample_config.categorical_features)
    assert len(names) >= len(sample_config.numeric_features)

def test_compute_permutation_importance(sample_config, split_sample_data):
    X_tr, X_te, y_tr, y_te = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=30, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    df_perm = compute_permutation_importance(pipe, X_te, y_te, n_repeats=3, scoring='accuracy')
    assert isinstance(df_perm, pd.DataFrame)
    assert 'Feature' in df_perm.columns and 'Mean_Importance' in df_perm.columns
    assert len(df_perm) == len(X_te.columns)

def test_mdi_sorted_descending(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=25, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    df_mdi = compute_mdi_importance(pipe, sample_config.numeric_features, sample_config.categorical_features)
    assert df_mdi['Importance'].is_monotonic_decreasing

import pytest
from app.preprocessing import build_tree_preprocessor
from app.models.random_forest import create_random_forest_pipeline
from app.tuning.cross_validation import evaluate_stratified_cv
from app.tuning.grid_search import run_rf_grid_search

def test_evaluate_stratified_cv_structure(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=30, max_depth=4, oob_score=False, random_state=42)
    summary = evaluate_stratified_cv(pipe, X_tr, y_tr, cv=3)
    for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        assert m in summary
        assert 'mean' in summary[m]
        assert 'std' in summary[m]
        assert 0.0 <= summary[m]['mean'] <= 1.0

def test_run_rf_grid_search(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=20, oob_score=False, random_state=42)
    param_grid = {'classifier__max_depth': [3, 5]}
    best_pipe, best_params, best_score = run_rf_grid_search(pipe, X_tr, y_tr, param_grid, cv=2, scoring='accuracy')
    assert 'classifier__max_depth' in best_params
    assert best_params['classifier__max_depth'] in [3, 5]
    assert 0.0 <= best_score <= 1.0

def test_grid_search_refit(sample_config, split_sample_data):
    X_tr, X_te, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=20, oob_score=False, random_state=42)
    param_grid = {'classifier__max_depth': [3]}
    best_pipe, _, _ = run_rf_grid_search(pipe, X_tr, y_tr, param_grid, cv=2)
    preds = best_pipe.predict(X_te)
    assert len(preds) == len(X_te)

def test_cv_std_non_negative(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=20, oob_score=False, random_state=42)
    summary = evaluate_stratified_cv(pipe, X_tr, y_tr, cv=2)
    assert summary['f1']['std'] >= 0.0

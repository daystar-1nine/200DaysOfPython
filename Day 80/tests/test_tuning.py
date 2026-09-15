from app.preprocessing.pipeline import build_preprocessor
from app.models.decision_tree import create_decision_tree_pipeline
from app.tuning.grids import get_logistic_param_grid, get_decision_tree_param_grid, get_random_forest_param_grid
from app.tuning.search import run_model_grid_search

def test_param_grids_return_dicts():
    assert isinstance(get_logistic_param_grid(), dict)
    assert isinstance(get_decision_tree_param_grid(), dict)
    assert isinstance(get_random_forest_param_grid(), dict)

def test_run_model_grid_search(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    param_grid = {'classifier__max_depth': [2, 4]}
    best_pipe, best_params, best_score = run_model_grid_search(pipe, X_tr, y_tr, param_grid, cv=2, scoring='roc_auc')
    assert 'classifier__max_depth' in best_params
    assert best_params['classifier__max_depth'] in [2, 4]
    assert 0.0 <= best_score <= 1.0

def test_tuned_pipeline_predicts(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    param_grid = {'classifier__max_depth': [3]}
    best_pipe, _, _ = run_model_grid_search(pipe, X_tr, y_tr, param_grid, cv=2)
    preds = best_pipe.predict(X_te)
    assert len(preds) == len(X_te)

def test_tuning_score_validity(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    param_grid = {'classifier__max_depth': [3]}
    _, _, best_score = run_model_grid_search(pipe, X_tr, y_tr, param_grid, cv=2, scoring='accuracy')
    assert 0.0 <= best_score <= 1.0

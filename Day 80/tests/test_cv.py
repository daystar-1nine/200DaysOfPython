from app.preprocessing.pipeline import build_preprocessor
from app.models.decision_tree import create_decision_tree_pipeline
from app.evaluation.cross_validation import run_stratified_cv

def test_run_stratified_cv_keys(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    cv_res = run_stratified_cv(pipe, X_tr, y_tr, cv=2)
    for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
        assert m in cv_res
        assert 'test_mean' in cv_res[m]
        assert 'test_std' in cv_res[m]

def test_cv_train_and_test_scores(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    cv_res = run_stratified_cv(pipe, X_tr, y_tr, cv=2)
    assert 'train_mean' in cv_res['roc_auc']
    assert 0.0 <= cv_res['roc_auc']['test_mean'] <= 1.0

def test_cv_std_non_negative(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    cv_res = run_stratified_cv(pipe, X_tr, y_tr, cv=2)
    assert cv_res['f1']['test_std'] >= 0.0

def test_cv_stratification(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=3)
    cv_res = run_stratified_cv(pipe, X_tr, y_tr, cv=3)
    assert cv_res['accuracy']['test_mean'] > 0.0

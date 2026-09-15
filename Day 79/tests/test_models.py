import numpy as np
from app.preprocessing import build_tree_preprocessor, build_linear_preprocessor
from app.models.logistic_baseline import create_logistic_pipeline
from app.models.decision_tree import create_decision_tree_pipeline
from app.models.random_forest import create_random_forest_pipeline

def test_logistic_baseline_pipeline(sample_config, split_sample_data):
    X_tr, X_te, y_tr, y_te = split_sample_data
    preproc = build_linear_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_logistic_pipeline(preproc, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    probs = pipe.predict_proba(X_te)
    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_decision_tree_pipeline(sample_config, split_sample_data):
    X_tr, X_te, y_tr, y_te = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(preproc, max_depth=4, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert set(preds).issubset({0, 1})

def test_random_forest_pipeline_oob_score(sample_config, split_sample_data):
    X_tr, X_te, y_tr, y_te = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=60, max_depth=5, bootstrap=True, oob_score=True, random_state=42)
    pipe.fit(X_tr, y_tr)
    rf = pipe.named_steps['classifier']
    assert hasattr(rf, 'oob_score_')
    assert 0.0 <= rf.oob_score_ <= 1.0

def test_random_forest_predict_proba_sums(sample_config, split_sample_data):
    X_tr, X_te, y_tr, y_te = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=50, random_state=42)
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)

def test_random_forest_estimators_count(sample_config, split_sample_data):
    X_tr, _, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=45, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    rf = pipe.named_steps['classifier']
    assert len(rf.estimators_) == 45

def test_random_forest_predictions_binary(sample_config, split_sample_data):
    X_tr, X_te, y_tr, _ = split_sample_data
    preproc = build_tree_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(preproc, n_estimators=40, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert set(preds).issubset({0, 1})

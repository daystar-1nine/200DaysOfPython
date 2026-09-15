import numpy as np
from app.preprocessing.pipeline import build_preprocessor
from app.models.random_forest import create_random_forest_pipeline

def test_random_forest_creation(sample_config):
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_random_forest_pipeline(prep, n_estimators=60, max_depth=8)
    assert pipe.named_steps['classifier'].n_estimators == 60

def test_random_forest_oob_score(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=50, oob_score=True, random_state=42)
    pipe.fit(X_tr, y_tr)
    rf = pipe.named_steps['classifier']
    assert hasattr(rf, 'oob_score_')
    assert 0.0 <= rf.oob_score_ <= 1.0

def test_random_forest_predictions(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=30, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(X_te)
    assert set(preds).issubset({0, 1})

def test_random_forest_estimators_count(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=25, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    assert len(pipe.named_steps['classifier'].estimators_) == 25

def test_random_forest_probabilities_bounded(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_random_forest_pipeline(prep, n_estimators=30, oob_score=False, random_state=42)
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

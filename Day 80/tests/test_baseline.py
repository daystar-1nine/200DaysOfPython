import numpy as np
from app.preprocessing.pipeline import build_preprocessor
from app.models.baseline import create_baseline_pipeline

def test_baseline_pipeline_structure(sample_config):
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_baseline_pipeline(prep)
    assert 'preprocessor' in pipe.named_steps
    assert 'classifier' in pipe.named_steps

def test_baseline_fitting(sample_config, sample_splits):
    X_tr, X_te, y_tr, y_te = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_baseline_pipeline(prep)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(X_te)

def test_baseline_predicts_majority_class(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_baseline_pipeline(prep)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    majority = y_tr.mode()[0]
    assert (preds == majority).all()

def test_baseline_proba_output(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_baseline_pipeline(prep)
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)

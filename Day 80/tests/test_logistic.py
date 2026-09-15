import numpy as np
from app.preprocessing.pipeline import build_preprocessor
from app.models.logistic import create_logistic_pipeline

def test_logistic_creation(sample_config):
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_logistic_pipeline(prep, C=1.0)
    assert pipe.named_steps['classifier'].C == 1.0

def test_logistic_training(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=True)
    pipe = create_logistic_pipeline(prep, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(X_te)

def test_logistic_binary_preds(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=True)
    pipe = create_logistic_pipeline(prep, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert set(preds).issubset({0, 1})

def test_logistic_probabilities_sum(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=True)
    pipe = create_logistic_pipeline(prep, random_state=42)
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

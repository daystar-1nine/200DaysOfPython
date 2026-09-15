import numpy as np
from app.preprocessing.pipeline import build_preprocessor
from app.models.decision_tree import create_decision_tree_pipeline

def test_decision_tree_creation(sample_config):
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features)
    pipe = create_decision_tree_pipeline(prep, max_depth=5)
    assert pipe.named_steps['classifier'].max_depth == 5

def test_decision_tree_training(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_decision_tree_pipeline(prep, max_depth=4)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(X_te)

def test_decision_tree_probabilities(sample_config, sample_splits):
    X_tr, X_te, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_decision_tree_pipeline(prep, max_depth=4)
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_decision_tree_feature_importances(sample_config, sample_splits):
    X_tr, _, y_tr, _ = sample_splits
    prep = build_preprocessor(sample_config.numeric_features, sample_config.categorical_features, scale_numeric=False)
    pipe = create_decision_tree_pipeline(prep, max_depth=4)
    pipe.fit(X_tr, y_tr)
    dt = pipe.named_steps['classifier']
    assert hasattr(dt, 'feature_importances_')
    assert abs(np.sum(dt.feature_importances_) - 1.0) < 1e-4

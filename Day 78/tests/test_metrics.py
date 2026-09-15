import numpy as np
from app.evaluation.metrics import calculate_metrics, compare_models
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor
from app.feature_engineering import engineer_features

def test_calculate_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    m = calculate_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0

def test_calculate_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = calculate_metrics(y_true, y_pred)
    assert "accuracy" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m

def test_compare_models(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m1 = build_decision_tree_model(prep, max_depth=2)
    m1.fit(df_feat, sample_df["Churn"])
    comp = compare_models({"Tree": m1}, df_feat, sample_df["Churn"])
    assert len(comp) == 1
    assert "F1" in comp.columns
    assert "ROC-AUC" in comp.columns

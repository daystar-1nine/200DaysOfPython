from app.tuning.cross_validation import run_stratified_cv, evaluate_depth_curve
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor
from app.feature_engineering import engineer_features

def test_run_stratified_cv(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    res = run_stratified_cv(m, df_feat, sample_df["Churn"], n_splits=3)
    assert "mean_f1" in res
    assert "mean_accuracy" in res
    assert "mean_roc_auc" in res
    assert len(res["all_f1"]) == 3

def test_evaluate_depth_curve(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    df_curve = evaluate_depth_curve(prep, df_feat, sample_df["Churn"], df_feat, sample_df["Churn"], depths=[2, 4])
    assert len(df_curve) == 2
    assert "Train_F1" in df_curve.columns
    assert "Test_F1" in df_curve.columns

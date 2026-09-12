from app.validation.cross_validation import run_stratified_cv
from app.models.logistic import build_logistic_model
from app.preprocessing import build_preprocessor
from app.feature_engineering import engineer_features

def test_stratified_cv_binary(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_logistic_model(prep)
    res = run_stratified_cv(model, df_feat, sample_df["Churn"], n_splits=3, is_multiclass=False)
    assert "mean_accuracy" in res
    assert "mean_f1_macro" in res
    assert "mean_roc_auc" in res
    assert len(res["all_accuracy"]) == 3

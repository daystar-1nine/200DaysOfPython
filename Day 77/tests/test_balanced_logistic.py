from app.models.balanced_logistic import build_balanced_logistic_model
from app.preprocessing import build_preprocessor
from app.feature_engineering import engineer_features

def test_build_balanced_logistic(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_balanced_logistic_model(prep, multi_class="auto")
    model.fit(df_feat, sample_df["Churn"])
    preds = model.predict(df_feat)
    assert len(preds) == len(sample_df)

def test_balanced_multiclass_ovr(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_balanced_logistic_model(prep, multi_class="ovr")
    model.fit(df_feat, sample_df["Risk_Level"])
    probs = model.predict_proba(df_feat)
    assert probs.shape[1] == 3

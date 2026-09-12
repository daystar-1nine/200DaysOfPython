from app.models.logistic import build_logistic_model
from app.preprocessing import build_preprocessor
from app.feature_engineering import engineer_features

def test_build_logistic_binary(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_logistic_model(prep, multi_class="auto")
    model.fit(df_feat, sample_df["Churn"])
    preds = model.predict(df_feat)
    assert set(preds).issubset({0, 1})

def test_build_logistic_multiclass(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_logistic_model(prep, multi_class="ovr")
    model.fit(df_feat, sample_df["Risk_Level"])
    preds = model.predict(df_feat)
    assert set(preds).issubset({"Low", "Medium", "High"})

def test_logistic_predict_proba(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    model = build_logistic_model(prep, multi_class="auto")
    model.fit(df_feat, sample_df["Churn"])
    probs = model.predict_proba(df_feat)
    assert probs.shape == (len(sample_df), 2)
    assert (probs >= 0).all() and (probs <= 1).all()

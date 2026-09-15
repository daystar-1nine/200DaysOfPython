from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor
from app.feature_engineering import engineer_features

def test_decision_tree_gini_fit(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, criterion="gini", max_depth=4)
    m.fit(df_feat, sample_df["Churn"])
    preds = m.predict(df_feat)
    assert set(preds).issubset({0, 1})

def test_decision_tree_entropy_fit(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, criterion="entropy", max_depth=4)
    m.fit(df_feat, sample_df["Churn"])
    preds = m.predict(df_feat)
    assert len(preds) == len(sample_df)

def test_decision_tree_predict_proba(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    m.fit(df_feat, sample_df["Churn"])
    probs = m.predict_proba(df_feat)
    assert probs.shape == (len(sample_df), 2)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_decision_tree_max_depth_respected(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    m.fit(df_feat, sample_df["Churn"])
    clf = m.named_steps["classifier"]
    assert clf.get_depth() <= 3

def test_decision_tree_min_samples_leaf(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, min_samples_leaf=15)
    m.fit(df_feat, sample_df["Churn"])
    clf = m.named_steps["classifier"]
    assert clf.get_n_leaves() > 0

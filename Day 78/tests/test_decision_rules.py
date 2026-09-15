from app.decision_rules import export_tree_rules, explain_customer_path
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor, get_transformed_feature_names
from app.feature_engineering import engineer_features

def test_export_tree_rules(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    m.fit(df_feat, sample_df["Churn"])
    
    feat_names = get_transformed_feature_names(prep, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    rules = export_tree_rules(m, feat_names)
    assert isinstance(rules, str)
    assert "class:" in rules

def test_explain_customer_path(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    m.fit(df_feat, sample_df["Churn"])
    
    feat_names = get_transformed_feature_names(prep, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    sample_row = df_feat.iloc[[0]]
    explanation = explain_customer_path(m, sample_row, feat_names)
    
    assert "Customer_ID" in explanation
    assert "Predicted_Class" in explanation
    assert "Churn_Probability" in explanation
    assert isinstance(explanation["Decision_Path"], list)

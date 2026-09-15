import numpy as np
from app.feature_importance import extract_feature_importance
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor, get_transformed_feature_names
from app.feature_engineering import engineer_features

def test_extract_feature_importance(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=4)
    m.fit(df_feat, sample_df["Churn"])
    
    feat_names = get_transformed_feature_names(prep, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    imp_df = extract_feature_importance(m, feat_names)
    
    assert len(imp_df) == len(feat_names)
    assert np.isclose(imp_df["Importance"].sum(), 1.0)
    assert (imp_df["Importance"].diff().dropna() <= 0).all()  # descending order

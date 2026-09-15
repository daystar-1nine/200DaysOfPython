import numpy as np
from app.permutation_importance import compute_permutation_importance
from app.models.decision_tree import build_decision_tree_model
from app.preprocessing import build_tree_preprocessor, get_transformed_feature_names
from app.feature_engineering import engineer_features

def test_compute_permutation_importance(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_tree_preprocessor(config=config)
    m = build_decision_tree_model(prep, max_depth=3)
    m.fit(df_feat, sample_df["Churn"])
    
    feat_names = get_transformed_feature_names(prep, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    perm_df = compute_permutation_importance(m, df_feat, sample_df["Churn"], feat_names, scoring="f1", n_repeats=3)
    
    assert len(perm_df) == len(df_feat.columns)
    assert "Importance_Mean" in perm_df.columns
    assert "Importance_Std" in perm_df.columns

import numpy as np
import pandas as pd
from app.models.tree_comparison import build_random_forest_model
from app.preprocessing import build_preprocessor
from app.feature_engineering import engineer_features

def test_tree_model_fits(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    rf = build_random_forest_model(prep, n_estimators=10, max_depth=4)
    rf.fit(df_feat, sample_df["Churn"])
    preds = rf.predict(df_feat)
    assert len(preds) == len(sample_df)

def test_tree_model_predict_proba(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    rf = build_random_forest_model(prep, n_estimators=10, max_depth=4)
    rf.fit(df_feat, sample_df["Risk_Level"])
    probs = rf.predict_proba(df_feat)
    assert probs.shape[1] == 3
    assert np.allclose(probs.sum(axis=1), 1.0)

import numpy as np
import pandas as pd
from app.evaluation.metrics import compare_models
from app.models.logistic import build_logistic_model
from app.models.balanced_logistic import build_balanced_logistic_model
from app.preprocessing import build_preprocessor
from app.feature_engineering import engineer_features

def test_compare_models_binary(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    m1 = build_logistic_model(prep)
    m2 = build_balanced_logistic_model(prep)
    m1.fit(df_feat, sample_df["Churn"])
    m2.fit(df_feat, sample_df["Churn"])
    
    comp_df = compare_models({"M1": m1, "M2": m2}, df_feat, sample_df["Churn"], is_multiclass=False)
    assert len(comp_df) == 2
    assert "ROC-AUC" in comp_df.columns
    assert "F1" in comp_df.columns

def test_compare_models_multiclass(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    m1 = build_logistic_model(prep, multi_class="ovr")
    m1.fit(df_feat, sample_df["Risk_Level"])
    
    comp_df = compare_models({"M1": m1}, df_feat, sample_df["Risk_Level"], is_multiclass=True)
    assert len(comp_df) == 1
    assert "Macro F1" in comp_df.columns
    assert "High Recall" in comp_df.columns

def test_insights_generation(sample_df, config):
    from app.insights import generate_business_insights
    bin_df = pd.DataFrame({"Model": ["M1"], "Accuracy": [0.9]})
    cost_df = pd.DataFrame({"Threshold": [0.5, 0.2], "Total_Cost": [10000.0, 5000.0]})
    ins = generate_business_insights(bin_df, cost_df, sample_df, 0.2, 0.4)
    assert len(ins) >= 4
    for item in ins:
        assert isinstance(item, str) and len(item) > 10

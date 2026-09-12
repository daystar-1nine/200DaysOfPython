from app.feature_engineering import engineer_features

def test_engineer_features_adds_columns(sample_df):
    df_feat = engineer_features(sample_df)
    expected = ["Charges_Per_Month_Ratio", "Support_Per_Tenure", "Complaint_Risk_Index", "Net_Monthly_Charges"]
    for col in expected:
        assert col in df_feat.columns

def test_engineer_features_preserves_rows(sample_df):
    df_feat = engineer_features(sample_df)
    assert len(df_feat) == len(sample_df)

def test_engineer_net_monthly_charges(sample_df):
    df_feat = engineer_features(sample_df)
    expected_net = sample_df["Monthly_Charges"] * (1.0 - sample_df["Discount"] / 100.0)
    assert (df_feat["Net_Monthly_Charges"].round(4) == expected_net.round(4)).all()

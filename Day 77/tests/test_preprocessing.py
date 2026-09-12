from app.preprocessing import build_preprocessor, get_feature_names
from app.feature_engineering import engineer_features

def test_preprocessor_builds(config):
    prep = build_preprocessor(config=config)
    assert prep is not None

def test_preprocessor_transforms(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    X = prep.fit_transform(df_feat)
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] > len(config.NUMERIC_FEATURES)

def test_preprocessor_feature_names(sample_df, config):
    df_feat = engineer_features(sample_df)
    prep = build_preprocessor(config=config)
    prep.fit(df_feat)
    names = get_feature_names(prep, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    assert len(names) == prep.transform(df_feat).shape[1]

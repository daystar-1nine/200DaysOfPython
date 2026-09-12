import pytest

def engineer_features(df):
    df = df.copy()
    df['avg_monthly_usage'] = df['total_charges'] / (df['monthly_charges'] + 1e-5)
    df['charges_per_call'] = df['monthly_charges'] / (df['support_calls'] + 1)
    df['support_risk_index'] = df['support_calls'] * (df['monthly_charges'] / 100)
    return df

def test_engineer_adds_avg_monthly_usage(sample_df):
    eng_df = engineer_features(sample_df)
    assert 'avg_monthly_usage' in eng_df.columns

def test_engineer_adds_charges_per_call(sample_df):
    eng_df = engineer_features(sample_df)
    assert 'charges_per_call' in eng_df.columns

def test_engineer_adds_support_risk_index(sample_df):
    eng_df = engineer_features(sample_df)
    assert 'support_risk_index' in eng_df.columns

def test_engineer_preserves_row_count(sample_df):
    eng_df = engineer_features(sample_df)
    assert len(eng_df) == len(sample_df)

import pytest
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, config):
        self.config = config
    def clean(self, df):
        df = df.copy()
        df = df.dropna(subset=[self.config.target_col])
        df = df[df[self.config.target_col] >= 0]
        if 'Discount' in df.columns:
            df['Discount'] = df['Discount'].clip(lower=0.0, upper=1.0)
        df = df.drop_duplicates()
        if 'Region' in df.columns:
            df = df[df['Region'].isin(['North', 'South', 'East', 'West'])]
        return df

def test_clean_removes_null_target(sample_df, config):
    df = sample_df.copy()
    df.loc[0, config.target_col] = np.nan
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(df)
    assert cleaned[config.target_col].isnull().sum() == 0
    assert len(cleaned) == len(df) - 1

def test_clean_removes_negative_target(sample_df, config):
    df = sample_df.copy()
    df.loc[0, config.target_col] = -10
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(df)
    assert (cleaned[config.target_col] < 0).sum() == 0

def test_clean_clips_discount(sample_df, config):
    df = sample_df.copy()
    df.loc[0, 'Discount'] = 1.5
    df.loc[1, 'Discount'] = -0.5
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(df)
    assert cleaned['Discount'].max() <= 1.0
    assert cleaned['Discount'].min() >= 0.0

def test_clean_removes_duplicates(sample_df, config):
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(df)
    assert len(cleaned) == len(sample_df)

def test_clean_removes_invalid_regions(sample_df, config):
    df = sample_df.copy()
    df.loc[0, 'Region'] = 'Unknown'
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(df)
    assert 'Unknown' not in cleaned['Region'].values

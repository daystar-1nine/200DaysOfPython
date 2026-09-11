"""
Unit tests for synthetic user generation module.
"""

import pytest
import pandas as pd

try:
    from app.generator import ExperimentGenerator
except (ImportError, ModuleNotFoundError):
    from generator import ExperimentGenerator

def test_generate_users_shape_and_columns():
    gen = ExperimentGenerator(seed=42)
    df = gen.generate_users(n_users=500, control_ratio=0.5)
    assert len(df) == 500
    expected_cols = {"user_id", "group", "converted", "revenue", "session_duration", "orders", "bounce", "refunded"}
    assert expected_cols.issubset(set(df.columns))

def test_generate_users_binary_outcomes():
    gen = ExperimentGenerator(seed=42)
    df = gen.generate_users(n_users=200)
    assert set(df["converted"].unique()).issubset({0, 1})
    assert set(df["bounce"].unique()).issubset({0, 1})
    assert set(df["refunded"].unique()).issubset({0, 1})

def test_generate_users_reproducibility():
    gen1 = ExperimentGenerator(seed=123)
    df1 = gen1.generate_users(n_users=100)
    gen2 = ExperimentGenerator(seed=123)
    df2 = gen2.generate_users(n_users=100)
    pd.testing.assert_frame_equal(df1, df2)

def test_generate_users_revenue_only_when_converted():
    gen = ExperimentGenerator(seed=42)
    df = gen.generate_users(n_users=300)
    non_conv = df[df["converted"] == 0]
    assert (non_conv["revenue"] == 0.0).all()
    assert (non_conv["orders"] == 0).all()

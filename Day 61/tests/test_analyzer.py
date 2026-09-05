"""
Unit Tests for Analytical Aggregations (analyzer.py).
"""

import pytest
import pandas as pd
import numpy as np
from app import analyzer


def test_get_monthly_revenue(sample_sales_df):
    """Tests monthly revenue grouping and aggregation."""
    res = analyzer.get_monthly_revenue(sample_sales_df)
    assert isinstance(res, pd.Series)
    assert len(res) == 12
    assert res.sum() == pytest.approx(sample_sales_df["Revenue"].sum())


def test_get_revenue_by_region(sample_sales_df):
    """Tests regional grouping and descending sorting."""
    res = analyzer.get_revenue_by_region(sample_sales_df)
    assert isinstance(res, pd.Series)
    assert len(res) == 4
    # Verify sorted descending
    assert res.iloc[0] >= res.iloc[1] >= res.iloc[2] >= res.iloc[3]
    assert res.sum() == pytest.approx(sample_sales_df["Revenue"].sum())


def test_get_revenue_by_category(sample_sales_df):
    """Tests category grouping and descending sorting."""
    res = analyzer.get_revenue_by_category(sample_sales_df)
    assert isinstance(res, pd.Series)
    assert len(res) == 3
    assert res.iloc[0] >= res.iloc[1] >= res.iloc[2]


def test_get_top_n_products(sample_sales_df):
    """Tests extraction of top N products."""
    top2 = analyzer.get_top_n_products(sample_sales_df, n=2)
    assert len(top2) == 2
    assert isinstance(top2, pd.Series)


def test_get_top_n_customers(sample_sales_df):
    """Tests extraction of top N customers."""
    top2 = analyzer.get_top_n_customers(sample_sales_df, n=2)
    assert len(top2) == 2
    assert "Customer Alpha" in top2.index or "Customer Beta" in top2.index


def test_get_quantity_distribution(sample_sales_df):
    """Tests extraction of non-null quantity series."""
    qty = analyzer.get_quantity_distribution(sample_sales_df)
    assert len(qty) == len(sample_sales_df)
    assert qty.min() >= 1


def test_get_revenue_vs_profit(sample_sales_df):
    """Tests revenue and profit pairing and correlation coefficient calculation."""
    rev, prof, r = analyzer.get_revenue_vs_profit(sample_sales_df)
    assert len(rev) == len(sample_sales_df)
    assert len(prof) == len(sample_sales_df)
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0


def test_get_category_revenue_share(sample_sales_df):
    """Tests that percentage share sums strictly to 100%."""
    shares = analyzer.get_category_revenue_share(sample_sales_df)
    assert pytest.approx(shares.sum(), 0.01) == 100.0


def test_get_monthly_rolling_revenue(sample_sales_df):
    """Tests 3-month rolling mean calculation."""
    rolling_df = analyzer.get_monthly_rolling_revenue(sample_sales_df, window=3)
    assert "Rolling_Avg" in rolling_df.columns
    assert len(rolling_df) == 12
    # Ensure no NaN values when min_periods=1
    assert not rolling_df["Rolling_Avg"].isna().any()

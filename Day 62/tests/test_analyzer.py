"""
Unit Tests for Analytics and KPI Calculations (analyzer.py).
"""

import pytest
import pandas as pd
from app import analyzer


def test_get_kpi_summary(mock_ecommerce_df):
    """Verifies executive KPI metrics are computed accurately."""
    kpi = analyzer.get_kpi_summary(mock_ecommerce_df)
    assert "total_revenue" in kpi
    assert "total_profit" in kpi
    assert "total_orders" in kpi
    assert "total_customers" in kpi
    assert "profit_margin" in kpi
    assert kpi["total_revenue"] == pytest.approx(mock_ecommerce_df["Revenue"].sum())
    assert kpi["total_profit"] == pytest.approx(mock_ecommerce_df["Profit"].sum())
    assert kpi["total_orders"] == 12
    assert kpi["total_customers"] == 4
    expected_margin = (kpi["total_profit"] / kpi["total_revenue"]) * 100
    assert kpi["profit_margin"] == pytest.approx(expected_margin)


def test_get_monthly_revenue(mock_ecommerce_df):
    """Verifies monthly revenue grouping."""
    m_rev = analyzer.get_monthly_revenue(mock_ecommerce_df)
    assert len(m_rev) == 12
    assert m_rev.sum() == pytest.approx(mock_ecommerce_df["Revenue"].sum())


def test_get_regional_revenue(mock_ecommerce_df):
    """Verifies regional aggregation and descending sorting."""
    reg = analyzer.get_regional_revenue(mock_ecommerce_df)
    assert len(reg) == 4
    assert reg.iloc[0] >= reg.iloc[1] >= reg.iloc[2] >= reg.iloc[3]


def test_get_category_revenue(mock_ecommerce_df):
    """Verifies category aggregation and descending sorting."""
    cat = analyzer.get_category_revenue(mock_ecommerce_df)
    assert len(cat) == 3
    assert cat.iloc[0] >= cat.iloc[1] >= cat.iloc[2]


def test_get_top_products(mock_ecommerce_df):
    """Verifies top N products extraction."""
    top2 = analyzer.get_top_products(mock_ecommerce_df, n=2)
    assert len(top2) == 2
    assert top2.iloc[0] <= top2.iloc[1]


def test_get_revenue_vs_profit(mock_ecommerce_df):
    """Verifies paired revenue and profit vectors and correlation float."""
    rev, prof, r = analyzer.get_revenue_vs_profit(mock_ecommerce_df)
    assert len(rev) == len(mock_ecommerce_df)
    assert len(prof) == len(mock_ecommerce_df)
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0


def test_get_quantity_distribution(mock_ecommerce_df):
    """Verifies non-null quantity series."""
    qty = analyzer.get_quantity_distribution(mock_ecommerce_df)
    assert len(qty) == len(mock_ecommerce_df)
    assert qty.min() >= 1


def test_get_monthly_rolling_revenue(mock_ecommerce_df):
    """Verifies 3-month rolling average calculation."""
    res = analyzer.get_monthly_rolling_revenue(mock_ecommerce_df, window=3)
    assert "Rolling_Avg" in res.columns
    assert len(res) == 12
    assert not res["Rolling_Avg"].isna().any()
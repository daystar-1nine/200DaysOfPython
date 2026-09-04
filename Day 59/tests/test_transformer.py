"""
Unit Tests for app/transformer.py module.
"""

from app.cleaner import clean_sales_data
from app.transformer import compute_derived_metrics


def test_compute_derived_metrics_financials(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)

    row0 = enriched_df.iloc[0]
    expected_rev = round(2 * 60000.0 * 0.90, 2)  # 108000.0
    expected_cost = round(2 * 45000.0, 2)        # 90000.0
    expected_profit = round(expected_rev - expected_cost, 2)  # 18000.0

    assert row0["Revenue"] == expected_rev
    assert row0["Cost"] == expected_cost
    assert row0["Profit"] == expected_profit


def test_compute_derived_metrics_dates(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)

    assert "Year" in enriched_df.columns
    assert "Month" in enriched_df.columns
    assert "Month_Name" in enriched_df.columns
    assert enriched_df["Year"].iloc[0] == 2026
    assert enriched_df["Month"].iloc[0] == 1
    assert enriched_df["Month_Name"].iloc[0] == "January"


def test_compute_derived_metrics_profit_margin(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)

    assert "Profit_Margin_%" in enriched_df.columns
    assert (enriched_df["Profit_Margin_%"] >= -100.0).all()


def test_compute_derived_metrics_positive_cost(sample_raw_sales_df):
    clean_df, _ = clean_sales_data(sample_raw_sales_df)
    enriched_df = compute_derived_metrics(clean_df)

    assert (enriched_df["Cost"] >= 0).all()

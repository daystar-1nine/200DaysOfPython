"""
Unit Tests for app/analysis/category.py module.
"""

from app.analysis.category import analyze_categories
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_analyze_categories_ranking(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cat_df = analyze_categories(trans_df)

    assert "Rank" in cat_df.columns
    assert cat_df["Rank"].iloc[0] == 1


def test_analyze_categories_profit_margin(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cat_df = analyze_categories(trans_df)

    assert "profit_margin_%" in cat_df.columns
    assert (cat_df["profit_margin_%"] >= -100.0).all()


def test_analyze_categories_revenue_share(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    cat_df = analyze_categories(trans_df)

    assert "Revenue_Share_%" in cat_df.columns
    total_share = cat_df["Revenue_Share_%"].sum()
    assert 99.0 <= total_share <= 101.0

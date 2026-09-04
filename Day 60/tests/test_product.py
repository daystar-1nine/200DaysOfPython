"""
Unit Tests for app/analysis/product.py module.
"""

from app.analysis.product import analyze_products
from app.cleaner import clean_sales_records
from app.transformer import transform_sales_data


def test_analyze_products_returns_quad(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    full_prod, top_rev, top_qty, top_prof = analyze_products(trans_df)

    assert len(top_rev) <= 10
    assert len(top_qty) <= 10
    assert len(top_prof) <= 10


def test_analyze_products_category_rankings(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    full_prod, _, _, _ = analyze_products(trans_df)

    assert "Category_Rank" in full_prod.columns
    assert "Overall_Rank" in full_prod.columns
    assert full_prod["Overall_Rank"].iloc[0] == 1


def test_analyze_products_aggregations(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    full_prod, _, _, _ = analyze_products(trans_df)

    assert "total_quantity" in full_prod.columns
    assert "avg_unit_price" in full_prod.columns
    assert (full_prod["total_quantity"] > 0).all()

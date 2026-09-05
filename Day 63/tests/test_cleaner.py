"""
Tests for Data Cleaner Module
=============================
"""

import pandas as pd
import numpy as np
from app.cleaner import clean_sales_data

def test_clean_sales_data_datetime_conversion(sample_sales_df):
    cleaned = clean_sales_data(sample_sales_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Order_Date"])

def test_clean_sales_data_numeric_coercion(sample_sales_df):
    sample_sales_df["Revenue"] = sample_sales_df["Revenue"].astype(str)
    cleaned = clean_sales_data(sample_sales_df)
    assert np.issubdtype(cleaned["Revenue"].dtype, np.number)

def test_clean_sales_data_customer_segment_derivation(sample_sales_df):
    cleaned = clean_sales_data(sample_sales_df)
    assert "Customer_Segment" in cleaned.columns
    assert set(cleaned["Customer_Segment"].unique()).issubset({"Consumer", "Corporate", "Home Office"})

def test_clean_sales_data_discount_percent_calculation(sample_sales_df):
    cleaned = clean_sales_data(sample_sales_df)
    assert "Discount_Percent" in cleaned.columns
    np.testing.assert_allclose(cleaned["Discount_Percent"], cleaned["Discount"] * 100.0)

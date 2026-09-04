"""
Unit Tests for app/insights.py and app/reports.py modules.
"""

from app.analysis.category import analyze_categories
from app.analysis.customer import analyze_customers
from app.analysis.outliers import detect_outliers_iqr
from app.analysis.overview import compute_overview_kpis
from app.analysis.product import analyze_products
from app.analysis.regional import analyze_regions
from app.analysis.statistics import compute_correlations, compute_numerical_statistics
from app.analysis.time_series import analyze_monthly_series
from app.cleaner import clean_sales_records
from app.insights import generate_business_insights
from app.reports import generate_data_quality_report, generate_executive_summary
from app.transformer import transform_sales_data
from app.validator import validate_sales_data


def test_generate_business_insights_count(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    overview = compute_overview_kpis(trans_df)
    regional = analyze_regions(trans_df)
    category = analyze_categories(trans_df)
    customer = analyze_customers(trans_df)
    _, top_rev, _, _ = analyze_products(trans_df)
    monthly = analyze_monthly_series(trans_df)
    corr, _ = compute_correlations(trans_df)
    outliers = detect_outliers_iqr(trans_df)

    insights = generate_business_insights(
        overview, regional, category, customer,
        top_rev, monthly, corr, outliers
    )

    assert isinstance(insights, list)
    assert len(insights) >= 8


def test_generate_business_insights_dynamic_content(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    overview = compute_overview_kpis(trans_df)
    regional = analyze_regions(trans_df)
    category = analyze_categories(trans_df)
    customer = analyze_customers(trans_df)
    _, top_rev, _, _ = analyze_products(trans_df)
    monthly = analyze_monthly_series(trans_df)
    corr, _ = compute_correlations(trans_df)
    outliers = detect_outliers_iqr(trans_df)

    insights = generate_business_insights(
        overview, regional, category, customer,
        top_rev, monthly, corr, outliers
    )

    combined_text = " ".join(insights)
    assert regional.iloc[0]["Region"] in combined_text
    assert category.iloc[0]["Category"] in combined_text


def test_generate_executive_summary_contains_kpis(sample_raw_sales_df):
    clean_df, _ = clean_sales_records(sample_raw_sales_df)
    trans_df = transform_sales_data(clean_df)
    overview = compute_overview_kpis(trans_df)
    regional = analyze_regions(trans_df)
    category = analyze_categories(trans_df)
    customer = analyze_customers(trans_df)
    _, top_rev, _, _ = analyze_products(trans_df)
    monthly = analyze_monthly_series(trans_df)
    stats_dict = compute_numerical_statistics(trans_df)
    insights = ["Insight 1", "Insight 2"]

    rpt = generate_executive_summary(
        overview, regional, category, top_rev,
        customer, monthly, stats_dict, insights
    )

    assert "EXECUTIVE BUSINESS INTELLIGENCE ANALYTICS REPORT" in rpt
    assert str(overview["total_orders"]) in rpt


def test_generate_data_quality_report(sample_raw_sales_df):
    clean_df, audit = clean_sales_records(sample_raw_sales_df)
    val_res = validate_sales_data(clean_df)
    trans_df = transform_sales_data(clean_df)
    outliers = detect_outliers_iqr(trans_df)

    dqr = generate_data_quality_report(audit, val_res, outliers)
    assert "DATA QUALITY & DOMAIN VALIDATION REPORT" in dqr
    assert "unique_order_ids" in dqr

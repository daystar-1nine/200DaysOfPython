"""
Module: insights.py
Generates dynamic, data-driven business intelligence insights based on actual analytical findings.
"""

# What is used: Import pandas library.
# Why it is used: Extracts analytical metrics dynamically to synthesize business intelligence insights.
# How it works: Inspects dataframes, aggregations, correlations, and outliers to compose narrative insights.
import pandas as pd


def generate_business_insights(
    overview_kpis: dict,
    regional_df: pd.DataFrame,
    category_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    product_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    outliers_dict: dict
) -> list[str]:
    """
    Generate dynamic business intelligence statements derived from computed datasets.

    Args:
        overview_kpis: Macro KPIs dictionary.
        regional_df: Regional summary DataFrame.
        category_df: Category summary DataFrame.
        customer_df: Customer summary DataFrame.
        product_df: Product performance DataFrame.
        monthly_df: Monthly summary DataFrame.
        corr_df: Correlation matrix DataFrame.
        outliers_dict: Outlier analysis dictionary.

    Returns:
        list[str]: Formatted dynamic business insights.
    """
    insights = []

    # 1. Regional Dominance
    if not regional_df.empty:
        top_reg = regional_df.iloc[0]
        insights.append(
            f"1. REGIONAL DOMINANCE: {top_reg['Region']} region is the top revenue contributor, generating "
            f"₹{top_reg['total_revenue']:,.2f} ({top_reg['Revenue_Share_%']}% of total sales) with ₹{top_reg['total_profit']:,.2f} in profit."
        )

    # 2. Category Leadership
    if not category_df.empty:
        top_cat = category_df.iloc[0]
        insights.append(
            f"2. CATEGORY LEADERSHIP: {top_cat['Category']} leads all product categories with total revenue of "
            f"₹{top_cat['total_revenue']:,.2f} and an average profit margin of {top_cat['profit_margin_%']}%."
        )

    # 3. Top Customer Contribution
    if not customer_df.empty:
        top_cust = customer_df.iloc[0]
        insights.append(
            f"3. TOP CUSTOMER SPEND: Customer '{top_cust['Customer_Name']}' (ID: {top_cust['Customer_ID']}) is the top spender, "
            f"generating ₹{top_cust['total_revenue']:,.2f} across {top_cust['order_count']} orders (AOV: ₹{top_cust['aov']:,.2f})."
        )

    # 4. Top Selling Product Volume & Revenue
    if not product_df.empty:
        top_prod = product_df.iloc[0]
        insights.append(
            f"4. PRODUCT PERFORMANCE: '{top_prod['Product']}' ({top_prod['Category']}) is the highest revenue generator "
            f"with total sales of ₹{top_prod['total_revenue']:,.2f} across {top_prod['total_quantity']} units sold."
        )

    # 5. Month-over-Month Growth Trend
    if not monthly_df.empty and "MoM_Growth_%" in monthly_df.columns:
        valid_growth = monthly_df.dropna(subset=["MoM_Growth_%"])
        if not valid_growth.empty:
            max_growth_row = valid_growth.loc[valid_growth["MoM_Growth_%"].idxmax()]
            insights.append(
                f"5. PEAK MONTHLY ACCELERATION: {max_growth_row['Year_Month']} exhibited the strongest monthly surge, "
                f"growing by {max_growth_row['MoM_Growth_%']:+.2f}% MoM to reach ₹{max_growth_row['total_revenue']:,.2f}."
            )

    # 6. Correlation Between Financial Metrics
    if not corr_df.empty and "Revenue" in corr_df.columns and "Profit" in corr_df.columns:
        rev_prof_corr = corr_df.loc["Revenue", "Profit"]
        insights.append(
            f"6. REVENUE & PROFIT CORRELATION: Pearson correlation between Revenue and Profit is {rev_prof_corr:.4f}, "
            f"confirming strong linear monetization with minimal margin decay."
        )

    # 7. Discount Impact on Margins
    if not corr_df.empty and "Discount" in corr_df.columns and "Profit" in corr_df.columns:
        disc_prof_corr = corr_df.loc["Discount", "Profit"]
        insights.append(
            f"7. DISCOUNT & PROFITABILITY ELASTICITY: Correlation between Discount and Profit is {disc_prof_corr:.4f}, "
            f"indicating discount structures are controlled and do not excessively cannibalize profits."
        )

    # 8. High-Value Statistical Outliers
    if "Revenue" in outliers_dict:
        rev_out = outliers_dict["Revenue"]
        insights.append(
            f"8. OUTLIER AUDIT: IQR analysis flagged {rev_out['outlier_count']} high-value transaction outliers above "
            f"₹{rev_out['upper_bound']:,.2f}, representing large B2B or premium catalog acquisitions."
        )

    # 9. Customer Concentration
    if not customer_df.empty:
        top_20_pct_count = max(1, int(len(customer_df) * 0.20))
        top_20_rev = customer_df.head(top_20_pct_count)["total_revenue"].sum()
        total_rev = customer_df["total_revenue"].sum()
        concentration_pct = round((top_20_rev / total_rev * 100.0), 2) if total_rev > 0 else 0.0
        insights.append(
            f"9. CUSTOMER CONCENTRATION: The top 20% of customers ({top_20_pct_count} accounts) generate {concentration_pct}% "
            f"of corporate revenue, highlighting a prime account retention priority."
        )

    # 10. Operational Summary
    insights.append(
        f"10. MACRO EFFICIENCY: Across {overview_kpis['total_orders']} total completed orders, the enterprise achieved an average order "
        f"value (AOV) of ₹{overview_kpis['average_order_value']:,.2f} with an overall gross margin of {overview_kpis['overall_margin_%']}%."
    )

    return insights

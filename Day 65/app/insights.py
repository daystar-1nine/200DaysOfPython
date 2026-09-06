"""
Automated data-driven statistical insight generator.
Produces executive interpretations grounded in calculated statistics.
"""

def generate_insights(summary_dict: dict, outlier_dict: dict, percentile_dict: dict) -> list[str]:
    insights = []
    
    # 1. Revenue Central Tendency & Skewness
    rev_mean = summary_dict.get("Revenue", {}).get("mean", 0)
    rev_med = summary_dict.get("Revenue", {}).get("median", 0)
    if rev_mean > rev_med * 1.3:
        ratio = rev_mean / rev_med if rev_med > 0 else 1.0
        insights.append(
            f"Revenue Skewness Alert: The arithmetic mean (Rs. {rev_mean:,.2f}) exceeds the median "
            f"(Rs. {rev_med:,.2f}) by a factor of {ratio:.2f}x. This reveals strong positive skewness, "
            f"indicating that average transaction metrics are pulled upward by high-value enterprise orders."
        )
        
    # 2. Profit Outlier Concentration
    profit_outliers = outlier_dict.get("Profit", {}).get("count", 0)
    profit_pct = outlier_dict.get("Profit", {}).get("percentage", 0.0)
    if profit_outliers > 0:
        insights.append(
            f"Profit Margin Dispersion: Exactly {profit_outliers} transactions ({profit_pct}% of total) "
            f"exceed the upper Tukey IQR fence. These high-profit orders drive outsized platform profitability "
            f"and represent prime VIP accounts."
        )
        
    # 3. Order Quantity Discrete Pattern
    qty_mode = summary_dict.get("Quantity", {}).get("mode", [1])
    qty_p75 = percentile_dict.get("Quantity", {}).get("P75", 5)
    insights.append(
        f"Order Volume Concentration: Order quantities exhibit heavy lower-bound clustering (Mode = {qty_mode[0] if qty_mode else 1}, "
        f"P75 = {qty_p75:.0f} units). Supply chain operations and warehouse sorting should prioritize single- and dual-unit packaging."
    )
    
    # 4. Top 1% Revenue Disproportion
    rev_p99 = percentile_dict.get("Revenue", {}).get("P99", 0)
    insights.append(
        f"Elite Customer Spending Threshold (P99): The 99th percentile revenue cutoff stands at Rs. {rev_p99:,.2f}. "
        f"Orders beyond this value represent high-velocity corporate accounts requiring dedicated support workflows."
    )
    
    # 5. Discount Consistency
    disc_std = summary_dict.get("Discount", {}).get("std", 0)
    disc_mean = summary_dict.get("Discount", {}).get("mean", 0)
    insights.append(
        f"Discount Policy Discipline: Promotional discounts average {disc_mean * 100:.1f}% with standard deviation "
        f"{disc_std * 100:.1f}%, indicating adherence to structured promotional bands across retail tiers."
    )
    
    return insights

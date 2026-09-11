"""
Financial impact and business translation modeling for A/B testing lifts.
"""

def project_annual_business_impact(
    absolute_lift: float,
    aov: float,
    monthly_visitors: int = 500000,
    cost_to_deploy: float = 25000.0
) -> dict:
    annual_visitors = monthly_visitors * 12
    incremental_conversions = int(round(annual_visitors * absolute_lift))
    projected_revenue = incremental_conversions * aov
    net_profit = projected_revenue - cost_to_deploy
    roi_pct = (net_profit / cost_to_deploy * 100.0) if cost_to_deploy > 0 else 0.0
    
    return {
        "monthly_visitors": monthly_visitors,
        "annual_visitors": annual_visitors,
        "absolute_conversion_lift": round(absolute_lift, 4),
        "average_order_value": round(aov, 2),
        "annual_incremental_conversions": incremental_conversions,
        "projected_annual_gross_revenue": round(projected_revenue, 2),
        "deployment_cost": round(cost_to_deploy, 2),
        "projected_net_annual_profit": round(net_profit, 2),
        "projected_roi_pct": round(roi_pct, 2)
    }

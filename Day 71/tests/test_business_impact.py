"""
Unit tests for business impact financial projection model.
"""

import pytest

try:
    from app.business_impact import project_annual_business_impact
except (ImportError, ModuleNotFoundError):
    from business_impact import project_annual_business_impact

def test_project_annual_business_impact():
    impact = project_annual_business_impact(
        absolute_lift=0.005,  # +0.5 pp
        aov=60.0,
        monthly_visitors=100000,
        cost_to_deploy=10000.0
    )
    # Annual visitors = 1,200,000
    # Conversions = 1,200,000 * 0.005 = 6,000
    # Gross Revenue = 6,000 * 60 = 360,000
    # Net Profit = 360,000 - 10,000 = 350,000
    assert impact["annual_incremental_conversions"] == 6000
    assert impact["projected_annual_gross_revenue"] == 360000.0
    assert impact["projected_net_annual_profit"] == 350000.0
    assert impact["projected_roi_pct"] == 3500.0

"""Tests for insight generator module."""
from app.insights import generate_insights

def test_generate_insights_triggers():
    summary = {
        "Revenue": {"mean": 80000.0, "median": 30000.0, "std": 110000.0},
        "Discount": {"mean": 0.10, "std": 0.07},
        "Quantity": {"mode": [1]}
    }
    outliers = {"Profit": {"count": 15, "percentage": 5.0}}
    percentiles = {
        "Revenue": {"P99": 500000.0},
        "Quantity": {"P75": 5.0}
    }
    insights = generate_insights(summary, outliers, percentiles)
    assert len(insights) >= 4
    assert any("Revenue Skewness Alert" in ins for ins in insights)
    assert any("Profit Margin Dispersion" in ins for ins in insights)

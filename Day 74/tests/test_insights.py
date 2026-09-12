import pytest
import pandas as pd

try:
    from app.insights import generate_business_insights
except ImportError:
    from insights import generate_business_insights

@pytest.fixture
def mock_coefs():
    return pd.DataFrame({
        'Feature': ['TV_Spend', 'Digital_Spend', 'Discount'],
        'Coefficient': [3.5, 2.1, -15.0]
    })

def test_insights_returns_list(mock_coefs):
    insights = generate_business_insights(mock_coefs)
    assert isinstance(insights, list)

def test_insights_not_empty(mock_coefs):
    insights = generate_business_insights(mock_coefs)
    assert len(insights) > 0

def test_insights_no_causal_language(mock_coefs):
    insights = generate_business_insights(mock_coefs)
    for insight in insights:
        assert 'causes' not in insight.lower()
        assert 'caused by' not in insight.lower()

def test_insights_contains_strings(mock_coefs):
    insights = generate_business_insights(mock_coefs)
    for insight in insights:
        assert isinstance(insight, str)

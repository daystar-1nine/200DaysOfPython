"""
Unit Tests for Custom Formatters (formatters.py).
"""

import pytest
from app.formatters import (
    format_currency,
    format_compact_inr,
    format_large_number,
    format_percentage
)


def test_format_currency():
    """Tests standard Indian Rupee formatting."""
    assert format_currency(25000) == "₹25,000"
    assert format_currency(1234567) == "₹1,234,567"
    assert format_currency(0) == "₹0"
    assert format_currency(None) == "₹0"


def test_format_compact_inr():
    """Tests compact denomination conversions (Cr, L, K)."""
    assert format_compact_inr(15000000) == "₹1.50 Cr"
    assert format_compact_inr(524000) == "₹5.2 L"
    assert format_compact_inr(45000) == "₹45 K"
    assert format_compact_inr(850) == "₹850"
    assert format_compact_inr(None) == "₹0"


def test_format_large_number():
    """Tests integer comma separation."""
    assert format_large_number(1250) == "1,250"
    assert format_large_number(1000000) == "1,000,000"
    assert format_large_number(5) == "5"
    assert format_large_number(None) == "0"


def test_format_percentage():
    """Tests ratio and percentage formatting."""
    assert format_percentage(24.56) == "24.6%"
    assert format_percentage(0.0) == "0.0%"
    assert format_percentage(100.0) == "100.0%"
    assert format_percentage(None) == "0.0%"
"""
Custom Formatting Utility Module for Matplotlib Tickers and KPI Tiles.
Provides Indian currency (INR), compact denomination (Lakh/Crore), and percentage formatters.
"""


def format_currency(value: float, pos=None) -> str:
    """
    Formats a numeric value as a standard Indian Rupee currency string with commas.

    # What is used: f-string formatting with comma separator and ₹ symbol
    # Why it is used: Transforms raw integers into professional financial labels
    # How it works: Formats float with comma groupers; prefixes with ₹
    """
    if value is None:
        return "₹0"
    return f"₹{value:,.0f}"


def format_compact_inr(value: float, pos=None) -> str:
    """
    Formats large currency amounts into compact Indian denominations (K, L, Cr).

    # What is used: Scaled arithmetic formatting
    # Why it is used: Keeps axis labels concise and prevents horizontal crowding
    # How it works: Divides by 1e7 for Crores, 1e5 for Lakhs, 1e3 for Thousands
    """
    if value is None:
        return "₹0"
    val = float(value)
    abs_val = abs(val)
    if abs_val >= 1e7:
        return f"₹{val * 1e-7:.2f} Cr"
    elif abs_val >= 1e5:
        return f"₹{val * 1e-5:.1f} L"
    elif abs_val >= 1e3:
        return f"₹{val * 1e-3:.0f} K"
    return f"₹{val:,.0f}"


def format_large_number(value: float, pos=None) -> str:
    """
    Formats a scalar count with thousands comma separators.

    # What is used: Integer formatting with commas
    # Why it is used: Enhances legibility for order and customer counts
    # How it works: Converts value to integer and inserts commas
    """
    if value is None:
        return "0"
    return f"{int(value):,}"


def format_percentage(value: float, pos=None) -> str:
    """
    Formats a ratio or percentage float into a clean percent string.

    # What is used: Percentage string formatting
    # Why it is used: Standardizes margin and growth indicators
    # How it works: Formats to one decimal place with % suffix
    """
    if value is None:
        return "0.0%"
    return f"{value:.1f}%"
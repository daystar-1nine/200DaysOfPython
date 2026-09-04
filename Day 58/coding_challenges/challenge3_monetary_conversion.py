"""
Day 58 - Coding Challenge 3: Multi-Currency String Cleaning to Numeric Float
Clean currency strings containing ₹, $, commas, and whitespace into numeric floats.
"""

# What is used: Import sys and pandas library.
# Why it is used: Core packages for system encoding configuration, string clean-up and numeric type conversion.
# How it works: Brings sys and pandas namespaces into module scope.
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def convert_currency_to_float(series: pd.Series) -> pd.Series:
    """
    Clean currency string Series (₹, $, commas, whitespace) into numeric floats.

    Args:
        series: Input Series containing currency string values.

    Returns:
        pd.Series: Parsed numeric float Series with NaNs for unparseable entries.
    """
    # What is used: Series.astype(str).str.replace() chain.
    # Why it is used: Strips out currency symbols (₹, $) and comma separators.
    # How it works: Replaces currency formatting tokens with empty strings.
    clean_str = (
        series.astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # What is used: pd.to_numeric(clean_str, errors="coerce").
    # Why it is used: Converts valid numeric strings to float, replacing invalid entries with NaN.
    # How it works: Coerces unparseable values to float NaNs.
    return pd.to_numeric(clean_str, errors="coerce")


if __name__ == "__main__":
    # What is used: Series containing messy multi-currency strings.
    # Why it is used: Test input for currency parsing function.
    # How it works: Holds ₹50,000, $1,200, 60000, ₹75,500, unknown.
    raw_prices = pd.Series(["₹50,000", "$1,200", "60000", "₹75,500", "unknown"])

    # What is used: Calling convert_currency_to_float.
    # Why it is used: Executes currency cleaning pipeline on Series.
    # How it works: Displays parsed float Series.
    numeric_prices = convert_currency_to_float(raw_prices)
    print("--- Raw Currency Inputs ---")
    print(raw_prices)

    print("\n--- Parsed Float Series ---")
    print(numeric_prices)

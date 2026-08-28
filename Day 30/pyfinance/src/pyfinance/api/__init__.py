"""PyFinance API Integration Package."""
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.api.client import CurrencyAPIClient

__all__ = ["CurrencyAPIClient"]

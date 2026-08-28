# ==============================================================================
# Program    : Currency Conversion Service
# Objective  : Business logic for live currency conversion and exchange rates.
# Concept    : API Service Abstraction
# Why Used   : Connects CLI commands to CurrencyAPIClient.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.api.client import CurrencyAPIClient
from pyfinance.models.expense import CurrencyRate

class CurrencyService:
    def __init__(self, api_client: CurrencyAPIClient | None = None):
        self.api_client = api_client or CurrencyAPIClient()

    def get_exchange_rate(self, base: str, target: str) -> CurrencyRate:
        base_upper = base.upper()
        target_upper = target.upper()
        rate = self.api_client.fetch_exchange_rate(base_upper, target_upper)
        return CurrencyRate(base=base_upper, target=target_upper, rate=rate, updated_at="Live Rates API")

    def convert_amount(self, amount: float, base: str, target: str) -> float:
        rate_obj = self.get_exchange_rate(base, target)
        return amount * rate_obj.rate

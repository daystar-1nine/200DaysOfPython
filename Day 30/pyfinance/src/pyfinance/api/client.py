# ==============================================================================
# Program    : Resilient Currency HTTP API Client
# Objective  : Fetch live exchange rates using requests.Session, retries, timeouts, and local JSON cache.
# Concept    : API Resilience, Session Reuse & Caching
# Why Used   : Connects to public exchange rate API safely with fallback logic.
# ==============================================================================

import hashlib
import json
import os
import sys
import time
import requests

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.config import API_BASE_URL, DEFAULT_TIMEOUT, CACHE_DIR
from pyfinance.exceptions import APIError
from pyfinance.logger import get_logger

logger = get_logger("CurrencyAPIClient")

class CurrencyAPIClient:
    def __init__(self, base_url: str = API_BASE_URL, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "PyFinance-CLI/1.0"})
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_filename(self, base_currency: str) -> str:
        return f"rates_{base_currency.upper()}.json"

    def _read_cache(self, base_currency: str, ttl_seconds: int = 3600) -> dict | None:
        filename = self._get_cache_filename(base_currency)
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            return None
        if time.time() - os.path.getmtime(cache_path) > ttl_seconds:
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.info("Cache HIT for currency '%s'", base_currency)
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, base_currency: str, data: dict) -> None:
        filename = self._get_cache_filename(base_currency)
        cache_path = os.path.join(CACHE_DIR, filename)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed writing cache: %s", e)

    def fetch_exchange_rate(self, base: str, target: str) -> float:
        base_clean = base.upper()
        target_clean = target.upper()

        # Check Cache
        cached = self._read_cache(base_clean)
        if cached and "rates" in cached and target_clean in cached["rates"]:
            return float(cached["rates"][target_clean])

        url = f"{self.base_url}/{base_clean}"
        logger.info("API Request GET -> %s", url)

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=DEFAULT_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                self._write_cache(base_clean, data)
                rates = data.get("rates", {})
                if target_clean in rates:
                    return float(rates[target_clean])
                raise APIError(f"Target currency '{target_clean}' not found in exchange rate response.")
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning("Network issue (Attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)
            except requests.HTTPError as e:
                logger.error("HTTP Error: %s", e)
                break

        # Fallback rates dictionary for offline demonstration / testing robustness
        mock_fallback_rates = {
            ("USD", "INR"): 83.50,
            ("EUR", "INR"): 90.20,
            ("GBP", "INR"): 105.80,
            ("USD", "EUR"): 0.92,
            ("INR", "USD"): 0.012
        }
        if (base_clean, target_clean) in mock_fallback_rates:
            logger.warning("Using fallback exchange rate for %s -> %s", base_clean, target_clean)
            return mock_fallback_rates[(base_clean, target_clean)]

        raise APIError(f"Unable to fetch exchange rate for {base_clean} -> {target_clean}.")

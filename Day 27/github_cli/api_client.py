# ==============================================================================
# Program    : Reusable HTTP API Client with Retries & File Caching
# Objective  : Build resilient HTTP client supporting requests.Session, retries, exponential backoff, and local JSON cache.
# Concept    : API Resilience, Connection Reuse (requests.Session) & Response Caching
# Why Used   : Encapsulates GET/POST calls, handles status codes, rate limits, and caching.
# ==============================================================================

import hashlib
import json
import os
import time
import requests
from config import GITHUB_API_BASE_URL, GITHUB_TOKEN, DEFAULT_TIMEOUT, CACHE_DIR
from logger import get_logger

logger = get_logger("APIClient")

class APIClient:
    def __init__(self, base_url: str = GITHUB_API_BASE_URL, token: str | None = GITHUB_TOKEN, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.session = requests.Session()
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Python-GitHub-CLI/2.0"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        self.session.headers.update(headers)
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_key(self, endpoint: str, params: dict | None) -> str:
        raw_key = f"{endpoint}?{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest() + ".json"

    def _read_cache(self, cache_file: str, ttl_seconds: int = 300) -> dict | None:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        if not os.path.exists(cache_path):
            return None
        # Check TTL
        if time.time() - os.path.getmtime(cache_path) > ttl_seconds:
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Cache HIT for '%s'", cache_file)
                return data
        except Exception:
            return None

    def _write_cache(self, cache_file: str, data: dict) -> None:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed writing cache: %s", e)

    def get(self, endpoint: str, params: dict | None = None, use_cache: bool = True) -> dict:
        cache_file = self._get_cache_key(endpoint, params)
        if use_cache:
            cached_data = self._read_cache(cache_file)
            if cached_data is not None:
                return cached_data

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info("API Request GET -> %s | Params: %s", url, params)

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if use_cache:
                    self._write_cache(cache_file, data)
                return data
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning("Network issue (Attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except requests.HTTPError as e:
                logger.error("HTTP Error %d: %s", resp.status_code, e)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    logger.warning("Rate limit hit. Waiting %d seconds...", retry_after)
                    time.sleep(retry_after)
                else:
                    raise

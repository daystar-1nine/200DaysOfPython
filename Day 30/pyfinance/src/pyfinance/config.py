# ==============================================================================
# Program    : PyFinance Centralized Configuration System
# Objective  : Centralize database paths, API endpoints, and logging configurations.
# Concept    : Configuration Management via Environment Variables
# Why Used   : Prevents hardcoding file paths and API keys in source code.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATABASE_PATH: str = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "data", "pyfinance.db"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_PATH: str = os.path.join(BASE_DIR, "logs", "pyfinance.log")
CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://open.er-api.com/v6/latest")
DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "10"))

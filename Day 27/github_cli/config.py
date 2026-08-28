# ==============================================================================
# Program    : GitHub CLI Configuration Module
# Objective  : Load environment variables, API base URLs, and configuration settings.
# Concept    : Configuration & Environment Variable Security (.env)
# Why Used   : Encapsulates GitHub API settings and API tokens.
# ==============================================================================

import os
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

GITHUB_API_BASE_URL: str = "https://api.github.com"
GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN", None)
DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "10"))
CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "cache")

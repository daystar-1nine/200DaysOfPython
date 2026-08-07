# ==============================================================================
# Module     : Weather Dashboard Configuration Loader
# Objective  : Load API keys and environment settings securely from .env.
# Concept    : Configuration Secrets Isolation
# Why Used   : Encapsulates environment configuration reading into a config module.
# ==============================================================================

import os

def load_env_vars(env_path=".env"):
    """Reads .env file line by line."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

# Load environment configuration
load_env_vars()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "demo_weather_api_key_998877")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Mumbai")

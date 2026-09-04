# ==============================================================================
# Program    : Application Settings Configuration (config.py)
# Objective  : Load environment settings for DATABASE_URL, LOG_LEVEL, and JWT keys.
# Concept    : Configuration Management & Environment Independence
# Why Used   : Provides global configuration settings for production, testing, and logging.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ecommerce_v5.db")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "prod_secret_key_day49_python200days_super_secret_string_32_bytes")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    PAYMENT_GATEWAY_URL: str = os.getenv("PAYMENT_GATEWAY_URL", "https://api.paymentgateway.com/v1/charge")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

settings = Settings()

def get_settings() -> Settings:
    return settings

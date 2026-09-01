# ==============================================================================
# Program    : Application Settings Configuration (config.py)
# Objective  : Load DATABASE_URL, JWT_SECRET_KEY, and security settings from .env file.
# Concept    : Configuration & Environment Security Management
# Why Used   : Encapsulates JWT secrets and expiration settings securely outside git.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ecommerce_v3.db")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev_secret_key_day47_python200days_super_secret_string")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

settings = Settings()

def get_settings() -> Settings:
    return settings

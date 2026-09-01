# ==============================================================================
# Program    : Application Settings Configuration (config.py)
# Objective  : Load DATABASE_URL from .env file via python-dotenv for Alembic & FastAPI.
# Concept    : Configuration & Environment Management
# Why Used   : Provides centralized database configuration string to Alembic env.py.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ecommerce_v2.db")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

settings = Settings()

def get_settings() -> Settings:
    return settings

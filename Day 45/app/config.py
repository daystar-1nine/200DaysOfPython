# ==============================================================================
# Program    : Application Settings Configuration (config.py)
# Objective  : Load DATABASE_URL from .env file via python-dotenv.
# Concept    : Configuration & Environment Management
# Why Used   : Encapsulates database connection strings.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ecommerce.db")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

settings = Settings()

def get_settings() -> Settings:
    return settings

# ==============================================================================
# Program    : Application Configuration Settings (config.py)
# Objective  : Read DATABASE_URL and environment variables via python-dotenv.
# Concept    : Environment Variables Management (Day 44 requirement)
# Why Used   : Keeps database credentials out of hardcoded source files.
# ==============================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings class reading environment variables."""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./users_v3.db")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

settings = Settings()

def get_settings() -> Settings:
    """Dependency callable returning application settings instance."""
    return settings

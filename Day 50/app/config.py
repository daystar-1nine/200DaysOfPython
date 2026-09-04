"""
===============================================================================
DAY 50 — CENTRALIZED CONFIGURATION MODULE
===============================================================================
This module defines application settings using Pydantic BaseSettings, loading
environment variables from .env files with default fallback values.
===============================================================================
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema reading environment variables."""

    # What is used: String settings for database connection string.
    # Why it is used: Provides central configuration for SQLAlchemy engine connection.
    # How it works: Reads DATABASE_URL from .env or defaults to SQLite file.
    DATABASE_URL: str = "sqlite:///./taskflow.db"

    # What is used: Security configuration attributes.
    # Why it is used: Configures JWT secret signing key, algorithm, and token TTL.
    # How it works: Passed into PyJWT encode and decode utility functions.
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # What is used: Observability and environment flags.
    # Why it is used: Controls log level severity and execution environment label.
    # How it works: Passed to logging configuration setup.
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    # What is used: Pydantic SettingsConfigDict configuration.
    # Why it is used: Instructs Pydantic to read settings from .env file automatically.
    # How it works: Specifies env_file = ".env" and case_sensitive = True.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# What is used: Singleton Settings instance instantiation.
# Why it is used: Shares a single cached settings instance across the application lifecycle.
# How it works: Instantiates Settings object at module import time.
settings = Settings()

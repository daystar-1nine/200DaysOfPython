"""
===============================================================================
DAY 50 — DATABASE CONNECTION ENGINE & SESSION PROVIDER
===============================================================================
This module initializes the SQLAlchemy 2.0 engine, declarative base class,
sessionmaker, and the get_db generator dependency for FastAPI route handlers.
===============================================================================
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.config import settings

# What is used: Conditional connect_args check for SQLite database engine.
# Why it is used: Disables same-thread check for SQLite in multi-threaded ASGI environment.
# How it works: Checks if DATABASE_URL starts with "sqlite" and passes check_same_thread=False.
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

# What is used: SQLAlchemy create_engine initialization.
# Why it is used: Manages low-level database connection pool and SQL dialect translation.
# How it works: Connects using settings.DATABASE_URL.
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, echo=False)

# What is used: sessionmaker factory configuration.
# Why it is used: Spawns new transactional database session instances per HTTP request context.
# How it works: Binds sessions to engine with autocommit=False and autoflush=False.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# What is used: DeclarativeBase base model class definition.
# Why it is used: Base class for all ORM models (User, Task).
# How it works: All models inherit from Base to register metadata.
class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency providing a database session per HTTP request lifecycle."""
    # What is used: Generator function with try...finally session cleanup.
    # Why it is used: Guarantees database sessions are closed cleanly after request finishes.
    # How it works: Yields SessionLocal instance to endpoint and calls db.close() in finally.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

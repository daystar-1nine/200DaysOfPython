# ==============================================================================
# Program    : Database Engine & Session Provider (database.py)
# Objective  : Configure SQLAlchemy engine, session factory, Base class, and get_db dependency generator.
# Concept    : SQLAlchemy Database Integration & Session Lifecycle (Day 44 requirement)
# Why Used   : Provides database session instances to FastAPI routes via Depends(get_db).
# ==============================================================================

import os
import sys
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import settings

# Create SQLAlchemy Database Engine
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# Create SessionLocal factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base Class for ORM models
class Base(DeclarativeBase):
    pass

# What is used : Database Session Dependency Generator (get_db)
# Why it is used: Opens database session per request and guarantees cleanup via try...finally
def get_db() -> Generator[Session, None, None]:
    """Dependency generator yielding database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

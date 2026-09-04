# ==============================================================================
# Program    : Database Connection Engine & Session Provider (database.py)
# Objective  : Configure SQLAlchemy engine, sessionmaker, Base class, and get_db session generator.
# Concept    : Database Session Integration & Readiness Checks
# Why Used   : Provides database session instances and readiness health check capabilities.
# ==============================================================================

import os
import sys
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import settings

connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_database_readiness() -> bool:
    """Execute lightweight query (SELECT 1) to verify database connectivity for readiness probe."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

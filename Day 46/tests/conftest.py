# ==============================================================================
# Pytest Configuration Fixtures (conftest.py)
# Objective  : Shared file-backed test database engine, Alembic migrations runner, and TestClient fixture.
# Concept    : Pytest Shared Test Fixtures & Migration Integration
# Why Used   : Runs Alembic migrations against test_ecommerce_v2.db and disposes engine cleanly.
# ==============================================================================

import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base, get_db
import app.models  # noqa: F401
from app.main import app

test_db_path = os.path.join(src_dir, "test_ecommerce_v2.db")
test_db_url = f"sqlite:///{test_db_path}"

engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="session", autouse=True)
def apply_migrations():
    """Apply Alembic migrations to test database prior to test execution."""
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass

    # Initialize Alembic config for test DB
    alembic_ini_path = os.path.join(src_dir, "alembic.ini")
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", os.path.join(src_dir, "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)

    # Run upgrade head to apply all 001..006 migrations
    command.upgrade(alembic_cfg, "head")

    yield

    # Dispose engine before removing test file to prevent Windows PermissionError
    engine.dispose()
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass

@pytest.fixture(autouse=True)
def setup_db_session(apply_migrations):
    """Clean data between test runs while retaining table schemas."""
    db = TestingSessionLocal()
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in existing_tables:
                db.execute(table.delete())
        db.commit()
    finally:
        db.close()

@pytest.fixture
def client():
    return TestClient(app)

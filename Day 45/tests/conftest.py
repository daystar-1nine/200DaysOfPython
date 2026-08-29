# ==============================================================================
# Pytest Configuration Fixtures (conftest.py)
# Objective  : Shared test database engine, session override, and TestClient fixture.
# Concept    : Pytest Shared Test Fixtures & Dependency Overrides
# Why Used   : Eliminates duplicate in-memory engine conflicts across test modules.
# ==============================================================================

import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base, get_db
# Explicitly import all models so Base.metadata contains all tables
import app.models  # noqa: F401
from app.main import app

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    return TestClient(app)

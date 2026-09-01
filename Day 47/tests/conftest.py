# ==============================================================================
# Pytest Configuration Fixtures (conftest.py)
# Objective  : Shared test database engine, Alembic migrations runner (001-007), TestClient, and auth headers.
# Concept    : Test Environment Setup & Token Fixture Providers
# Why Used   : Runs Alembic migrations and provides user/admin Bearer token headers.
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
from app.models.user import User
import app.models  # noqa: F401
from app.main import app
from app.security import hash_password, create_access_token

test_db_path = os.path.join(src_dir, "test_ecommerce_v3.db")
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

    alembic_ini_path = os.path.join(src_dir, "alembic.ini")
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", os.path.join(src_dir, "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)

    command.upgrade(alembic_cfg, "head")

    yield

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

@pytest.fixture
def test_user(client):
    """Fixture providing a created normal user record."""
    db = TestingSessionLocal()
    try:
        user = User(
            name="Regular User",
            email="user@example.com",
            age=25,
            password_hash=hash_password("UserPassword123!"),
            role="user"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

@pytest.fixture
def test_admin(client):
    """Fixture providing a created admin user record."""
    db = TestingSessionLocal()
    try:
        admin = User(
            name="Admin User",
            email="admin@example.com",
            age=30,
            password_hash=hash_password("AdminPassword123!"),
            role="admin"
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        return admin
    finally:
        db.close()

@pytest.fixture
def user_token_headers(test_user):
    """Bearer authorization headers for regular authenticated user."""
    token = create_access_token({"sub": str(test_user.id), "email": test_user.email, "role": test_user.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_token_headers(test_admin):
    """Bearer authorization headers for admin user."""
    token = create_access_token({"sub": str(test_admin.id), "email": test_admin.email, "role": test_admin.role})
    return {"Authorization": f"Bearer {token}"}

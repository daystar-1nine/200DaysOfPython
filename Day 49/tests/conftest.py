# ==============================================================================
# Program    : Shared Pytest Test Fixtures & Database Setup (conftest.py)
# Objective  : Configure isolated SQLite test engine, Alembic migrations, TestClient, and auth headers.
# Concept    : Pytest Fixtures & Dependency Overriding
# Why Used   : Ensures test isolation across 60+ test cases without modifying production/dev database.
# ==============================================================================

import os
import sys
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from alembic.config import Config
from alembic import command

# Add Day 49 root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models.user import User
from app.models.product import Product
from app.security import hash_password, create_access_token

TEST_DB_FILE = os.path.join(os.path.dirname(__file__), "test_ecommerce_v5.db")
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_FILE}"

test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Session-wide fixture running Alembic migrations on isolated test database."""
    if os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except PermissionError:
            pass

    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    alembic_cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic"))

    # Execute Alembic migration history (001 -> 007)
    command.upgrade(alembic_cfg, "head")

    yield

    if os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except PermissionError:
            pass

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Function-scoped database session fixture with clean table teardown between tests."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        # Clean all data rows after each test execution to ensure total isolation
        try:
            session.execute(text("DELETE FROM order_items;"))
            session.execute(text("DELETE FROM orders;"))
            session.execute(text("DELETE FROM products;"))
            session.execute(text("DELETE FROM users;"))
            session.commit()
        except Exception:
            session.rollback()
        session.close()

@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def normal_user(db_session: Session) -> User:
    """Fixture creating a standard user account."""
    user = User(
        name="Suraj Test User",
        email="suraj.user@example.com",
        password_hash=hash_password("UserPassword123!"),
        role="user",
        age=21
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def admin_user(db_session: Session) -> User:
    """Fixture creating an admin user account."""
    admin = User(
        name="Suraj Admin User",
        email="suraj.admin@example.com",
        password_hash=hash_password("AdminPassword123!"),
        role="admin",
        age=25
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin

@pytest.fixture
def normal_user_headers(normal_user: User) -> dict:
    """Fixture generating Authorization Bearer header for regular user."""
    token = create_access_token(data={"sub": str(normal_user.id), "email": normal_user.email, "role": normal_user.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_user_headers(admin_user: User) -> dict:
    """Fixture generating Authorization Bearer header for admin user."""
    token = create_access_token(data={"sub": str(admin_user.id), "email": admin_user.email, "role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_product(db_session: Session) -> Product:
    """Fixture creating a sample catalog product."""
    product = Product(
        name="Mechanical Keyboard",
        price=120.0,
        stock=10,
        description="RGB Wireless Mechanical Keyboard",
        category="Electronics"
    )
    db_session.add(product)
    db_session.commit()
    db_session.refresh(product)
    return product

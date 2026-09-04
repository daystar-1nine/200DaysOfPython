"""
===============================================================================
DAY 50 — PYTEST FIXTURES (TEST SETUP & DEPINJECTION OVERRIDES)
===============================================================================
This module provides Pytest fixtures for in-memory SQLite test databases,
TestClient initialization, and pre-authenticated user headers.
===============================================================================
"""

import pytest
from typing import Generator, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.task import Task
from app.security import hash_password, create_access_token

# In-memory SQLite engine for isolated test runs
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Provide a clean in-memory database session per test function."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with get_db dependency override."""
    def override_get_db() -> Generator[Session, None, None]:
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
    """Fixture creating a standard user account in test database."""
    user = User(
        name="Standard User",
        email="user@example.com",
        password_hash=hash_password("password123"),
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_auth_headers(normal_user: User) -> Dict[str, str]:
    """Fixture providing Authorization headers for standard user."""
    token = create_access_token(data={"sub": str(normal_user.id), "role": normal_user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_user(db_session: Session) -> User:
    """Fixture creating an admin user account in test database."""
    user = User(
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("adminpass123"),
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_auth_headers(admin_user: User) -> Dict[str, str]:
    """Fixture providing Authorization headers for admin user."""
    token = create_access_token(data={"sub": str(admin_user.id), "role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user(db_session: Session) -> User:
    """Fixture creating a second standard user account for ownership isolation tests."""
    user = User(
        name="Other User",
        email="other@example.com",
        password_hash=hash_password("password123"),
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user_auth_headers(other_user: User) -> Dict[str, str]:
    """Fixture providing Authorization headers for second user."""
    token = create_access_token(data={"sub": str(other_user.id), "role": other_user.role})
    return {"Authorization": f"Bearer {token}"}

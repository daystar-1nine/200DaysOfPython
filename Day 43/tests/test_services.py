# ==============================================================================
# Test Suite : UserService Unit Tests (test_services.py)
# Objective  : Unit testing of UserService business logic using mocked/isolated UserRepository.
# Concept    : Unit Testing Service Layer
# Why Used   : Asserts business validation rules and exception raising.
# ==============================================================================

import os
import sys
import unittest
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserNotFoundError, UserAlreadyExistsError
from app.models.user import UserCreate, UserUpdate, UserPatch
from app.repositories.user_repository import UserRepository
from app.services.user_service import UserService

def test_service_list_users():
    repo = UserRepository()
    service = UserService(repo)
    users = service.list_users()
    assert len(users) == 4

def test_service_get_user_valid():
    repo = UserRepository()
    service = UserService(repo)
    user = service.get_user(1)
    assert user["name"] == "Suraj Sawant"

def test_service_get_user_missing_raises_error():
    repo = UserRepository()
    service = UserService(repo)
    with pytest.raises(UserNotFoundError):
        service.get_user(999)

def test_service_create_user_success():
    repo = UserRepository()
    service = UserService(repo)
    payload = UserCreate(name="Developer", email="dev@example.com", age=25)
    created = service.create_user(payload)
    assert created["id"] == 5

def test_service_create_user_duplicate_email_raises_error():
    repo = UserRepository()
    service = UserService(repo)
    payload = UserCreate(name="Duplicate", email="suraj@example.com", age=22)
    with pytest.raises(UserAlreadyExistsError):
        service.create_user(payload)

def test_service_replace_user():
    repo = UserRepository()
    service = UserService(repo)
    payload = UserUpdate(name="Suraj Replaced", email="suraj.replaced@example.com", age=22)
    updated = service.replace_user(1, payload)
    assert updated["name"] == "Suraj Replaced"

def test_service_patch_user():
    repo = UserRepository()
    service = UserService(repo)
    payload = UserPatch(age=23)
    patched = service.patch_user(1, payload)
    assert patched["age"] == 23

def test_service_delete_user():
    repo = UserRepository()
    service = UserService(repo)
    assert service.delete_user(4) is True
    with pytest.raises(UserNotFoundError):
        service.get_user(4)

class TestServiceRunner(unittest.TestCase):
    def test_service_standalone(self):
        repo = UserRepository()
        service = UserService(repo)
        self.assertIsNotNone(service)

if __name__ == "__main__":
    unittest.main()

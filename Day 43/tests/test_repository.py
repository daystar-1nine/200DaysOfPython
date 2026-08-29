# ==============================================================================
# Test Suite : UserRepository Unit Tests (test_repository.py)
# Objective  : Direct unit testing of UserRepository CRUD and query operations.
# Concept    : Unit Testing Data Access Layer
# Why Used   : Asserts in-memory repository persistence behavior.
# ==============================================================================

import os
import sys
import unittest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.repositories.user_repository import UserRepository

def test_repo_get_all_initial():
    repo = UserRepository()
    users = repo.get_all()
    assert len(users) == 4
    assert users[0]["name"] == "Suraj Sawant"

def test_repo_get_by_id_valid():
    repo = UserRepository()
    user = repo.get_by_id(1)
    assert user is not None
    assert user["email"] == "suraj@example.com"

def test_repo_get_by_id_missing():
    repo = UserRepository()
    assert repo.get_by_id(999) is None

def test_repo_get_by_email():
    repo = UserRepository()
    user = repo.get_by_email("alex@example.com")
    assert user is not None
    assert user["id"] == 2

def test_repo_search_by_name():
    repo = UserRepository()
    matches = repo.search_by_name("suraj")
    assert len(matches) == 1
    assert matches[0]["id"] == 1

def test_repo_create_user():
    repo = UserRepository()
    new_u = repo.create("New User", "new@example.com", 25)
    assert new_u["id"] == 5
    assert repo.get_by_id(5) is not None

def test_repo_update_user():
    repo = UserRepository()
    updated = repo.update(1, "Suraj Updated", "suraj.updated@example.com", 22)
    assert updated is not None
    assert updated["name"] == "Suraj Updated"

def test_repo_delete_user():
    repo = UserRepository()
    res = repo.delete(4)
    assert res is True
    assert repo.get_by_id(4) is None

class TestRepoRunner(unittest.TestCase):
    def test_repo_standalone(self):
        repo = UserRepository()
        self.assertEqual(len(repo.get_all()), 4)

if __name__ == "__main__":
    unittest.main()

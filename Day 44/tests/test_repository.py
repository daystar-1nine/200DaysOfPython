# ==============================================================================
# Test Suite : UserRepository Database Unit Tests (test_repository.py)
# Objective  : Unit testing of UserRepository using in-memory SQLite database session.
# Concept    : Database Unit Testing with SQLAlchemy Session
# Why Used   : Asserts database CRUD operations and ILIKE keyword search behavior.
# ==============================================================================

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base
from app.repositories.user_repository import UserRepository

class TestUserRepository(unittest.TestCase):
    def setUp(self):
        # Create an isolated in-memory SQLite database for each test
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.repo = UserRepository(self.db)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_repo_create_user(self):
        user = self.repo.create("Suraj Sawant", "suraj@example.com", 21)
        self.assertIsNotNone(user.id)
        self.assertEqual(user.name, "Suraj Sawant")
        self.assertEqual(user.email, "suraj@example.com")

    def test_repo_get_by_id(self):
        created = self.repo.create("Alex Mercer", "alex@example.com", 25)
        found = self.repo.get_by_id(created.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.email, "alex@example.com")

    def test_repo_get_by_email(self):
        self.repo.create("John Doe", "john@example.com", 30)
        found = self.repo.get_by_email("john@example.com")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "John Doe")

    def test_repo_search_by_name_sql_ilike(self):
        self.repo.create("Suraj Sawant", "suraj@example.com", 21)
        self.repo.create("Suraj Kumar", "suraj.k@example.com", 24)
        self.repo.create("Jane Smith", "jane@example.com", 28)

        results = self.repo.search_by_name("suraj")
        self.assertEqual(len(results), 2)

    def test_repo_get_all_pagination(self):
        for i in range(1, 15):
            self.repo.create(f"User {i}", f"user{i}@example.com", 20 + i)

        page1 = self.repo.get_all(skip=0, limit=5)
        self.assertEqual(len(page1), 5)
        page2 = self.repo.get_all(skip=5, limit=5)
        self.assertEqual(len(page2), 5)

    def test_repo_update_user(self):
        user = self.repo.create("Suraj Old", "suraj.old@example.com", 20)
        updated = self.repo.update(user.id, "Suraj Updated", "suraj.new@example.com", 22)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.name, "Suraj Updated")

    def test_repo_delete_user(self):
        user = self.repo.create("ToDelete", "delete@example.com", 20)
        self.assertTrue(self.repo.delete(user.id))
        self.assertIsNone(self.repo.get_by_id(user.id))

    def test_repo_delete_nonexistent(self):
        self.assertFalse(self.repo.delete(999))

if __name__ == "__main__":
    unittest.main()

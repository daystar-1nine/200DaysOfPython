# ==============================================================================
# Test Suite : Integration Tests for Expense Repository Layer
# Objective  : Verify ExpenseRepository CRUD and search database operations.
# Concept    : Repository Pattern Integration Testing
# Why Used   : Asserts SQL query execution against SQLite database.
# ==============================================================================

import os
import sys
import unittest

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 30", "pyfinance", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database
from pyfinance.models.expense import Expense
from pyfinance.repositories.expense_repository import ExpenseRepository

def test_repository_crud_cycle(repository):
    # 1. Create
    exp = Expense(id=None, amount=350.0, category="Food", description="Dinner", date="2026-08-30")
    eid = repository.add(exp)
    assert eid > 0

    # 2. Read
    fetched = repository.get_by_id(eid)
    assert fetched is not None
    assert fetched.amount == 350.0

    # 3. Update
    updated = repository.update(eid, amount=400.0)
    assert updated is True
    assert repository.get_by_id(eid).amount == 400.0

    # 4. Delete
    deleted = repository.delete(eid)
    assert deleted is True
    assert repository.get_by_id(eid) is None

class TestRepositoryIntegrationRunner(unittest.TestCase):
    def test_repository_standalone(self):
        db_file = "temp_pyfinance_repo_int.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass
        db = Database(db_file)
        repo = ExpenseRepository(db)
        try:
            eid = repo.add(Expense(id=None, amount=150.0, category="Shopping", description="Socks", date="2026-08-30"))
            self.assertGreater(eid, 0)
            self.assertEqual(repo.get_by_id(eid).amount, 150.0)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()

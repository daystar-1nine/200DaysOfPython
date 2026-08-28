# ==============================================================================
# Pytest Config : Shared Test Fixtures (conftest.py)
# Objective  : Provide isolated temporary database and seeded services to all unit/integration tests.
# Concept    : Pytest Shared Fixture Scope & Dependency Injection
# Why Used   : Prevents code duplication across test files and protects production database.
# ==============================================================================

import os
import sys
import pytest

# Append Day 28 expense_tracker package path
base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pkg_root = os.path.join(base_repo, "Day 28", "expense_tracker", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.database import Database
from expense_tracker.services.expense_service import ExpenseService
from expense_tracker.services.report_service import ReportService

@pytest.fixture
def test_db_file(tmp_path):
    """Provides isolated temporary database path."""
    db_path = str(tmp_path / "test_pytest_expenses.db")
    yield db_path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

@pytest.fixture
def database(test_db_file):
    """Provides Database instance."""
    return Database(test_db_file)

@pytest.fixture
def expense_service(database):
    """Provides ExpenseService with injected test Database."""
    return ExpenseService(database)

@pytest.fixture
def report_service(database):
    """Provides ReportService with injected test Database."""
    return ReportService(database)

@pytest.fixture
def seeded_services(expense_service, report_service):
    """Provides services pre-seeded with User #1 and Category #1."""
    uid = expense_service.add_user("Suraj Sawant", "suraj_pytest@example.com")
    cid = expense_service.add_category("Food")
    return {
        "user_id": uid,
        "category_id": cid,
        "expense_service": expense_service,
        "report_service": report_service
    }

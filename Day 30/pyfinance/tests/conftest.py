# ==============================================================================
# Pytest Config : Shared PyFinance Test Fixtures
# Objective  : Provide isolated temporary databases, repositories, and services to unit & integration tests.
# Concept    : Pytest Fixtures & Shared Scope
# Why Used   : Ensures 100% test isolation without touching production databases.
# ==============================================================================

import os
import sys
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.database import Database
from pyfinance.repositories.expense_repository import ExpenseRepository
from pyfinance.services.expense_service import ExpenseService
from pyfinance.services.report_service import ReportService
from pyfinance.services.budget_service import BudgetService

@pytest.fixture
def temp_db_path(tmp_path):
    db_file = str(tmp_path / "test_pyfinance.db")
    yield db_file
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except OSError:
            pass

@pytest.fixture
def database(temp_db_path):
    return Database(temp_db_path)

@pytest.fixture
def repository(database):
    return ExpenseRepository(database)

@pytest.fixture
def expense_service(repository):
    return ExpenseService(repository)

@pytest.fixture
def report_service(repository):
    return ReportService(repository)

@pytest.fixture
def budget_service(repository):
    return BudgetService(repository)

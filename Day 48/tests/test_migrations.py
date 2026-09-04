# ==============================================================================
# Program    : Alembic Migrations & Schema Inspection Tests (test_migrations.py)
# Objective  : Verify Alembic revision execution, table existence, column schemas, and downgrade/upgrade rollbacks.
# Concept    : Database Schema Migration Testing
# Why Used   : Ensures database schema evolution (001-007) is reliable across environments.
# ==============================================================================

import os
import sys
from sqlalchemy import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.conftest import test_engine

def test_migration_tables_exist():
    """Verify Alembic migrations created all expected tables."""
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    assert "users" in tables
    assert "products" in tables
    assert "orders" in tables
    assert "order_items" in tables
    assert "alembic_version" in tables

def test_migration_users_columns_exist():
    """Verify users table contains columns added across revisions (phone, created_at, password_hash, role)."""
    inspector = inspect(test_engine)
    columns = {col["name"]: col for col in inspector.get_columns("users")}

    assert "id" in columns
    assert "name" in columns
    assert "email" in columns
    assert "age" in columns
    assert "phone" in columns
    assert "password_hash" in columns
    assert "role" in columns
    assert "created_at" in columns

def test_migration_indexes_exist():
    """Verify B-tree indexes added in revision 005 exist."""
    inspector = inspect(test_engine)
    user_indexes = [idx["name"] for idx in inspector.get_indexes("users")]
    product_indexes = [idx["name"] for idx in inspector.get_indexes("products")]

    assert "idx_users_email" in user_indexes
    assert "idx_products_name" in product_indexes

def test_migration_downgrade_and_upgrade():
    """Verify Alembic downgrade to 006 and upgrade back to head succeeds cleanly."""
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(test_engine.url))
    alembic_cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic"))

    # Downgrade revision 007 -> 006
    command.downgrade(alembic_cfg, "006")
    inspector_downgraded = inspect(test_engine)
    cols_downgraded = {col["name"] for col in inspector_downgraded.get_columns("users")}
    assert "password_hash" not in cols_downgraded
    assert "role" not in cols_downgraded

    # Upgrade revision 006 -> head (007)
    command.upgrade(alembic_cfg, "head")
    inspector_upgraded = inspect(test_engine)
    cols_upgraded = {col["name"] for col in inspector_upgraded.get_columns("users")}
    assert "password_hash" in cols_upgraded
    assert "role" in cols_upgraded

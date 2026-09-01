# ==============================================================================
# Test Suite : Alembic Migration Tests (test_migrations.py)
# Objective  : Test Alembic migration upgrade, downgrade, column additions, and index presence.
# Concept    : Migration Integration Testing (Day 46 requirement)
# Why Used   : Asserts schema evolution integrity and rollback capabilities.
# ==============================================================================

import os
import sys
import pytest
from sqlalchemy import inspect
from alembic.config import Config
from alembic import command

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tests.conftest import engine, test_db_url

def test_database_tables_exist():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "products" in tables
    assert "orders" in tables
    assert "order_items" in tables

def test_users_table_columns_exist():
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("users")]
    assert "id" in columns
    assert "name" in columns
    assert "email" in columns
    assert "phone" in columns        # Added in Migration 002
    assert "created_at" in columns   # Added in Migration 006

def test_products_table_columns_exist():
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("products")]
    assert "id" in columns
    assert "name" in columns
    assert "price" in columns
    assert "stock" in columns
    assert "description" in columns
    assert "category" in columns     # Added in Migration 003
    assert "created_at" in columns   # Added in Migration 004

def test_indexes_exist():
    inspector = inspect(engine)
    user_indexes = [idx["name"] for idx in inspector.get_indexes("users")]
    product_indexes = [idx["name"] for idx in inspector.get_indexes("products")]
    assert "idx_users_email" in user_indexes
    assert "idx_products_name" in product_indexes

def test_migration_downgrade_and_upgrade():
    alembic_ini_path = os.path.join(src_dir, "alembic.ini")
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", os.path.join(src_dir, "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)

    # Downgrade -1 (from 006 to 005) -> created_at should be dropped from users
    command.downgrade(alembic_cfg, "-1")
    inspector = inspect(engine)
    cols_after_downgrade = [col["name"] for col in inspector.get_columns("users")]
    assert "created_at" not in cols_after_downgrade

    # Re-upgrade to head (to 006)
    command.upgrade(alembic_cfg, "head")
    inspector_re = inspect(engine)
    cols_after_upgrade = [col["name"] for col in inspector_re.get_columns("users")]
    assert "created_at" in cols_after_upgrade

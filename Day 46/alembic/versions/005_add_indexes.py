# ==============================================================================
# Migration  : 005_add_indexes.py
# Objective  : Add B-tree indexes on users.email and products.name.
# Concept    : Database Index Creation & Query Optimization
# Why Used   : Accelerates lookups for user email authentication and product search.
# ==============================================================================

"""add indexes on users email and products name

Revision ID: 005
Revises: 004
Create Date: 2026-09-01 10:40:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_products_name', 'products', ['name'], unique=False)

def downgrade() -> None:
    op.drop_index('idx_products_name', table_name='products')
    op.drop_index('idx_users_email', table_name='users')

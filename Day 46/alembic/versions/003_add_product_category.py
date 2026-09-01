# ==============================================================================
# Migration  : 003_add_product_category.py
# Objective  : Add optional category column to products table.
# Concept    : Product Schema Evolution
# Why Used   : Extends products table schema for product categorization.
# ==============================================================================

"""add product category column

Revision ID: 003
Revises: 002
Create Date: 2026-09-01 10:20:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.add_column(sa.Column('category', sa.String(length=50), nullable=True))

def downgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.drop_column('category')

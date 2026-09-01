# ==============================================================================
# Migration  : 004_add_product_created_at.py
# Objective  : Add created_at timestamp column to products table.
# Concept    : Timestamp Column Addition
# Why Used   : Tracks product creation timestamp for catalog auditing.
# ==============================================================================

"""add product created_at column

Revision ID: 004
Revises: 003
Create Date: 2026-09-01 10:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))

def downgrade() -> None:
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.drop_column('created_at')

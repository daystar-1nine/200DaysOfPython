"""add category column to products table

Revision ID: 003
Revises: 002
Create Date: 2026-09-01 00:10:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('products', sa.Column('category', sa.String(length=100), nullable=True))

def downgrade() -> None:
    op.drop_column('products', 'category')

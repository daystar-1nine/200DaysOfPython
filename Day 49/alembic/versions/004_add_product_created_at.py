"""add created_at column to products table

Revision ID: 004
Revises: 003
Create Date: 2026-09-01 00:15:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('products', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False))

def downgrade() -> None:
    op.drop_column('products', 'created_at')

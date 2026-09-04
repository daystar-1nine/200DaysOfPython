"""add indexes to users email and products name

Revision ID: 005
Revises: 004
Create Date: 2026-09-01 00:20:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_products_name', 'products', ['name'], unique=False)

def downgrade() -> None:
    op.drop_index('idx_products_name', table_name='products')
    op.drop_index('idx_users_email', table_name='users')

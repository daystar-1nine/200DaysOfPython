"""add phone column to users table

Revision ID: 002
Revises: 001
Create Date: 2026-09-01 00:05:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('users', sa.Column('phone', sa.String(length=20), nullable=True))

def downgrade() -> None:
    op.drop_column('users', 'phone')

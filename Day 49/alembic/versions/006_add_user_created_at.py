"""add created_at column to users table

Revision ID: 006
Revises: 005
Create Date: 2026-09-01 00:25:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('users', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False))

def downgrade() -> None:
    op.drop_column('users', 'created_at')

"""add password_hash and role columns to users table

Revision ID: 007
Revises: 006
Create Date: 2026-09-01 00:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('users', sa.Column('password_hash', sa.String(length=255), nullable=False, server_default='pbkdf2:sha256:default'))
    op.add_column('users', sa.Column('role', sa.String(length=50), nullable=False, server_default='user'))

def downgrade() -> None:
    op.drop_column('users', 'role')
    op.drop_column('users', 'password_hash')

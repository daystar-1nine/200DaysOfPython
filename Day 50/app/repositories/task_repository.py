"""
===============================================================================
DAY 50 — TASK REPOSITORY (DATA ACCESS LAYER)
===============================================================================
This module provides database abstraction for Task entity persistence, including
status/priority filtering, full-text search, pagination, and ownership verification.
===============================================================================
"""

from typing import Sequence
from sqlalchemy import select, or_
from sqlalchemy.orm import Session
from app.models.task import Task


class TaskRepository:
    """Repository handling SQL operations on the 'tasks' table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, task: Task) -> Task:
        """Persist a new Task entity."""
        # What is used: Session add, commit, and refresh operations.
        # Why it is used: Inserts task row into table.
        # How it works: Saves Task model to DB and refreshes auto-generated fields.
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get_by_id(self, task_id: int) -> Task | None:
        """Fetch a single task by ID."""
        # What is used: Select query filtering by task_id.
        # Why it is used: Retrieves task record for view/update/delete operations.
        # How it works: Compiles select(Task) where Task.id == task_id.
        stmt = select(Task).where(Task.id == task_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_user_tasks(
        self,
        user_id: int,
        status: str | None = None,
        priority: str | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Task]:
        """Fetch tasks belonging to a specific user with filtering and pagination."""
        # What is used: Dynamic select query construction with ILIKE search and filters.
        # Why it is used: Filters user tasks by status, priority, search text, and applies limit/offset.
        # How it works: Appends .where() clauses conditionally to select(Task) statement.
        stmt = select(Task).where(Task.user_id == user_id)

        if status:
            stmt = stmt.where(Task.status == status)
        if priority:
            stmt = stmt.where(Task.priority == priority)
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(or_(Task.title.ilike(pattern), Task.description.ilike(pattern)))

        stmt = stmt.order_by(Task.created_at.desc()).offset(offset).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def list_all_tasks(self, offset: int = 0, limit: int = 100) -> Sequence[Task]:
        """Fetch all tasks across all users (admin privilege)."""
        # What is used: Select query returning all tasks for admin reporting.
        # Why it is used: Provides dataset for admin tasks overview endpoint.
        # How it works: Executes select(Task) with offset and limit pagination.
        stmt = select(Task).order_by(Task.id.desc()).offset(offset).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def update(self, task: Task) -> Task:
        """Commit updates to an existing Task entity."""
        # What is used: Session commit and refresh.
        # Why it is used: Saves updated Task entity attributes to database.
        # How it works: Commits active transaction and reloads entity state.
        self.db.commit()
        self.db.refresh(task)
        return task

    def delete(self, task: Task) -> None:
        """Delete a Task entity."""
        # What is used: Session delete and commit.
        # Why it is used: Removes task row from database table.
        # How it works: Issues DELETE statement for task entity.
        self.db.delete(task)
        self.db.commit()

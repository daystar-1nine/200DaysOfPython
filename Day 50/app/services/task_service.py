"""
===============================================================================
DAY 50 — TASK SERVICE (BUSINESS LOGIC LAYER)
===============================================================================
This module manages Task business operations including task creation, user task
isolation checks, status/priority filtering, partial updates, and deletion.
===============================================================================
"""

from typing import Sequence
from app.models.task import Task
from app.schemas.task import TaskCreate, TaskUpdate, TaskPatch
from app.repositories.task_repository import TaskRepository
from app.exceptions import NotFoundError, ForbiddenError
from app.logging_config import logger


class TaskService:
    """Service executing operations on Task items."""

    def __init__(self, task_repo: TaskRepository) -> None:
        self.task_repo = task_repo

    def create_task(self, user_id: int, payload: TaskCreate) -> Task:
        """Create a new task for the authenticated user."""
        # What is used: Task ORM entity instantiation and repository persistence.
        # Why it is used: Links created task item to the authenticated user ID.
        # How it works: Instantiates Task model and saves via task_repo.create.
        task = Task(
            title=payload.title,
            description=payload.description,
            status=payload.status.value,
            priority=payload.priority.value,
            due_date=payload.due_date,
            user_id=user_id,
        )
        created_task = self.task_repo.create(task)
        logger.info({"event": "task_created", "task_id": created_task.id, "user_id": user_id})
        return created_task

    def get_task(self, task_id: int, current_user_id: int, current_user_role: str = "user") -> Task:
        """Retrieve a task by ID enforcing user ownership or admin permission."""
        # What is used: Task existence and ownership authorization validation.
        # Why it is used: Prevents User A from viewing tasks owned by User B (isolation).
        # How it works: Raises NotFoundError (404) if missing or ForbiddenError (403) if unauthorized.
        task = self.task_repo.get_by_id(task_id)
        if not task:
            raise NotFoundError(f"Task with ID {task_id} not found.")
        if current_user_role != "admin" and task.user_id != current_user_id:
            raise ForbiddenError("Access denied. You do not own this task.")
        return task

    def list_tasks(
        self,
        user_id: int,
        status: str | None = None,
        priority: str | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Task]:
        """List tasks belonging to current user with optional filtering."""
        # What is used: TaskRepository list_user_tasks invocation.
        # Why it is used: Returns filtered, paginated list of tasks owned by user.
        # How it works: Passes user_id and filter options to task_repo.
        return self.task_repo.list_user_tasks(
            user_id=user_id,
            status=status,
            priority=priority,
            search=search,
            offset=offset,
            limit=limit,
        )

    def update_task(self, task_id: int, current_user_id: int, payload: TaskUpdate, current_user_role: str = "user") -> Task:
        """Replace task details (PUT)."""
        # What is used: Task ownership verification and attribute assignment.
        # Why it is used: Implements full resource replacement.
        # How it works: Updates all task attributes and commits changes.
        task = self.get_task(task_id, current_user_id, current_user_role)
        task.title = payload.title
        task.description = payload.description
        task.status = payload.status.value
        task.priority = payload.priority.value
        task.due_date = payload.due_date
        return self.task_repo.update(task)

    def patch_task(self, task_id: int, current_user_id: int, payload: TaskPatch, current_user_role: str = "user") -> Task:
        """Partially update task details (PATCH)."""
        # What is used: Selective attribute assignment using dict exclude_unset.
        # Why it is used: Applies updates only to fields explicitly provided in request body.
        # How it works: Iterates model_dump(exclude_unset=True) and sets attributes on Task model.
        task = self.get_task(task_id, current_user_id, current_user_role)
        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                if key == "status":
                    setattr(task, key, payload.status.value if payload.status else task.status)
                elif key == "priority":
                    setattr(task, key, payload.priority.value if payload.priority else task.priority)
                else:
                    setattr(task, key, value)
        return self.task_repo.update(task)

    def delete_task(self, task_id: int, current_user_id: int, current_user_role: str = "user") -> None:
        """Delete task entity."""
        # What is used: Task ownership check and repository deletion.
        # Why it is used: Deletes task entity cleanly.
        # How it works: Enforces ownership check then calls task_repo.delete.
        task = self.get_task(task_id, current_user_id, current_user_role)
        self.task_repo.delete(task)
        logger.info({"event": "task_deleted", "task_id": task_id, "user_id": current_user_id})

    def list_all_admin_tasks(self, offset: int = 0, limit: int = 100) -> Sequence[Task]:
        """List all tasks across all users (Admin privilege)."""
        # What is used: Repository list_all_tasks.
        # Why it is used: Allows admin users to audit all system tasks.
        # How it works: Calls task_repo.list_all_tasks.
        return self.task_repo.list_all_tasks(offset=offset, limit=limit)

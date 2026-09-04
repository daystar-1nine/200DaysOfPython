"""
===============================================================================
DAY 50 — TASKS ROUTER MODULE
===============================================================================
This module defines endpoints for managing user tasks (CRUD, filtering, search,
pagination, and authorization isolation).
===============================================================================
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.task import (
    TaskCreate,
    TaskUpdate,
    TaskPatch,
    TaskResponse,
    TaskStatus,
    TaskPriority,
)
from app.repositories.task_repository import TaskRepository
from app.services.task_service import TaskService
from app.dependencies.auth import get_current_user

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def get_task_service(db: Session = Depends(get_db)) -> TaskService:
    """Dependency instantiating TaskService with task repository."""
    return TaskService(TaskRepository(db))


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(
    payload: TaskCreate,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Create a new task item for the current authenticated user."""
    # What is used: TaskService create_task execution.
    # Why it is used: Links newly created task entity to current_user.id.
    # How it works: Saves task and returns TaskResponse schema.
    task = service.create_task(user_id=current_user.id, payload=payload)
    return TaskResponse.model_validate(task)


@router.get("", response_model=List[TaskResponse], status_code=status.HTTP_200_OK)
def list_user_tasks(
    status_filter: Optional[TaskStatus] = Query(None, alias="status", description="Filter by task status"),
    priority_filter: Optional[TaskPriority] = Query(None, alias="priority", description="Filter by task priority"),
    search: Optional[str] = Query(None, min_length=1, description="Search title and description"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=100, description="Pagination limit"),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> List[TaskResponse]:
    """List tasks belonging to current user with filtering, search, and pagination."""
    # What is used: TaskService list_tasks with query parameters.
    # Why it is used: Retrieves user tasks matching status, priority, search text, and pagination limit/offset.
    # How it works: Converts enum values to strings and executes filtered repo query.
    tasks = service.list_tasks(
        user_id=current_user.id,
        status=status_filter.value if status_filter else None,
        priority=priority_filter.value if priority_filter else None,
        search=search,
        offset=offset,
        limit=limit,
    )
    return [TaskResponse.model_validate(t) for t in tasks]


@router.get("/{task_id}", response_model=TaskResponse, status_code=status.HTTP_200_OK)
def get_task_by_id(
    task_id: int,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Get task details by ID (User task ownership enforced)."""
    # What is used: TaskService get_task execution.
    # Why it is used: Enforces task ownership isolation check.
    # How it works: Fetches task or raises 404/403.
    task = service.get_task(task_id=task_id, current_user_id=current_user.id, current_user_role=current_user.role)
    return TaskResponse.model_validate(task)


@router.put("/{task_id}", response_model=TaskResponse, status_code=status.HTTP_200_OK)
def update_task(
    task_id: int,
    payload: TaskUpdate,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Replace task details (PUT)."""
    # What is used: TaskService update_task execution.
    # Why it is used: Full resource update.
    # How it works: Replaces task attributes and returns updated TaskResponse.
    task = service.update_task(task_id=task_id, current_user_id=current_user.id, payload=payload, current_user_role=current_user.role)
    return TaskResponse.model_validate(task)


@router.patch("/{task_id}", response_model=TaskResponse, status_code=status.HTTP_200_OK)
def patch_task(
    task_id: int,
    payload: TaskPatch,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Partially update task details (PATCH)."""
    # What is used: TaskService patch_task execution.
    # Why it is used: Updates only provided attributes.
    # How it works: Updates non-null fields in Task model.
    task = service.patch_task(task_id=task_id, current_user_id=current_user.id, payload=payload, current_user_role=current_user.role)
    return TaskResponse.model_validate(task)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> None:
    """Delete a task item by ID."""
    # What is used: TaskService delete_task execution.
    # Why it is used: Removes task entity from database.
    # How it works: Issues task deletion if ownership verified.
    service.delete_task(task_id=task_id, current_user_id=current_user.id, current_user_role=current_user.role)

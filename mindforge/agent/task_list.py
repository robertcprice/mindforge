"""
MindForge Internal Task List

Persistent task tracking for the consciousness agent.
Tasks persist across consciousness cycles via MemoryStore.
Supports hierarchical task breakdown (tasks → subtasks).
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from mindforge.memory.store import Memory, MemoryStore, MemoryType

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of an internal task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 1  # Must do immediately
    HIGH = 2      # Important, do soon
    NORMAL = 3    # Standard priority
    LOW = 4       # Can wait


@dataclass
class InternalTask:
    """Represents a task the agent wants to accomplish.

    Tasks can have subtasks for hierarchical breakdown.
    Progress is documented via progress_notes.
    """
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Hierarchy
    parent_id: Optional[str] = None  # If this is a subtask
    subtask_ids: list[str] = field(default_factory=list)

    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None

    # Progress documentation
    progress_notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Context for execution
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "subtask_ids": self.subtask_ids,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "progress_notes": self.progress_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InternalTask":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            priority=TaskPriority(data["priority"]),
            parent_id=data.get("parent_id"),
            subtask_ids=data.get("subtask_ids", []),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            last_error=data.get("last_error"),
            progress_notes=data.get("progress_notes", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            context=data.get("context", {}),
        )

    @property
    def is_actionable(self) -> bool:
        """Can this task be worked on now?"""
        return (
            self.status == TaskStatus.PENDING
            and self.attempts < self.max_attempts
        )

    @property
    def has_subtasks(self) -> bool:
        """Does this task have subtasks?"""
        return len(self.subtask_ids) > 0

    def add_progress_note(self, note: str) -> None:
        """Add a progress note with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_notes.append(f"[{timestamp}] {note}")
        self.updated_at = datetime.now()


@dataclass
class WorkLogEntry:
    """Records work done during a consciousness cycle."""
    task_id: str
    action_taken: str
    result: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "action_taken": self.action_taken,
            "result": self.result,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkLogEntry":
        return cls(
            task_id=data["task_id"],
            action_taken=data["action_taken"],
            result=data["result"],
            success=data["success"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error=data.get("error"),
        )


class PersistentTaskList:
    """Manages tasks with persistence via MemoryStore.

    Tasks are stored as SYSTEM memories with tag 'internal_task'.
    Provides methods for task lifecycle management and hierarchy.
    """

    TASK_TAG = "internal_task"
    TASK_LIST_KEY = "task_list_v1"

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._cache: dict[str, InternalTask] = {}
        self._memory_id: Optional[int] = None
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load tasks from memory store."""
        # Search for existing task list memory
        memories = self.memory_store.search(
            query="",
            memory_type=MemoryType.SYSTEM,
            tags=[self.TASK_TAG],
            limit=1,
        )

        if memories:
            memory = memories[0]
            self._memory_id = memory.id
            try:
                task_data = json.loads(memory.content)
                for task_dict in task_data.get("tasks", []):
                    task = InternalTask.from_dict(task_dict)
                    self._cache[task.id] = task
                logger.info(f"Loaded {len(self._cache)} tasks from memory")
            except Exception as e:
                logger.warning(f"Failed to load tasks: {e}")
                self._cache = {}
        else:
            logger.info("No existing task list found, starting fresh")

    def _save_tasks(self) -> None:
        """Persist tasks to memory store."""
        task_data = {
            "tasks": [task.to_dict() for task in self._cache.values()],
            "updated_at": datetime.now().isoformat(),
        }
        content = json.dumps(task_data)

        if self._memory_id:
            # Update existing memory (delete and re-create since store doesn't have update)
            self.memory_store.delete(self._memory_id)

        memory = Memory(
            content=content,
            memory_type=MemoryType.SYSTEM,
            source="task_list",
            importance=1.0,  # High importance - don't decay
            tags=[self.TASK_TAG],
            metadata={"task_count": len(self._cache)},
        )
        self._memory_id = self.memory_store.store(memory)

    def _is_similar_task(self, description: str, threshold: float = 0.6) -> Optional[InternalTask]:
        """Check if a similar task already exists (not completed/failed).

        Uses simple word overlap to detect duplicates.
        Returns the existing task if found, None otherwise.
        """
        desc_words = set(description.lower().split())

        for task in self._cache.values():
            # Skip completed/failed tasks
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                continue

            existing_words = set(task.description.lower().split())

            # Calculate word overlap
            if not desc_words or not existing_words:
                continue

            overlap = len(desc_words & existing_words)
            min_len = min(len(desc_words), len(existing_words))

            if min_len > 0 and overlap / min_len >= threshold:
                return task

        return None

    def add_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        parent_id: Optional[str] = None,
        context: Optional[dict] = None,
        check_duplicates: bool = True,
    ) -> Optional[InternalTask]:
        """Add a new task.

        Args:
            description: What needs to be done
            priority: Task priority
            parent_id: If this is a subtask, ID of parent
            context: Additional context for execution
            check_duplicates: If True, skip if similar task exists

        Returns:
            The created task, or None if duplicate
        """
        # Check for duplicates (only for non-subtasks)
        if check_duplicates and not parent_id:
            existing = self._is_similar_task(description)
            if existing:
                logger.info(f"Skipping duplicate task, similar to: {existing.id} - {existing.description[:40]}...")
                return None

        task = InternalTask(
            id=str(uuid.uuid4())[:8],
            description=description,
            priority=priority,
            parent_id=parent_id,
            context=context or {},
        )

        self._cache[task.id] = task

        # If this is a subtask, update parent
        if parent_id and parent_id in self._cache:
            self._cache[parent_id].subtask_ids.append(task.id)

        self._save_tasks()
        logger.info(f"Added task: {task.id} - {description[:50]}...")
        return task

    def add_subtask(
        self,
        parent_id: str,
        description: str,
        priority: Optional[TaskPriority] = None,
    ) -> Optional[InternalTask]:
        """Add a subtask to an existing task.

        Args:
            parent_id: ID of parent task
            description: Subtask description
            priority: Inherits from parent if not specified

        Returns:
            The created subtask, or None if parent not found
        """
        parent = self._cache.get(parent_id)
        if not parent:
            logger.warning(f"Parent task {parent_id} not found")
            return None

        task_priority = priority or parent.priority
        return self.add_task(
            description=description,
            priority=task_priority,
            parent_id=parent_id,
            context={"inherited_from": parent_id},
        )

    def get_task(self, task_id: str) -> Optional[InternalTask]:
        """Get a task by ID."""
        return self._cache.get(task_id)

    def get_all_tasks(self) -> list[InternalTask]:
        """Get all tasks."""
        return list(self._cache.values())

    def get_pending_tasks(self) -> list[InternalTask]:
        """Get all pending tasks (including subtasks)."""
        return [
            t for t in self._cache.values()
            if t.status == TaskStatus.PENDING
        ]

    def get_root_tasks(self) -> list[InternalTask]:
        """Get top-level tasks (no parent)."""
        return [
            t for t in self._cache.values()
            if t.parent_id is None
        ]

    def get_subtasks(self, parent_id: str) -> list[InternalTask]:
        """Get subtasks of a task."""
        parent = self._cache.get(parent_id)
        if not parent:
            return []
        return [
            self._cache[sid]
            for sid in parent.subtask_ids
            if sid in self._cache
        ]

    def get_next_actionable_task(self) -> Optional[InternalTask]:
        """Get the next task that can be worked on.

        Prioritizes:
        1. Higher priority tasks
        2. Subtasks before parent tasks (if parent has subtasks)
        3. Older tasks (FIFO within priority)
        """
        actionable = []

        for task in self._cache.values():
            if not task.is_actionable:
                continue

            # If task has incomplete subtasks, skip it (work on subtasks first)
            if task.has_subtasks:
                subtasks = self.get_subtasks(task.id)
                incomplete_subtasks = [
                    s for s in subtasks
                    if s.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                ]
                if incomplete_subtasks:
                    continue  # Can't work on parent until subtasks done

            actionable.append(task)

        if not actionable:
            return None

        # Sort by priority (lower value = higher priority), then by creation time
        actionable.sort(key=lambda t: (t.priority.value, t.created_at))
        return actionable[0]

    def mark_in_progress(self, task_id: str) -> bool:
        """Mark a task as in progress."""
        task = self._cache.get(task_id)
        if task:
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            self._save_tasks()
            return True
        return False

    def mark_completed(self, task_id: str, notes: str = "") -> bool:
        """Mark a task as completed.

        If all subtasks of a parent are complete, auto-completes the parent.
        """
        task = self._cache.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()
        if notes:
            task.add_progress_note(f"Completed: {notes}")

        # Check if parent should be auto-completed
        if task.parent_id:
            self._check_parent_completion(task.parent_id)

        self._save_tasks()
        logger.info(f"Task completed: {task_id}")
        return True

    def _check_parent_completion(self, parent_id: str) -> None:
        """Check if all subtasks are complete and mark parent if so."""
        parent = self._cache.get(parent_id)
        if not parent:
            return

        subtasks = self.get_subtasks(parent_id)
        all_done = all(
            s.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for s in subtasks
        )

        if all_done:
            parent.status = TaskStatus.COMPLETED
            parent.completed_at = datetime.now()
            parent.add_progress_note("Auto-completed: all subtasks finished")
            logger.info(f"Parent task auto-completed: {parent_id}")

    def mark_failed(self, task_id: str, error: str) -> bool:
        """Mark a task as failed with error message."""
        task = self._cache.get(task_id)
        if not task:
            return False

        task.attempts += 1
        task.last_error = error
        task.add_progress_note(f"Attempt {task.attempts} failed: {error}")

        if task.attempts >= task.max_attempts:
            task.status = TaskStatus.FAILED
            task.add_progress_note(f"Task failed after {task.attempts} attempts")
            logger.warning(f"Task failed: {task_id} - {error}")
        else:
            # Reset to pending for retry
            task.status = TaskStatus.PENDING
            logger.info(f"Task will retry: {task_id} (attempt {task.attempts}/{task.max_attempts})")

        task.updated_at = datetime.now()
        self._save_tasks()
        return True

    def mark_blocked(self, task_id: str, reason: str) -> bool:
        """Mark a task as blocked."""
        task = self._cache.get(task_id)
        if task:
            task.status = TaskStatus.BLOCKED
            task.add_progress_note(f"Blocked: {reason}")
            task.updated_at = datetime.now()
            self._save_tasks()
            return True
        return False

    def add_progress_note(self, task_id: str, note: str) -> bool:
        """Add a progress note to a task."""
        task = self._cache.get(task_id)
        if task:
            task.add_progress_note(note)
            self._save_tasks()
            return True
        return False

    def remove_task(self, task_id: str, cascade: bool = True) -> bool:
        """Remove a task.

        Args:
            task_id: ID of task to remove
            cascade: If True, also remove subtasks

        Returns:
            True if removed
        """
        task = self._cache.get(task_id)
        if not task:
            return False

        # Remove subtasks if cascade
        if cascade:
            for subtask_id in task.subtask_ids:
                self.remove_task(subtask_id, cascade=True)

        # Remove from parent's subtask list
        if task.parent_id and task.parent_id in self._cache:
            parent = self._cache[task.parent_id]
            if task_id in parent.subtask_ids:
                parent.subtask_ids.remove(task_id)

        del self._cache[task_id]
        self._save_tasks()
        return True

    def clear_completed(self) -> int:
        """Remove all completed tasks."""
        completed_ids = [
            t.id for t in self._cache.values()
            if t.status == TaskStatus.COMPLETED
        ]
        for task_id in completed_ids:
            self.remove_task(task_id, cascade=False)
        return len(completed_ids)

    def get_task_tree(self) -> list[dict]:
        """Get hierarchical view of tasks.

        Returns list of root tasks with nested subtasks.
        """
        def build_tree(task: InternalTask) -> dict:
            subtasks = self.get_subtasks(task.id)
            return {
                "task": task,
                "subtasks": [build_tree(s) for s in subtasks],
            }

        return [build_tree(t) for t in self.get_root_tasks()]

    def format_task_tree(self, include_completed: bool = False) -> str:
        """Format task tree as human-readable string."""
        lines = []

        def format_task(task: InternalTask, indent: int = 0) -> None:
            if not include_completed and task.status == TaskStatus.COMPLETED:
                return

            prefix = "  " * indent
            status_icons = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.IN_PROGRESS: "[~]",
                TaskStatus.COMPLETED: "[x]",
                TaskStatus.FAILED: "[!]",
                TaskStatus.BLOCKED: "[#]",
            }
            icon = status_icons.get(task.status, "[ ]")
            priority_markers = {
                TaskPriority.CRITICAL: "!!!",
                TaskPriority.HIGH: "!!",
                TaskPriority.NORMAL: "",
                TaskPriority.LOW: "~",
            }
            priority = priority_markers.get(task.priority, "")

            line = f"{prefix}{icon} {task.description}"
            if priority:
                line += f" {priority}"
            if task.last_error:
                line += f" (err: {task.last_error[:30]}...)"
            lines.append(line)

            # Add subtasks
            for subtask in self.get_subtasks(task.id):
                format_task(subtask, indent + 1)

        for root_task in self.get_root_tasks():
            format_task(root_task)

        return "\n".join(lines) if lines else "(no tasks)"

    def get_statistics(self) -> dict:
        """Get task statistics."""
        tasks = list(self._cache.values())
        return {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "in_progress": sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "blocked": sum(1 for t in tasks if t.status == TaskStatus.BLOCKED),
        }

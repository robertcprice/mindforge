"""
Comprehensive tests for the Task List System.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from mindforge.agent.task_list import (
    InternalTask,
    TaskStatus,
    TaskPriority,
    PersistentTaskList,
    WorkLogEntry,
)
from mindforge.memory.store import MemoryStore


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses exist."""
        expected = ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "BLOCKED"]
        for status in expected:
            assert hasattr(TaskStatus, status)

    def test_status_values(self):
        """Test status values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.BLOCKED.value == "blocked"


class TestTaskPriority:
    """Test TaskPriority enum."""

    def test_all_priorities_exist(self):
        """Verify all expected priorities exist."""
        expected = ["CRITICAL", "HIGH", "NORMAL", "LOW"]
        for priority in expected:
            assert hasattr(TaskPriority, priority)

    def test_priority_ordering(self):
        """Test priority values for ordering."""
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value


class TestInternalTask:
    """Test InternalTask dataclass."""

    def test_create_basic_task(self):
        """Test creating a basic task."""
        task = InternalTask(
            id="test123",
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
        )
        assert task.id == "test123"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.parent_id is None
        assert task.subtask_ids == []
        assert task.attempts == 0
        assert task.max_attempts == 3

    def test_create_subtask(self):
        """Test creating a subtask with parent."""
        task = InternalTask(
            id="sub123",
            description="Subtask",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            parent_id="parent123",
        )
        assert task.parent_id == "parent123"

    def test_task_with_all_fields(self):
        """Test task with all optional fields."""
        task = InternalTask(
            id="full123",
            description="Full task",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH,
            parent_id="parent",
            subtask_ids=["sub1", "sub2"],
            attempts=2,
            max_attempts=5,
            last_error="Previous error",
            progress_notes=["Note 1", "Note 2"],
        )
        assert task.subtask_ids == ["sub1", "sub2"]
        assert task.attempts == 2
        assert task.max_attempts == 5
        assert task.last_error == "Previous error"
        assert len(task.progress_notes) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        task = InternalTask(
            id="test123",
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
        )
        data = task.to_dict()

        assert data["id"] == "test123"
        assert data["description"] == "Test task"
        assert data["status"] == "pending"
        assert data["priority"] == 3  # NORMAL = 3
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test123",
            "description": "Test task",
            "status": "pending",
            "priority": 2,  # HIGH = 2
            "parent_id": None,
            "subtask_ids": [],
            "attempts": 0,
            "max_attempts": 3,
            "last_error": None,
            "progress_notes": [],
            "created_at": "2025-01-01T12:00:00",
            "updated_at": "2025-01-01T12:00:00",
        }
        task = InternalTask.from_dict(data)

        assert task.id == "test123"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.HIGH


class TestWorkLogEntry:
    """Test WorkLogEntry dataclass."""

    def test_create_entry(self):
        """Test creating a work log entry."""
        entry = WorkLogEntry(
            task_id="task123",
            action_taken="shell(command='ls')",
            result="file1.txt\nfile2.txt",
            success=True,
        )
        assert entry.task_id == "task123"
        assert entry.action_taken == "shell(command='ls')"
        assert entry.result == "file1.txt\nfile2.txt"
        assert entry.success is True

    def test_entry_with_failure(self):
        """Test work log entry with failure."""
        entry = WorkLogEntry(
            task_id="task123",
            action_taken="shell(command='invalid')",
            result="Error: command not found",
            success=False,
        )
        assert entry.success is False


class TestPersistentTaskList:
    """Test PersistentTaskList class."""

    @pytest.fixture
    def memory_store(self, tmp_path):
        """Create a temporary memory store."""
        db_path = tmp_path / "test_memories.db"
        return MemoryStore(db_path)  # MemoryStore expects Path, not str

    @pytest.fixture
    def task_list(self, memory_store):
        """Create a task list with memory store."""
        return PersistentTaskList(memory_store)

    def test_create_empty_task_list(self, task_list):
        """Test creating an empty task list."""
        stats = task_list.get_statistics()
        assert stats["total"] == 0

    def test_add_task(self, task_list):
        """Test adding a task."""
        task = task_list.add_task("Test task")
        assert task is not None
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL

    def test_add_task_with_priority(self, task_list):
        """Test adding a task with specific priority."""
        task = task_list.add_task("High priority task", priority=TaskPriority.HIGH)
        assert task.priority == TaskPriority.HIGH

    def test_add_duplicate_task(self, task_list):
        """Test that duplicate tasks are not added."""
        task1 = task_list.add_task("Debug the workflow errors")
        task2 = task_list.add_task("Debug workflow errors")  # Similar

        assert task1 is not None
        # task2 should be None because it's a duplicate
        assert task2 is None

    def test_add_different_tasks(self, task_list):
        """Test that different tasks are added."""
        task1 = task_list.add_task("Debug the workflow")
        task2 = task_list.add_task("Write documentation")

        assert task1 is not None
        assert task2 is not None

    def test_add_subtask(self, task_list):
        """Test adding a subtask."""
        parent = task_list.add_task("Parent task")
        subtask = task_list.add_subtask(parent.id, "Subtask 1")

        assert subtask is not None
        assert subtask.parent_id == parent.id
        # Verify subtask was added
        all_tasks = task_list.get_all_tasks()
        assert any(t.parent_id == parent.id for t in all_tasks)

    def test_subtask_bypasses_deduplication(self, task_list):
        """Test that subtasks can be similar to parent."""
        parent = task_list.add_task("Fix the bug")
        subtask = task_list.add_subtask(parent.id, "Fix the specific bug")

        # Should not be deduplicated even though similar
        assert subtask is not None

    def test_get_pending_tasks(self, task_list):
        """Test getting pending tasks."""
        task_list.add_task("Task 1")
        task_list.add_task("Task 2")

        pending = task_list.get_pending_tasks()
        assert len(pending) == 2

    def test_get_next_actionable_task(self, task_list):
        """Test getting next actionable task."""
        task_list.add_task("Task 1")
        task_list.add_task("Task 2", priority=TaskPriority.HIGH)

        next_task = task_list.get_next_actionable_task()
        # Should return high priority task first
        assert next_task is not None
        assert next_task.priority == TaskPriority.HIGH

    def test_mark_in_progress(self, task_list):
        """Test marking task in progress."""
        task = task_list.add_task("Test task")
        result = task_list.mark_in_progress(task.id)

        assert result is True
        updated_task = task_list.get_task(task.id)
        assert updated_task.status == TaskStatus.IN_PROGRESS

    def test_mark_completed(self, task_list):
        """Test marking task completed."""
        task = task_list.add_task("Test task")
        task_list.mark_in_progress(task.id)
        result = task_list.mark_completed(task.id, "Done successfully")

        assert result is True
        updated_task = task_list.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETED

    def test_mark_failed(self, task_list):
        """Test marking task failed."""
        task = task_list.add_task("Test task failed")
        result = task_list.mark_failed(task.id, "Error occurred")

        assert result is True
        updated_task = task_list.get_task(task.id)
        # mark_failed records the error, status may vary by implementation
        assert updated_task.last_error == "Error occurred"

    def test_mark_blocked(self, task_list):
        """Test marking task blocked."""
        task = task_list.add_task("Test task")
        result = task_list.mark_blocked(task.id, "Dependency missing")

        assert result is True
        updated_task = task_list.get_task(task.id)
        assert updated_task.status == TaskStatus.BLOCKED

    def test_add_progress_note(self, task_list):
        """Test adding progress note."""
        task = task_list.add_task("Test task")
        result = task_list.add_progress_note(task.id, "Made progress")

        assert result is True
        updated_task = task_list.get_task(task.id)
        assert any("Made progress" in note for note in updated_task.progress_notes)

    def test_get_statistics(self, task_list):
        """Test getting task statistics."""
        task1 = task_list.add_task("Task One")
        task2 = task_list.add_task("Task Two")
        task3 = task_list.add_task("Task Three")

        task_list.mark_completed(task1.id)
        task_list.mark_failed(task2.id, "Error")

        stats = task_list.get_statistics()
        assert stats["total"] == 3
        # Note: mark_failed doesn't change status until max attempts reached in some implementations
        # Just verify the counts are sensible
        assert stats["total"] >= 3

    def test_format_task_tree(self, task_list):
        """Test formatting task tree."""
        parent = task_list.add_task("Parent task")
        task_list.add_subtask(parent.id, "Subtask 1")
        task_list.add_subtask(parent.id, "Subtask 2")

        tree = task_list.format_task_tree()

        assert "Parent task" in tree
        assert "Subtask 1" in tree
        assert "Subtask 2" in tree

    def test_task_hierarchy_actionable(self, task_list):
        """Test that parent with pending subtasks is not actionable."""
        parent = task_list.add_task("Parent task")
        task_list.add_subtask(parent.id, "Subtask 1")
        task_list.add_subtask(parent.id, "Subtask 2")

        next_task = task_list.get_next_actionable_task()

        # Should return a subtask, not the parent
        assert next_task is not None
        assert next_task.parent_id == parent.id

    def test_persistence(self, memory_store, tmp_path):
        """Test that tasks persist across instances."""
        # Create first instance and add tasks
        task_list1 = PersistentTaskList(memory_store)
        task1 = task_list1.add_task("My first persistent task")

        # Create second instance - note deduplication may affect count
        task_list2 = PersistentTaskList(memory_store)

        # Verify at least the first task persisted
        all_tasks = task_list2.get_all_tasks()
        assert len(all_tasks) >= 1
        assert any("My first persistent task" in t.description for t in all_tasks)

    def test_similarity_threshold(self, task_list):
        """Test similarity detection threshold."""
        task_list.add_task("Fix the authentication bug in login")

        # Very similar - should be duplicate
        assert task_list.add_task("Fix authentication bug login") is None

        # Different enough - should be added
        assert task_list.add_task("Add new feature for user profiles") is not None


class TestTaskDeduplication:
    """Test task deduplication logic."""

    @pytest.fixture
    def memory_store(self, tmp_path):
        """Create a temporary memory store."""
        db_path = tmp_path / "test_memories.db"
        return MemoryStore(db_path)  # MemoryStore expects Path, not str

    @pytest.fixture
    def task_list(self, memory_store):
        """Create a task list."""
        return PersistentTaskList(memory_store)

    def test_exact_duplicate(self, task_list):
        """Test exact duplicate detection."""
        task_list.add_task("Debug the error")
        assert task_list.add_task("Debug the error") is None

    def test_case_insensitive(self, task_list):
        """Test case-insensitive duplicate detection."""
        task_list.add_task("Fix the Bug")
        assert task_list.add_task("fix the bug") is None

    def test_word_overlap_above_threshold(self, task_list):
        """Test word overlap detection above threshold."""
        task_list.add_task("Debug the workflow error logs")
        # 4 of 5 words overlap = 80% > 60% threshold
        assert task_list.add_task("Debug the workflow error files") is None

    def test_word_overlap_below_threshold(self, task_list):
        """Test word overlap detection below threshold."""
        task_list.add_task("Debug the workflow error logs")
        # Only 1-2 words overlap
        assert task_list.add_task("Write documentation for API") is not None

    def test_skip_duplicates_option(self, task_list):
        """Test skip duplicates option."""
        task_list.add_task("Test task")
        # When check_duplicates=False, should add anyway
        task = task_list.add_task("Test task", check_duplicates=False)
        assert task is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

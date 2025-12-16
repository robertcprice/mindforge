#!/usr/bin/env python3
"""
Test script for Task Manager CLI built by Ada + Dev.

This tests the code that emerged from dual-agent collaboration.
"""

import os
import json
from pathlib import Path

# Import the task manager module
from task_manager_cli import Task, TaskManager


def test_task_validation():
    """Test Task class validation."""
    print("Testing Task validation...")

    # Test valid task
    task = Task(
        title="Test Task",
        description="A test task",
        priority="high",
        due_date="2024-12-31",
        tags=["test", "demo"]
    )
    assert task.priority == "high"
    assert task.due_date == "2024-12-31"
    print("  ✓ Valid task created")

    # Test priority case insensitivity
    task2 = Task("Task2", "Desc", "HIGH", "2024-12-31", [])
    assert task2.priority == "high"
    print("  ✓ Priority case normalization works")

    task3 = Task("Task3", "Desc", "Medium", "2024-12-31", [])
    assert task3.priority == "medium"
    print("  ✓ Mixed case priority works")

    # Test invalid priority
    try:
        Task("Bad", "Desc", "urgent", "2024-12-31", [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid priority" in str(e)
        print("  ✓ Invalid priority rejected correctly")

    # Test invalid date
    try:
        Task("Bad", "Desc", "high", "12-31-2024", [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid date format" in str(e)
        print("  ✓ Invalid date rejected correctly")

    # Test invalid date (impossible date)
    try:
        Task("Bad", "Desc", "high", "2024-13-45", [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid date format" in str(e)
        print("  ✓ Impossible date rejected correctly")

    print("  All validation tests passed!\n")


def test_task_manager_crud():
    """Test TaskManager CRUD operations."""
    print("Testing TaskManager CRUD...")

    # Use temp file
    test_file = Path("/tmp/test_tasks.json")
    if test_file.exists():
        test_file.unlink()

    manager = TaskManager(str(test_file))

    # Test add
    task1 = Task("Buy groceries", "Milk, eggs, bread", "high", "2024-12-20", ["shopping", "urgent"])
    manager.add_task(task1)
    assert len(manager.tasks) == 1
    print("  ✓ Task added")

    task2 = Task("Write report", "Q4 summary", "medium", "2024-12-25", ["work"])
    manager.add_task(task2)
    assert len(manager.tasks) == 2
    print("  ✓ Second task added")

    task3 = Task("Clean house", "Deep clean", "low", "2024-12-30", ["home", "chores"])
    manager.add_task(task3)
    assert len(manager.tasks) == 3
    print("  ✓ Third task added")

    # Test persistence
    manager2 = TaskManager(str(test_file))
    assert len(manager2.tasks) == 3
    assert manager2.tasks[0].title == "Buy groceries"
    print("  ✓ Persistence works (load from file)")

    # Test find by title
    found = manager.find_task_by_title("Buy groceries")
    assert found is not None
    assert found.title == "Buy groceries"
    print("  ✓ Find by title works")

    found_case = manager.find_task_by_title("BUY GROCERIES")
    assert found_case is not None
    print("  ✓ Find by title is case-insensitive")

    not_found = manager.find_task_by_title("Nonexistent")
    assert not_found is None
    print("  ✓ Find returns None for missing task")

    # Test update
    updated = Task("Buy groceries", "Milk, eggs, bread, butter", "high", "2024-12-21", ["shopping"])
    manager.update_task(task1, updated)
    assert manager.tasks[0].due_date == "2024-12-21"
    print("  ✓ Task update works")

    # Test remove
    manager.remove_task(manager.tasks[1])  # Remove "Write report"
    assert len(manager.tasks) == 2
    print("  ✓ Task removal works")

    # Cleanup
    test_file.unlink()
    print("  All CRUD tests passed!\n")


def test_task_manager_filtering():
    """Test TaskManager filtering capabilities."""
    print("Testing TaskManager filtering...")

    test_file = Path("/tmp/test_filter_tasks.json")
    if test_file.exists():
        test_file.unlink()

    manager = TaskManager(str(test_file))

    # Add diverse tasks
    manager.add_task(Task("Task A", "", "high", "2024-12-01", ["work", "urgent"]))
    manager.add_task(Task("Task B", "", "high", "2024-12-05", ["personal"]))
    manager.add_task(Task("Task C", "", "medium", "2024-12-10", ["work"]))
    manager.add_task(Task("Task D", "", "low", "2024-12-15", ["personal", "optional"]))
    manager.add_task(Task("Task E", "", "low", "2024-12-20", ["work"]))

    # Filter by single priority
    high = manager.filter_tasks({"priority": ["high"]})
    assert len(high) == 2
    print(f"  ✓ Filter by priority=high: {len(high)} tasks")

    # Filter by multiple priorities
    high_medium = manager.filter_tasks({"priority": ["high", "medium"]})
    assert len(high_medium) == 3
    print(f"  ✓ Filter by priority=high,medium: {len(high_medium)} tasks")

    # Filter by single tag
    work = manager.filter_tasks({"tags": ["work"]})
    assert len(work) == 3
    print(f"  ✓ Filter by tag=work: {len(work)} tasks")

    # Filter by multiple tags
    work_urgent = manager.filter_tasks({"tags": ["work", "urgent"]})
    assert len(work_urgent) == 3  # Tasks with work OR urgent
    print(f"  ✓ Filter by tags=work,urgent: {len(work_urgent)} tasks")

    # Filter by priority AND tags
    high_work = manager.filter_tasks({"priority": ["high"], "tags": ["work"]})
    assert len(high_work) == 1  # Only Task A
    print(f"  ✓ Filter by priority=high AND tag=work: {len(high_work)} tasks")

    # Filter with case-insensitive tags
    work_upper = manager.filter_tasks({"tags": ["WORK"]})
    assert len(work_upper) == 3
    print(f"  ✓ Case-insensitive tag filtering works")

    # Empty filter returns all
    all_tasks = manager.filter_tasks({})
    assert len(all_tasks) == 5
    print(f"  ✓ Empty filter returns all: {len(all_tasks)} tasks")

    # Cleanup
    test_file.unlink()
    print("  All filtering tests passed!\n")


def test_task_serialization():
    """Test Task serialization to/from dict."""
    print("Testing Task serialization...")

    original = Task(
        title="Test",
        description="Description",
        priority="medium",
        due_date="2024-06-15",
        tags=["a", "b", "c"]
    )

    # Convert to dict
    d = original.to_dict()
    assert d["title"] == "Test"
    assert d["priority"] == "medium"
    assert d["tags"] == ["a", "b", "c"]
    print("  ✓ to_dict() works")

    # Convert back
    restored = Task.from_dict(d)
    assert restored.title == original.title
    assert restored.priority == original.priority
    assert restored.due_date == original.due_date
    assert restored.tags == original.tags
    print("  ✓ from_dict() works")

    # Test JSON round-trip
    json_str = json.dumps(d)
    parsed = json.loads(json_str)
    restored2 = Task.from_dict(parsed)
    assert restored2.title == original.title
    print("  ✓ JSON round-trip works")

    print("  All serialization tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TESTING TASK MANAGER CLI")
    print("Built by Ada (Architect) + Dev (Developer)")
    print("=" * 60)
    print()

    test_task_validation()
    test_task_manager_crud()
    test_task_manager_filtering()
    test_task_serialization()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

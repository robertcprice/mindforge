#!/usr/bin/env python3
"""
Task Manager CLI - Built by Dual-Agent Collaboration
=====================================================

This task management system was designed and implemented through a collaborative
conversation between two AI agents:

    - Ada (Architect): Proposed clean architecture, validation patterns, and phases
    - Dev (Developer): Refined filtering logic, CLI implementation, and edge cases

The agents iterated through 6 turns of discussion, building on each other's ideas
to create this complete, functional CLI tool.

COLLABORATION SUMMARY:
    Turn 1 (Ada): Proposed Task/TaskManager architecture with JSON persistence
    Turn 2 (Dev): Suggested CLI commands and validation enhancements
    Turn 3 (Ada): Prioritized validation, added validate_priority/validate_date
    Turn 4 (Dev): Improved filtering with case-insensitive matching
    Turn 5 (Ada): Complete CLI implementation with error handling
    Turn 6 (Dev): Final Task class refinements with robust validation

Usage:
    python task_manager_cli.py

Commands:
    add     - Add a new task with title, description, priority, due date, tags
    list    - List all tasks sorted by due date
    filter  - Filter tasks by priority and/or tags
    remove  - Remove a task by title
    update  - Update an existing task
    exit    - Exit the program
"""

import json
from datetime import datetime
from pathlib import Path


class Task:
    """
    Represents a single task with validation.

    Designed by Ada (Architect), refined by Dev (Developer).

    Attributes:
        title: Task title
        description: Detailed description
        priority: One of "high", "medium", "low" (validated)
        due_date: ISO format date string YYYY-MM-DD (validated)
        tags: List of string tags for categorization
    """

    def __init__(self, title: str, description: str, priority: str,
                 due_date: str, tags: list):
        self.title = title
        self.description = description
        self.priority = self._validate_priority(priority)
        self.due_date = self._validate_date(due_date)
        self.tags = [tag.strip() for tag in tags if tag.strip()]

    def _validate_priority(self, priority: str) -> str:
        """
        Validate and normalize priority.

        Ada's design: Enforce strict priority values
        Dev's refinement: Case-insensitive matching
        """
        valid_priorities = ["high", "medium", "low"]
        if priority.lower() not in valid_priorities:
            raise ValueError(
                f"Invalid priority: '{priority}'. Must be one of {valid_priorities}"
            )
        return priority.lower()

    def _validate_date(self, date_str: str) -> str:
        """
        Validate and normalize date format.

        Ada's design: Ensure ISO format (YYYY-MM-DD)
        Dev's refinement: Strict parsing with clear error messages
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format: '{date_str}'. Use ISO format (YYYY-MM-DD)"
            )

    def __repr__(self) -> str:
        return f"Task({self.title}, {self.priority}, {self.due_date}, {self.tags})"

    def to_dict(self) -> dict:
        """Convert task to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "due_date": self.due_date,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Create task from dictionary (JSON deserialization)."""
        return cls(
            title=data["title"],
            description=data["description"],
            priority=data["priority"],
            due_date=data["due_date"],
            tags=data["tags"]
        )


class TaskManager:
    """
    Manages a collection of tasks with JSON persistence.

    Ada's architecture:
        - Centralized task storage
        - JSON file persistence
        - CRUD operations (Create, Read, Update, Delete)

    Dev's enhancements:
        - Case-insensitive filtering
        - Multiple filter support
        - Robust error handling
    """

    def __init__(self, file_path: str = "tasks.json"):
        self.file_path = Path(file_path)
        self.tasks: list[Task] = []
        self.load()

    def load(self) -> None:
        """Load tasks from JSON file."""
        try:
            if self.file_path.exists():
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.tasks = [Task.from_dict(task) for task in data]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load tasks ({e}). Starting fresh.")
            self.tasks = []

    def save(self) -> None:
        """Save tasks to JSON file."""
        with open(self.file_path, "w") as f:
            json.dump([task.to_dict() for task in self.tasks], f, indent=2)

    def add_task(self, task: Task) -> None:
        """Add a new task and save."""
        self.tasks.append(task)
        self.save()

    def remove_task(self, task: Task) -> None:
        """Remove a task and save."""
        self.tasks.remove(task)
        self.save()

    def update_task(self, old_task: Task, new_task: Task) -> None:
        """Replace an existing task with an updated version."""
        index = self.tasks.index(old_task)
        self.tasks[index] = new_task
        self.save()

    def list_tasks(self) -> list[Task]:
        """Return all tasks."""
        return self.tasks

    def filter_tasks(self, filters: dict = None) -> list[Task]:
        """
        Filter tasks by priority and/or tags.

        Dev's implementation:
            - Case-insensitive matching for both priorities and tags
            - Support for multiple priorities (e.g., ["high", "medium"])
            - Support for multiple tags (e.g., ["work", "urgent"])
            - Empty filters return all matching tasks

        Args:
            filters: Dict with optional keys:
                - "priority": list of priority strings
                - "tags": list of tag strings

        Returns:
            List of tasks matching the filters
        """
        if filters is None:
            filters = {}

        priority_filters = filters.get("priority", [])
        tag_filters = filters.get("tags", [])

        # Normalize filters (case-insensitive) - Dev's enhancement
        priority_filters = [p.lower() for p in priority_filters if p]
        tag_filters = [t.lower() for t in tag_filters if t]

        return [
            task for task in self.tasks
            if (not priority_filters or task.priority in priority_filters) and
               (not tag_filters or any(tag.lower() in tag_filters for tag in task.tags))
        ]

    def find_task_by_title(self, title: str) -> Task | None:
        """Find a task by its title (case-insensitive)."""
        for task in self.tasks:
            if task.title.lower() == title.lower():
                return task
        return None


def print_task(task: Task, verbose: bool = False) -> None:
    """Pretty print a task."""
    priority_colors = {
        "high": "\033[91m",    # Red
        "medium": "\033[93m",  # Yellow
        "low": "\033[92m",     # Green
    }
    reset = "\033[0m"

    color = priority_colors.get(task.priority, "")
    print(f"  {color}[{task.priority.upper()}]{reset} {task.title}")
    print(f"           Due: {task.due_date}")
    if task.tags:
        print(f"           Tags: {', '.join(task.tags)}")
    if verbose and task.description:
        print(f"           {task.description}")


def main():
    """
    Main CLI loop - Ada's Phase 2 implementation with Dev's refinements.

    Features:
        - Input validation with clear error messages
        - Error handling for all operations
        - User-friendly prompts
    """
    print("=" * 60)
    print("  TASK MANAGER CLI")
    print("  Built by Ada (Architect) + Dev (Developer)")
    print("  Through Dual-Agent Collaboration")
    print("=" * 60)
    print()

    manager = TaskManager()

    while True:
        print()
        command = input("Command (add/list/filter/remove/update/exit): ").strip().lower()

        if command == "add":
            try:
                print("\n--- Add New Task ---")
                title = input("  Title: ").strip()
                if not title:
                    print("Error: Title cannot be empty.")
                    continue

                description = input("  Description: ").strip()
                priority = input("  Priority (high/medium/low): ").strip()
                due_date = input("  Due date (YYYY-MM-DD): ").strip()
                tags_input = input("  Tags (comma-separated): ").strip()
                tags = [t.strip() for t in tags_input.split(",") if t.strip()]

                task = Task(title, description, priority, due_date, tags)
                manager.add_task(task)
                print(f"\n✓ Task '{title}' added successfully!")

            except ValueError as e:
                print(f"\n✗ Error: {e}")

        elif command == "list":
            tasks = manager.list_tasks()
            if not tasks:
                print("\nNo tasks found.")
            else:
                print(f"\n--- All Tasks ({len(tasks)}) ---")
                for task in sorted(tasks, key=lambda t: t.due_date):
                    print_task(task)

        elif command == "filter":
            print("\n--- Filter Tasks ---")
            filters = {}

            priority = input("  Priority (comma-separated, or Enter for all): ").strip()
            if priority:
                filters["priority"] = [p.strip() for p in priority.split(",")]

            tags = input("  Tags (comma-separated, or Enter for all): ").strip()
            if tags:
                filters["tags"] = [t.strip() for t in tags.split(",")]

            filtered = manager.filter_tasks(filters)
            if not filtered:
                print("\nNo tasks match the filter.")
            else:
                print(f"\n--- Filtered Tasks ({len(filtered)}) ---")
                for task in sorted(filtered, key=lambda t: t.due_date):
                    print_task(task)

        elif command == "remove":
            title = input("\nEnter task title to remove: ").strip()
            task = manager.find_task_by_title(title)

            if task:
                confirm = input(f"Remove '{task.title}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    manager.remove_task(task)
                    print(f"\n✓ Task '{task.title}' removed.")
                else:
                    print("\nRemoval cancelled.")
            else:
                print(f"\n✗ Task '{title}' not found.")

        elif command == "update":
            title = input("\nEnter task title to update: ").strip()
            task = manager.find_task_by_title(title)

            if task:
                print(f"\n--- Updating '{task.title}' ---")
                print("(Press Enter to keep current value)\n")

                try:
                    new_title = input(f"  Title [{task.title}]: ").strip() or task.title
                    new_desc = input(f"  Description [{task.description}]: ").strip() or task.description
                    new_priority = input(f"  Priority [{task.priority}]: ").strip() or task.priority
                    new_date = input(f"  Due date [{task.due_date}]: ").strip() or task.due_date
                    current_tags = ', '.join(task.tags) if task.tags else 'none'
                    new_tags_input = input(f"  Tags [{current_tags}]: ").strip()
                    new_tags = [t.strip() for t in new_tags_input.split(",") if t.strip()] if new_tags_input else task.tags

                    updated_task = Task(new_title, new_desc, new_priority, new_date, new_tags)
                    manager.update_task(task, updated_task)
                    print(f"\n✓ Task updated successfully!")

                except ValueError as e:
                    print(f"\n✗ Error: {e}")
            else:
                print(f"\n✗ Task '{title}' not found.")

        elif command == "exit":
            print("\nGoodbye!")
            break

        else:
            print(f"\n✗ Unknown command: '{command}'")
            print("Available commands: add, list, filter, remove, update, exit")


if __name__ == "__main__":
    main()

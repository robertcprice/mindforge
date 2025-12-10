"""
MindForge Task Scheduler

Manages scheduled tasks for MindForge operations:
- Memory consolidation
- Importance decay
- Cleanup operations
- Health checks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import schedule

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    name: str
    callback: Callable
    interval_minutes: int
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True

    # Execution tracking
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    def run(self) -> bool:
        """Execute the task.

        Returns:
            True if successful, False if error
        """
        try:
            self.callback()
            self.last_run = datetime.now()
            self.run_count += 1
            logger.debug(f"Task '{self.name}' completed (run #{self.run_count})")
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Task '{self.name}' failed: {e}")
            return False


class TaskScheduler:
    """Manages scheduled background tasks.

    Uses the 'schedule' library for time-based scheduling.
    """

    def __init__(self):
        """Initialize task scheduler."""
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scheduler = schedule

    def add_task(
        self,
        name: str,
        callback: Callable,
        interval_minutes: int,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> None:
        """Add a task to the scheduler.

        Args:
            name: Unique task name
            callback: Function to call
            interval_minutes: Interval between runs
            priority: Task priority
        """
        task = ScheduledTask(
            name=name,
            callback=callback,
            interval_minutes=interval_minutes,
            priority=priority,
        )
        self._tasks[name] = task

        # Schedule with the library
        self._scheduler.every(interval_minutes).minutes.do(
            self._run_task, name
        ).tag(name)

        logger.info(f"Scheduled task '{name}' every {interval_minutes} minutes")

    def remove_task(self, name: str) -> bool:
        """Remove a task from the scheduler."""
        if name in self._tasks:
            self._scheduler.clear(name)
            del self._tasks[name]
            logger.info(f"Removed task '{name}'")
            return True
        return False

    def _run_task(self, name: str) -> None:
        """Internal task runner."""
        task = self._tasks.get(name)
        if task and task.enabled:
            task.run()

    def run_now(self, name: str) -> bool:
        """Run a task immediately.

        Args:
            name: Task name

        Returns:
            True if task ran successfully
        """
        task = self._tasks.get(name)
        if task:
            return task.run()
        logger.warning(f"Task '{name}' not found")
        return False

    def enable_task(self, name: str) -> None:
        """Enable a task."""
        if name in self._tasks:
            self._tasks[name].enabled = True

    def disable_task(self, name: str) -> None:
        """Disable a task."""
        if name in self._tasks:
            self._tasks[name].enabled = False

    def start(self, blocking: bool = False) -> None:
        """Start the scheduler.

        Args:
            blocking: If True, blocks the current thread
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        logger.info("Task scheduler started")

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            self._scheduler.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Task scheduler stopped")

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "task_count": len(self._tasks),
            "tasks": [
                {
                    "name": t.name,
                    "interval_minutes": t.interval_minutes,
                    "enabled": t.enabled,
                    "run_count": t.run_count,
                    "error_count": t.error_count,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                }
                for t in self._tasks.values()
            ],
        }

    def list_pending(self) -> list[str]:
        """List pending scheduled jobs."""
        return [str(job) for job in self._scheduler.get_jobs()]


def create_default_tasks(
    consolidate_memory: Callable,
    decay_importance: Callable,
    cleanup_old: Callable,
    health_check: Callable,
) -> list[tuple]:
    """Create default MindForge tasks.

    Args:
        consolidate_memory: Function to consolidate short-term to long-term memory
        decay_importance: Function to decay memory importance
        cleanup_old: Function to cleanup old memories
        health_check: Function to perform health check

    Returns:
        List of (name, callback, interval_minutes, priority) tuples
    """
    return [
        ("memory_consolidation", consolidate_memory, 60, TaskPriority.NORMAL),
        ("importance_decay", decay_importance, 120, TaskPriority.LOW),
        ("memory_cleanup", cleanup_old, 360, TaskPriority.LOW),  # Every 6 hours
        ("health_check", health_check, 5, TaskPriority.HIGH),
    ]

"""
MindForge Daemon

The always-on background service that keeps MindForge thinking.
Coordinates:
- Trigger monitoring
- Task scheduling
- File watching
- Health monitoring
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from mindforge.scheduler.tasks import TaskScheduler, TaskPriority
from mindforge.scheduler.triggers import TriggerManager, Trigger, TriggerType, create_default_triggers

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for watch triggers."""

    def __init__(self, trigger_manager: TriggerManager):
        """Initialize handler.

        Args:
            trigger_manager: TriggerManager to notify of changes
        """
        self.trigger_manager = trigger_manager

    def on_any_event(self, event):
        """Handle any file system event."""
        if not event.is_directory:
            path = Path(event.src_path)
            self.trigger_manager.record_file_change(path)
            logger.debug(f"File change detected: {path}")


@dataclass
class DaemonStatus:
    """Status information for the daemon."""
    running: bool = False
    start_time: Optional[datetime] = None
    thought_count: int = 0
    task_count: int = 0
    trigger_count: int = 0
    last_thought: Optional[datetime] = None
    last_task: Optional[datetime] = None
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0


class MindForgeDaemon:
    """The always-on daemon service for MindForge.

    This daemon:
    - Monitors triggers and fires them when conditions are met
    - Runs scheduled tasks
    - Watches files for changes
    - Generates spontaneous thoughts
    - Maintains system health
    """

    def __init__(
        self,
        thought_callback: Callable = None,
        consolidate_callback: Callable = None,
        decay_callback: Callable = None,
        cleanup_callback: Callable = None,
        watch_paths: list[Path] = None,
    ):
        """Initialize the daemon.

        Args:
            thought_callback: Function to call when generating thoughts
            consolidate_callback: Function for memory consolidation
            decay_callback: Function for importance decay
            cleanup_callback: Function for cleanup
            watch_paths: Directories to watch for changes
        """
        # Callbacks
        self._thought_callback = thought_callback or self._default_thought_callback
        self._consolidate_callback = consolidate_callback or self._default_callback
        self._decay_callback = decay_callback or self._default_callback
        self._cleanup_callback = cleanup_callback or self._default_callback

        # Managers
        self.trigger_manager = TriggerManager()
        self.task_scheduler = TaskScheduler()

        # File watcher
        self._observer: Optional[Observer] = None
        self._watch_paths = watch_paths or []

        # Status
        self.status = DaemonStatus()

        # Control
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_thread: Optional[threading.Thread] = None

        # Setup signal handlers
        self._setup_signals()

        # Setup defaults
        self._setup_default_triggers()
        self._setup_default_tasks()

    def _default_thought_callback(self) -> None:
        """Default thought callback (placeholder)."""
        logger.info("Spontaneous thought generated (no custom callback)")
        self.status.thought_count += 1
        self.status.last_thought = datetime.now()

    def _default_callback(self) -> None:
        """Default callback for tasks."""
        logger.debug("Default task callback executed")

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()

        # Only set handlers in main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    def _setup_default_triggers(self) -> None:
        """Setup default triggers."""
        for trigger in create_default_triggers(self._thought_callback):
            self.trigger_manager.add_trigger(trigger)

        logger.info(f"Setup {len(self.trigger_manager._triggers)} default triggers")

    def _setup_default_tasks(self) -> None:
        """Setup default scheduled tasks."""
        # Memory consolidation every hour
        self.task_scheduler.add_task(
            name="memory_consolidation",
            callback=self._consolidate_callback,
            interval_minutes=60,
            priority=TaskPriority.NORMAL,
        )

        # Importance decay every 2 hours
        self.task_scheduler.add_task(
            name="importance_decay",
            callback=self._decay_callback,
            interval_minutes=120,
            priority=TaskPriority.LOW,
        )

        # Cleanup every 6 hours
        self.task_scheduler.add_task(
            name="memory_cleanup",
            callback=self._cleanup_callback,
            interval_minutes=360,
            priority=TaskPriority.LOW,
        )

        # Health check every 5 minutes
        self.task_scheduler.add_task(
            name="health_check",
            callback=self._health_check,
            interval_minutes=5,
            priority=TaskPriority.HIGH,
        )

        logger.info(f"Setup {len(self.task_scheduler._tasks)} default tasks")

    def _health_check(self) -> None:
        """Perform a health check."""
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
                self.status.errors.append({
                    "time": datetime.now().isoformat(),
                    "type": "high_memory",
                    "message": f"Memory usage at {memory.percent}%",
                })
        except ImportError:
            pass

        logger.debug("Health check completed")

    def _setup_file_watcher(self) -> None:
        """Setup file system watcher."""
        if not self._watch_paths:
            return

        self._observer = Observer()
        handler = FileChangeHandler(self.trigger_manager)

        for path in self._watch_paths:
            if path.exists() and path.is_dir():
                self._observer.schedule(handler, str(path), recursive=True)
                logger.info(f"Watching directory: {path}")

        self._observer.start()

    async def _main_loop(self) -> None:
        """Main daemon loop."""
        logger.info("Daemon main loop started")
        check_interval = 1  # Check triggers every second

        while self._running:
            try:
                # Check triggers
                fired = self.trigger_manager.check_and_fire()
                if fired:
                    self.status.trigger_count += len(fired)
                    logger.debug(f"Triggers fired: {fired}")

                # Small sleep to prevent busy waiting
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.status.errors.append({
                    "time": datetime.now().isoformat(),
                    "type": "main_loop_error",
                    "message": str(e),
                })
                await asyncio.sleep(5)  # Back off on error

    def start(self, blocking: bool = True) -> None:
        """Start the daemon.

        Args:
            blocking: If True, blocks until stop() is called
        """
        if self._running:
            logger.warning("Daemon already running")
            return

        logger.info("=" * 60)
        logger.info("MINDFORGE DAEMON STARTING")
        logger.info("=" * 60)

        self._running = True
        self.status.running = True
        self.status.start_time = datetime.now()

        # Start file watcher
        self._setup_file_watcher()

        # Start task scheduler
        self.task_scheduler.start(blocking=False)

        logger.info("Daemon components started")
        logger.info(f"Triggers: {len(self.trigger_manager._triggers)}")
        logger.info(f"Tasks: {len(self.task_scheduler._tasks)}")
        logger.info(f"Watch paths: {len(self._watch_paths)}")

        if blocking:
            # Run in current thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._main_loop())
            finally:
                self._loop.close()
        else:
            # Run in background thread
            def run_async():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                try:
                    self._loop.run_until_complete(self._main_loop())
                finally:
                    self._loop.close()

            self._main_thread = threading.Thread(target=run_async, daemon=True)
            self._main_thread.start()

    def stop(self) -> None:
        """Stop the daemon."""
        if not self._running:
            return

        logger.info("Stopping daemon...")

        self._running = False
        self.status.running = False

        # Stop file watcher
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        # Stop task scheduler
        self.task_scheduler.stop()

        # Stop main loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._main_thread:
            self._main_thread.join(timeout=5)
            self._main_thread = None

        logger.info("=" * 60)
        logger.info("MINDFORGE DAEMON STOPPED")
        logger.info(f"Uptime: {self.status.uptime_seconds():.0f} seconds")
        logger.info(f"Thoughts: {self.status.thought_count}")
        logger.info(f"Triggers: {self.status.trigger_count}")
        logger.info(f"Errors: {len(self.status.errors)}")
        logger.info("=" * 60)

    def get_status(self) -> dict:
        """Get daemon status as dictionary."""
        return {
            "running": self.status.running,
            "start_time": self.status.start_time.isoformat() if self.status.start_time else None,
            "uptime_seconds": self.status.uptime_seconds(),
            "thought_count": self.status.thought_count,
            "trigger_count": self.status.trigger_count,
            "task_count": self.status.task_count,
            "last_thought": self.status.last_thought.isoformat() if self.status.last_thought else None,
            "error_count": len(self.status.errors),
            "triggers": self.trigger_manager.list_triggers(),
            "tasks": self.task_scheduler.get_status()["tasks"],
        }

    def add_trigger(self, trigger: Trigger) -> None:
        """Add a custom trigger."""
        self.trigger_manager.add_trigger(trigger)

    def add_watch_path(self, path: Path) -> None:
        """Add a path to watch for changes."""
        self._watch_paths.append(path)
        if self._running and self._observer:
            handler = FileChangeHandler(self.trigger_manager)
            self._observer.schedule(handler, str(path), recursive=True)

    def trigger_thought(self) -> None:
        """Manually trigger a thought."""
        self._thought_callback()


def run() -> None:
    """Entry point for running the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting MindForge daemon...")

    daemon = MindForgeDaemon()

    try:
        daemon.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        daemon.stop()


if __name__ == "__main__":
    run()

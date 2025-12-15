"""
Conch Scheduler System

Provides the "always on, always thinking" capability:
- Background daemon for continuous operation
- Time-based triggers for spontaneous thoughts
- Event-driven triggers (file changes, memory thresholds)
- Task scheduling and coordination
"""

from conch.scheduler.daemon import ConchDaemon
from conch.scheduler.triggers import TriggerManager, Trigger, TriggerType
from conch.scheduler.tasks import TaskScheduler

__all__ = ["ConchDaemon", "TriggerManager", "Trigger", "TriggerType", "TaskScheduler"]

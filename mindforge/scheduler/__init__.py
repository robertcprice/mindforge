"""
MindForge Scheduler System

Provides the "always on, always thinking" capability:
- Background daemon for continuous operation
- Time-based triggers for spontaneous thoughts
- Event-driven triggers (file changes, memory thresholds)
- Task scheduling and coordination
"""

from mindforge.scheduler.daemon import MindForgeDaemon
from mindforge.scheduler.triggers import TriggerManager, Trigger, TriggerType
from mindforge.scheduler.tasks import TaskScheduler

__all__ = ["MindForgeDaemon", "TriggerManager", "Trigger", "TriggerType", "TaskScheduler"]

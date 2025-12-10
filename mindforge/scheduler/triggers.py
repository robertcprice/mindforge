"""
MindForge Trigger System

Defines triggers that activate spontaneous thought generation:
- Time-based: Periodic intervals
- Event-based: File changes, system events
- Threshold-based: Memory accumulation, need urgency
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers."""
    TIME_INTERVAL = "time_interval"      # Fire every N minutes
    TIME_SCHEDULE = "time_schedule"      # Fire at specific times
    FILE_CHANGE = "file_change"          # File/directory changed
    MEMORY_THRESHOLD = "memory_threshold"  # Memory count exceeds threshold
    NEED_URGENCY = "need_urgency"        # A need becomes urgent
    IDLE = "idle"                        # No activity for N minutes
    MANUAL = "manual"                    # Manually triggered


@dataclass
class Trigger:
    """Represents a single trigger configuration."""

    trigger_type: TriggerType
    name: str
    callback: Callable[[], None]

    # Configuration based on type
    interval_minutes: int = 30           # For TIME_INTERVAL
    schedule_times: list[str] = None     # For TIME_SCHEDULE (e.g., ["09:00", "14:00"])
    watch_paths: list[Path] = None       # For FILE_CHANGE
    threshold_value: int = 10            # For MEMORY_THRESHOLD
    urgency_threshold: float = 0.8       # For NEED_URGENCY
    idle_minutes: int = 10               # For IDLE

    # State
    enabled: bool = True
    last_fired: Optional[datetime] = None
    fire_count: int = 0

    def __post_init__(self):
        if self.schedule_times is None:
            self.schedule_times = []
        if self.watch_paths is None:
            self.watch_paths = []

    def should_fire(self, context: dict = None) -> bool:
        """Check if this trigger should fire."""
        if not self.enabled:
            return False

        context = context or {}
        now = datetime.now()

        if self.trigger_type == TriggerType.TIME_INTERVAL:
            if self.last_fired is None:
                return True
            elapsed = (now - self.last_fired).total_seconds() / 60
            return elapsed >= self.interval_minutes

        elif self.trigger_type == TriggerType.TIME_SCHEDULE:
            current_time = now.strftime("%H:%M")
            if current_time in self.schedule_times:
                # Only fire once per scheduled time
                if self.last_fired:
                    last_time = self.last_fired.strftime("%H:%M")
                    last_date = self.last_fired.date()
                    if last_time == current_time and last_date == now.date():
                        return False
                return True
            return False

        elif self.trigger_type == TriggerType.MEMORY_THRESHOLD:
            memory_count = context.get("new_memory_count", 0)
            return memory_count >= self.threshold_value

        elif self.trigger_type == TriggerType.NEED_URGENCY:
            max_urgency = context.get("max_need_urgency", 0)
            return max_urgency >= self.urgency_threshold

        elif self.trigger_type == TriggerType.IDLE:
            last_activity = context.get("last_activity_time")
            if last_activity is None:
                return False
            idle_time = (now - last_activity).total_seconds() / 60
            # Fire once when crossing threshold
            if idle_time >= self.idle_minutes:
                if self.last_fired is None:
                    return True
                # Don't fire again until activity resumes
                time_since_last = (now - self.last_fired).total_seconds() / 60
                return time_since_last >= self.idle_minutes * 2

        elif self.trigger_type == TriggerType.FILE_CHANGE:
            changed_files = context.get("changed_files", [])
            for path in self.watch_paths:
                if any(str(path) in str(f) for f in changed_files):
                    return True
            return False

        elif self.trigger_type == TriggerType.MANUAL:
            return context.get("manual_trigger") == self.name

        return False

    def fire(self) -> None:
        """Execute the trigger callback."""
        try:
            self.callback()
            self.last_fired = datetime.now()
            self.fire_count += 1
            logger.debug(f"Trigger '{self.name}' fired (count: {self.fire_count})")
        except Exception as e:
            logger.error(f"Trigger '{self.name}' callback failed: {e}")


class TriggerManager:
    """Manages multiple triggers and checks them against context."""

    def __init__(self):
        """Initialize trigger manager."""
        self._triggers: dict[str, Trigger] = {}
        self._context: dict = {
            "new_memory_count": 0,
            "max_need_urgency": 0,
            "last_activity_time": datetime.now(),
            "changed_files": [],
        }

    def add_trigger(self, trigger: Trigger) -> None:
        """Add a trigger to the manager."""
        self._triggers[trigger.name] = trigger
        logger.info(f"Added trigger: {trigger.name} ({trigger.trigger_type.value})")

    def remove_trigger(self, name: str) -> bool:
        """Remove a trigger by name."""
        if name in self._triggers:
            del self._triggers[name]
            logger.info(f"Removed trigger: {name}")
            return True
        return False

    def enable_trigger(self, name: str) -> None:
        """Enable a trigger."""
        if name in self._triggers:
            self._triggers[name].enabled = True

    def disable_trigger(self, name: str) -> None:
        """Disable a trigger."""
        if name in self._triggers:
            self._triggers[name].enabled = False

    def update_context(self, **kwargs) -> None:
        """Update the context used for trigger evaluation."""
        self._context.update(kwargs)

    def record_activity(self) -> None:
        """Record that activity has occurred."""
        self._context["last_activity_time"] = datetime.now()

    def add_memory(self, count: int = 1) -> None:
        """Record that memories have been added."""
        self._context["new_memory_count"] = self._context.get("new_memory_count", 0) + count

    def reset_memory_count(self) -> None:
        """Reset the new memory count (after consolidation)."""
        self._context["new_memory_count"] = 0

    def set_need_urgency(self, urgency: float) -> None:
        """Update the maximum need urgency."""
        self._context["max_need_urgency"] = urgency

    def record_file_change(self, path: Path) -> None:
        """Record a file change event."""
        changed = self._context.get("changed_files", [])
        changed.append(path)
        # Keep only recent changes
        self._context["changed_files"] = changed[-100:]

    def check_and_fire(self) -> list[str]:
        """Check all triggers and fire those that should activate.

        Returns:
            List of trigger names that fired
        """
        fired = []

        for name, trigger in self._triggers.items():
            if trigger.should_fire(self._context):
                trigger.fire()
                fired.append(name)

        return fired

    def get_trigger(self, name: str) -> Optional[Trigger]:
        """Get a trigger by name."""
        return self._triggers.get(name)

    def list_triggers(self) -> list[dict]:
        """List all triggers with their status."""
        return [
            {
                "name": t.name,
                "type": t.trigger_type.value,
                "enabled": t.enabled,
                "fire_count": t.fire_count,
                "last_fired": t.last_fired.isoformat() if t.last_fired else None,
            }
            for t in self._triggers.values()
        ]

    def fire_manual(self, name: str) -> bool:
        """Manually fire a trigger by name."""
        if name in self._triggers:
            self._triggers[name].fire()
            return True
        return False


def create_default_triggers(thought_callback: Callable) -> list[Trigger]:
    """Create the default set of triggers for MindForge.

    Args:
        thought_callback: Function to call when generating a thought

    Returns:
        List of default triggers
    """
    return [
        # Periodic spontaneous thought
        Trigger(
            trigger_type=TriggerType.TIME_INTERVAL,
            name="periodic_thought",
            callback=thought_callback,
            interval_minutes=30,
        ),

        # Morning reflection
        Trigger(
            trigger_type=TriggerType.TIME_SCHEDULE,
            name="morning_reflection",
            callback=thought_callback,
            schedule_times=["09:00"],
        ),

        # Memory consolidation trigger
        Trigger(
            trigger_type=TriggerType.MEMORY_THRESHOLD,
            name="memory_overflow",
            callback=thought_callback,
            threshold_value=20,
        ),

        # Urgent need response
        Trigger(
            trigger_type=TriggerType.NEED_URGENCY,
            name="urgent_need",
            callback=thought_callback,
            urgency_threshold=0.8,
        ),

        # Idle mind wandering
        Trigger(
            trigger_type=TriggerType.IDLE,
            name="idle_wandering",
            callback=thought_callback,
            idle_minutes=15,
        ),
    ]

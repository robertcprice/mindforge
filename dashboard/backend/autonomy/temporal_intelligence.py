"""
Temporal Intelligence - Time-based optimization for PM-1000 autonomous operation.

Analyzes temporal patterns, optimizes scheduling, tracks deadlines,
and learns from activity history to improve timing decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
import math
import logging
import json

logger = logging.getLogger(__name__)


class TimeWindow(Enum):
    """Time windows for scheduling."""
    IMMEDIATE = "immediate"      # Within minutes
    SHORT = "short"             # Within hours
    MEDIUM = "medium"           # Within a day
    LONG = "long"               # Within a week
    DEFERRED = "deferred"       # No specific deadline


class ActivityLevel(Enum):
    """Activity level categorization."""
    IDLE = 0          # No activity
    LOW = 1           # Light activity
    MODERATE = 2      # Normal activity
    HIGH = 3          # Heavy activity
    PEAK = 4          # Maximum activity


class DayPart(Enum):
    """Parts of the day for pattern analysis."""
    EARLY_MORNING = "early_morning"    # 5-8 AM
    MORNING = "morning"                # 8-12 PM
    AFTERNOON = "afternoon"            # 12-5 PM
    EVENING = "evening"                # 5-9 PM
    NIGHT = "night"                    # 9 PM - 12 AM
    LATE_NIGHT = "late_night"          # 12-5 AM


@dataclass
class TimeSlot:
    """Represents a time slot for scheduling."""
    start: datetime
    end: datetime
    activity_level: ActivityLevel
    available_capacity: float  # 0.0 to 1.0
    reserved_tasks: List[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "activity_level": self.activity_level.name,
            "available_capacity": self.available_capacity,
            "reserved_tasks": self.reserved_tasks,
            "duration_minutes": self.duration_minutes
        }


@dataclass
class Deadline:
    """Represents a deadline for a task or goal."""
    id: str
    description: str
    due_at: datetime
    priority: int  # 1-5, 5 being highest
    is_hard: bool  # Hard deadline vs soft/flexible
    warning_hours: float = 24.0
    escalation_hours: float = 4.0
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completed_at: Optional[datetime] = None

    @property
    def time_remaining(self) -> timedelta:
        if self.completed:
            return timedelta(0)
        return self.due_at - datetime.now()

    @property
    def hours_remaining(self) -> float:
        return self.time_remaining.total_seconds() / 3600

    @property
    def is_overdue(self) -> bool:
        return not self.completed and datetime.now() > self.due_at

    @property
    def is_warning(self) -> bool:
        return not self.completed and self.hours_remaining <= self.warning_hours

    @property
    def is_escalation(self) -> bool:
        return not self.completed and self.hours_remaining <= self.escalation_hours

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "due_at": self.due_at.isoformat(),
            "priority": self.priority,
            "is_hard": self.is_hard,
            "hours_remaining": self.hours_remaining,
            "is_overdue": self.is_overdue,
            "is_warning": self.is_warning,
            "is_escalation": self.is_escalation,
            "completed": self.completed
        }


@dataclass
class TemporalPattern:
    """Learned pattern about time-based behavior."""
    pattern_type: str
    day_part: Optional[DayPart]
    day_of_week: Optional[int]  # 0-6, Monday=0
    frequency: float  # How often this pattern occurs (0-1)
    confidence: float  # Confidence in the pattern (0-1)
    observations: int
    last_observed: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleRecommendation:
    """Recommendation for when to perform a task."""
    task_id: str
    recommended_start: datetime
    recommended_window: TimeWindow
    confidence: float
    reasoning: str
    alternatives: List[Tuple[datetime, float]] = field(default_factory=list)
    constraints_satisfied: List[str] = field(default_factory=list)
    constraints_violated: List[str] = field(default_factory=list)


class ActivityTracker:
    """Tracks activity patterns over time."""

    def __init__(self, history_days: int = 30):
        self.history_days = history_days
        self.activity_log: List[Dict[str, Any]] = []
        self.hourly_patterns: Dict[int, List[float]] = defaultdict(list)
        self.daily_patterns: Dict[int, List[float]] = defaultdict(list)
        self.task_completion_times: Dict[str, List[float]] = defaultdict(list)

    def record_activity(self, activity_type: str, level: ActivityLevel,
                       duration_minutes: float, metadata: Dict[str, Any] = None):
        """Record an activity event."""
        now = datetime.now()
        entry = {
            "timestamp": now.isoformat(),
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "activity_type": activity_type,
            "level": level.value,
            "duration_minutes": duration_minutes,
            "metadata": metadata or {}
        }
        self.activity_log.append(entry)

        # Update patterns
        self.hourly_patterns[now.hour].append(level.value)
        self.daily_patterns[now.weekday()].append(level.value)

        # Prune old data
        self._prune_old_data()

    def record_task_completion(self, task_type: str, duration_minutes: float):
        """Record task completion time for learning."""
        self.task_completion_times[task_type].append(duration_minutes)
        # Keep only recent completions
        if len(self.task_completion_times[task_type]) > 100:
            self.task_completion_times[task_type] = self.task_completion_times[task_type][-100:]

    def get_expected_activity_level(self, target_time: datetime) -> ActivityLevel:
        """Predict activity level for a given time."""
        hour = target_time.hour
        day = target_time.weekday()

        # Get hourly average
        hour_data = self.hourly_patterns.get(hour, [])
        day_data = self.daily_patterns.get(day, [])

        if not hour_data and not day_data:
            return ActivityLevel.MODERATE  # Default

        avg_level = 0
        count = 0

        if hour_data:
            avg_level += sum(hour_data) / len(hour_data)
            count += 1

        if day_data:
            avg_level += sum(day_data) / len(day_data)
            count += 1

        if count > 0:
            avg_level /= count

        # Map to enum
        if avg_level < 0.5:
            return ActivityLevel.IDLE
        elif avg_level < 1.5:
            return ActivityLevel.LOW
        elif avg_level < 2.5:
            return ActivityLevel.MODERATE
        elif avg_level < 3.5:
            return ActivityLevel.HIGH
        else:
            return ActivityLevel.PEAK

    def get_expected_duration(self, task_type: str) -> Optional[float]:
        """Get expected duration for a task type based on history."""
        completions = self.task_completion_times.get(task_type, [])
        if not completions:
            return None
        return sum(completions) / len(completions)

    def get_best_hours(self, count: int = 3) -> List[int]:
        """Get the hours with lowest activity (best for autonomous work)."""
        if not self.hourly_patterns:
            return [2, 3, 4]  # Default to late night

        hour_avgs = {}
        for hour, levels in self.hourly_patterns.items():
            hour_avgs[hour] = sum(levels) / len(levels) if levels else 2.0

        sorted_hours = sorted(hour_avgs.items(), key=lambda x: x[1])
        return [h[0] for h in sorted_hours[:count]]

    def _prune_old_data(self):
        """Remove data older than history_days."""
        cutoff = datetime.now() - timedelta(days=self.history_days)
        cutoff_str = cutoff.isoformat()
        self.activity_log = [
            a for a in self.activity_log
            if a["timestamp"] > cutoff_str
        ]


class DeadlineManager:
    """Manages deadlines and urgency tracking."""

    def __init__(self):
        self.deadlines: Dict[str, Deadline] = {}
        self.deadline_history: List[Dict[str, Any]] = []

    def add_deadline(self, deadline: Deadline) -> bool:
        """Add a new deadline."""
        if deadline.id in self.deadlines:
            logger.warning(f"Deadline {deadline.id} already exists")
            return False

        self.deadlines[deadline.id] = deadline
        logger.info(f"Added deadline: {deadline.id}, due: {deadline.due_at}")
        return True

    def complete_deadline(self, deadline_id: str) -> bool:
        """Mark a deadline as completed."""
        if deadline_id not in self.deadlines:
            return False

        deadline = self.deadlines[deadline_id]
        deadline.completed = True
        deadline.completed_at = datetime.now()

        # Record in history
        self.deadline_history.append({
            "id": deadline_id,
            "due_at": deadline.due_at.isoformat(),
            "completed_at": deadline.completed_at.isoformat(),
            "was_on_time": not deadline.is_overdue,
            "hours_early": -deadline.hours_remaining if not deadline.is_overdue else 0
        })

        return True

    def remove_deadline(self, deadline_id: str) -> bool:
        """Remove a deadline."""
        if deadline_id in self.deadlines:
            del self.deadlines[deadline_id]
            return True
        return False

    def get_active_deadlines(self) -> List[Deadline]:
        """Get all active (incomplete) deadlines sorted by urgency."""
        active = [d for d in self.deadlines.values() if not d.completed]
        return sorted(active, key=lambda d: (not d.is_hard, d.due_at))

    def get_overdue(self) -> List[Deadline]:
        """Get all overdue deadlines."""
        return [d for d in self.deadlines.values() if d.is_overdue]

    def get_warnings(self) -> List[Deadline]:
        """Get all deadlines in warning state."""
        return [d for d in self.deadlines.values() if d.is_warning and not d.is_overdue]

    def get_escalations(self) -> List[Deadline]:
        """Get all deadlines in escalation state."""
        return [d for d in self.deadlines.values() if d.is_escalation and not d.is_overdue]

    def get_urgency_score(self, deadline_id: str) -> float:
        """Calculate urgency score for a deadline (0-1, higher = more urgent)."""
        if deadline_id not in self.deadlines:
            return 0.0

        deadline = self.deadlines[deadline_id]

        if deadline.completed:
            return 0.0

        if deadline.is_overdue:
            return 1.0

        hours = deadline.hours_remaining

        # Exponential urgency as deadline approaches
        if hours <= 1:
            return 0.95
        elif hours <= deadline.escalation_hours:
            return 0.85 + (0.1 * (1 - hours / deadline.escalation_hours))
        elif hours <= deadline.warning_hours:
            return 0.5 + (0.35 * (1 - hours / deadline.warning_hours))
        elif hours <= 72:  # 3 days
            return 0.2 + (0.3 * (1 - hours / 72))
        else:
            return max(0.1, 0.2 * math.exp(-hours / 168))  # Week decay

    def get_metrics(self) -> Dict[str, Any]:
        """Get deadline management metrics."""
        active = self.get_active_deadlines()

        return {
            "total_deadlines": len(self.deadlines),
            "active_deadlines": len(active),
            "overdue_count": len(self.get_overdue()),
            "warning_count": len(self.get_warnings()),
            "escalation_count": len(self.get_escalations()),
            "completion_rate": self._calculate_completion_rate(),
            "avg_hours_early": self._calculate_avg_early()
        }

    def _calculate_completion_rate(self) -> float:
        """Calculate on-time completion rate from history."""
        if not self.deadline_history:
            return 1.0
        on_time = sum(1 for d in self.deadline_history if d["was_on_time"])
        return on_time / len(self.deadline_history)

    def _calculate_avg_early(self) -> float:
        """Calculate average hours completed early."""
        early_completions = [d["hours_early"] for d in self.deadline_history if d["hours_early"] > 0]
        if not early_completions:
            return 0.0
        return sum(early_completions) / len(early_completions)


class ScheduleOptimizer:
    """Optimizes task scheduling based on temporal patterns and constraints."""

    def __init__(self, activity_tracker: ActivityTracker,
                 deadline_manager: DeadlineManager):
        self.activity_tracker = activity_tracker
        self.deadline_manager = deadline_manager
        self.scheduled_tasks: Dict[str, TimeSlot] = {}

        # Configuration
        self.slot_duration_minutes = 30
        self.max_lookahead_hours = 168  # 1 week
        self.preferred_hours = list(range(9, 18))  # 9 AM - 6 PM default

    def set_preferred_hours(self, hours: List[int]):
        """Set preferred working hours."""
        self.preferred_hours = hours

    def recommend_time(self, task_id: str, task_type: str,
                      estimated_minutes: float,
                      constraints: Dict[str, Any] = None) -> ScheduleRecommendation:
        """
        Recommend optimal time to perform a task.

        Args:
            task_id: Unique task identifier
            task_type: Type of task for duration estimation
            estimated_minutes: Estimated task duration
            constraints: Optional scheduling constraints

        Returns:
            ScheduleRecommendation with optimal timing
        """
        constraints = constraints or {}
        now = datetime.now()

        # Check for deadline constraint
        deadline_id = constraints.get("deadline_id")
        deadline = self.deadline_manager.deadlines.get(deadline_id) if deadline_id else None

        # Get historical duration if available
        historical_duration = self.activity_tracker.get_expected_duration(task_type)
        effective_duration = historical_duration or estimated_minutes

        # Generate candidate times
        candidates = self._generate_candidates(
            now,
            effective_duration,
            deadline,
            constraints
        )

        if not candidates:
            # Fallback to immediate if no good slots found
            return ScheduleRecommendation(
                task_id=task_id,
                recommended_start=now,
                recommended_window=TimeWindow.IMMEDIATE,
                confidence=0.3,
                reasoning="No optimal time slots found, recommending immediate execution",
                constraints_violated=["no_optimal_slots"]
            )

        # Score and rank candidates
        scored = []
        for candidate_time in candidates:
            score, reasons = self._score_candidate(
                candidate_time,
                effective_duration,
                deadline,
                constraints
            )
            scored.append((candidate_time, score, reasons))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_time, best_score, best_reasons = scored[0]

        # Determine time window
        hours_until = (best_time - now).total_seconds() / 3600
        window = self._determine_window(hours_until)

        # Build alternatives
        alternatives = [(t, s) for t, s, _ in scored[1:4]]

        # Check constraints
        satisfied, violated = self._check_constraints(
            best_time, effective_duration, deadline, constraints
        )

        return ScheduleRecommendation(
            task_id=task_id,
            recommended_start=best_time,
            recommended_window=window,
            confidence=best_score,
            reasoning="; ".join(best_reasons),
            alternatives=alternatives,
            constraints_satisfied=satisfied,
            constraints_violated=violated
        )

    def reserve_slot(self, task_id: str, start: datetime,
                    duration_minutes: float) -> bool:
        """Reserve a time slot for a task."""
        end = start + timedelta(minutes=duration_minutes)
        activity_level = self.activity_tracker.get_expected_activity_level(start)

        slot = TimeSlot(
            start=start,
            end=end,
            activity_level=activity_level,
            available_capacity=0.0,  # Fully reserved
            reserved_tasks=[task_id]
        )

        self.scheduled_tasks[task_id] = slot
        return True

    def release_slot(self, task_id: str) -> bool:
        """Release a reserved time slot."""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            return True
        return False

    def get_availability(self, start: datetime, end: datetime) -> List[TimeSlot]:
        """Get available time slots in a range."""
        slots = []
        current = start

        while current < end:
            slot_end = min(current + timedelta(minutes=self.slot_duration_minutes), end)

            # Check if any reserved task overlaps
            is_reserved = False
            for task_id, reserved in self.scheduled_tasks.items():
                if (current < reserved.end and slot_end > reserved.start):
                    is_reserved = True
                    break

            if not is_reserved:
                activity_level = self.activity_tracker.get_expected_activity_level(current)
                slot = TimeSlot(
                    start=current,
                    end=slot_end,
                    activity_level=activity_level,
                    available_capacity=1.0 - (activity_level.value / 4.0)
                )
                slots.append(slot)

            current = slot_end

        return slots

    def _generate_candidates(self, start: datetime, duration_minutes: float,
                            deadline: Optional[Deadline],
                            constraints: Dict[str, Any]) -> List[datetime]:
        """Generate candidate start times."""
        candidates = []
        end_limit = start + timedelta(hours=self.max_lookahead_hours)

        if deadline:
            # Don't schedule beyond deadline minus buffer
            buffer = timedelta(minutes=duration_minutes * 1.5)
            end_limit = min(end_limit, deadline.due_at - buffer)

        # Generate slots every 30 minutes
        current = start
        while current < end_limit:
            # Skip if outside preferred hours (unless urgent)
            if current.hour in self.preferred_hours or (deadline and deadline.is_warning):
                candidates.append(current)

            current += timedelta(minutes=self.slot_duration_minutes)

        # Also add best hours from activity patterns
        best_hours = self.activity_tracker.get_best_hours(3)
        for day_offset in range(7):
            for hour in best_hours:
                candidate = start.replace(hour=hour, minute=0, second=0, microsecond=0)
                candidate += timedelta(days=day_offset)
                if start <= candidate < end_limit and candidate not in candidates:
                    candidates.append(candidate)

        return sorted(set(candidates))

    def _score_candidate(self, candidate: datetime, duration_minutes: float,
                        deadline: Optional[Deadline],
                        constraints: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score a candidate time slot."""
        score = 0.5  # Base score
        reasons = []

        # Activity level scoring (lower activity = better)
        activity = self.activity_tracker.get_expected_activity_level(candidate)
        activity_bonus = (4 - activity.value) / 4 * 0.3
        score += activity_bonus
        if activity.value <= 1:
            reasons.append(f"Low activity period ({activity.name})")

        # Preferred hours bonus
        if candidate.hour in self.preferred_hours:
            score += 0.1
            reasons.append("Within preferred hours")

        # Deadline proximity scoring
        if deadline:
            hours_before_deadline = (deadline.due_at - candidate).total_seconds() / 3600
            if hours_before_deadline < duration_minutes / 60:
                score -= 0.5  # Penalty for cutting it too close
                reasons.append("Too close to deadline")
            elif hours_before_deadline < deadline.warning_hours:
                score += 0.15  # Small bonus for getting it done before warning
                reasons.append("Completes before warning period")
            elif hours_before_deadline < 72:  # Within 3 days
                score += 0.1

        # Reserved slot penalty
        end_time = candidate + timedelta(minutes=duration_minutes)
        for task_id, reserved in self.scheduled_tasks.items():
            if candidate < reserved.end and end_time > reserved.start:
                score -= 0.4
                reasons.append(f"Conflicts with {task_id}")
                break

        # Time of day preferences
        hour = candidate.hour
        if 9 <= hour <= 11:  # Morning peak
            score += 0.05
            reasons.append("Morning hours")
        elif 14 <= hour <= 16:  # Afternoon good
            score += 0.03
        elif 22 <= hour or hour <= 5:  # Late night
            if not constraints.get("allow_night", False):
                score -= 0.1
                reasons.append("Late night (not preferred)")

        # Weekend handling
        if candidate.weekday() >= 5:  # Weekend
            if constraints.get("allow_weekend", True):
                score += 0.05  # Slight bonus for keeping weekdays clear
                reasons.append("Weekend slot")
            else:
                score -= 0.3
                reasons.append("Weekend (not allowed)")

        return (max(0, min(1, score)), reasons)

    def _determine_window(self, hours_until: float) -> TimeWindow:
        """Determine time window category."""
        if hours_until <= 0.5:
            return TimeWindow.IMMEDIATE
        elif hours_until <= 4:
            return TimeWindow.SHORT
        elif hours_until <= 24:
            return TimeWindow.MEDIUM
        elif hours_until <= 168:
            return TimeWindow.LONG
        else:
            return TimeWindow.DEFERRED

    def _check_constraints(self, time: datetime, duration: float,
                          deadline: Optional[Deadline],
                          constraints: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Check which constraints are satisfied/violated."""
        satisfied = []
        violated = []

        # Deadline constraint
        if deadline:
            end_time = time + timedelta(minutes=duration)
            if end_time <= deadline.due_at:
                satisfied.append(f"Completes before deadline ({deadline.id})")
            else:
                violated.append(f"Misses deadline ({deadline.id})")

        # Preferred hours
        if time.hour in self.preferred_hours:
            satisfied.append("Within preferred hours")
        elif constraints.get("require_preferred_hours", False):
            violated.append("Outside preferred hours")

        # Weekend constraint
        if time.weekday() < 5:
            satisfied.append("Weekday")
        elif not constraints.get("allow_weekend", True):
            violated.append("Weekend not allowed")

        return satisfied, violated


class TemporalIntelligence:
    """
    Main temporal intelligence system for PM-1000.

    Integrates activity tracking, deadline management, and schedule
    optimization for intelligent time-based decision making.
    """

    def __init__(self):
        self.activity_tracker = ActivityTracker()
        self.deadline_manager = DeadlineManager()
        self.schedule_optimizer = ScheduleOptimizer(
            self.activity_tracker,
            self.deadline_manager
        )
        self.patterns: List[TemporalPattern] = []

        # Learning state
        self.pattern_observations: Dict[str, int] = defaultdict(int)
        self.last_pattern_analysis: Optional[datetime] = None

    def record_activity(self, activity_type: str, level: ActivityLevel,
                       duration_minutes: float, metadata: Dict[str, Any] = None):
        """Record an activity for pattern learning."""
        self.activity_tracker.record_activity(
            activity_type, level, duration_minutes, metadata
        )

    def record_task_completion(self, task_type: str, duration_minutes: float,
                              deadline_id: Optional[str] = None):
        """Record task completion for learning and deadline tracking."""
        self.activity_tracker.record_task_completion(task_type, duration_minutes)

        if deadline_id:
            self.deadline_manager.complete_deadline(deadline_id)

    def add_deadline(self, id: str, description: str, due_at: datetime,
                    priority: int = 3, is_hard: bool = False,
                    warning_hours: float = 24.0) -> bool:
        """Add a new deadline."""
        deadline = Deadline(
            id=id,
            description=description,
            due_at=due_at,
            priority=priority,
            is_hard=is_hard,
            warning_hours=warning_hours
        )
        return self.deadline_manager.add_deadline(deadline)

    def recommend_schedule(self, task_id: str, task_type: str,
                          estimated_minutes: float,
                          deadline_id: Optional[str] = None,
                          constraints: Dict[str, Any] = None) -> ScheduleRecommendation:
        """Get scheduling recommendation for a task."""
        constraints = constraints or {}
        if deadline_id:
            constraints["deadline_id"] = deadline_id

        return self.schedule_optimizer.recommend_time(
            task_id, task_type, estimated_minutes, constraints
        )

    def get_urgency_ranking(self) -> List[Dict[str, Any]]:
        """Get all deadlines ranked by urgency."""
        deadlines = self.deadline_manager.get_active_deadlines()
        ranked = []

        for deadline in deadlines:
            urgency = self.deadline_manager.get_urgency_score(deadline.id)
            ranked.append({
                "deadline": deadline.to_dict(),
                "urgency_score": urgency,
                "status": self._get_deadline_status(deadline)
            })

        ranked.sort(key=lambda x: x["urgency_score"], reverse=True)
        return ranked

    def get_available_slots(self, hours_ahead: int = 24) -> List[TimeSlot]:
        """Get available time slots for the next N hours."""
        start = datetime.now()
        end = start + timedelta(hours=hours_ahead)
        return self.schedule_optimizer.get_availability(start, end)

    def analyze_patterns(self) -> List[TemporalPattern]:
        """Analyze activity patterns from collected data."""
        now = datetime.now()

        # Only re-analyze every hour
        if self.last_pattern_analysis:
            hours_since = (now - self.last_pattern_analysis).total_seconds() / 3600
            if hours_since < 1:
                return self.patterns

        new_patterns = []

        # Analyze hourly patterns
        for hour, levels in self.activity_tracker.hourly_patterns.items():
            if len(levels) >= 5:  # Minimum observations
                avg_level = sum(levels) / len(levels)
                day_part = self._hour_to_day_part(hour)

                pattern = TemporalPattern(
                    pattern_type="hourly_activity",
                    day_part=day_part,
                    day_of_week=None,
                    frequency=len(levels) / 30,  # Normalize by month
                    confidence=min(len(levels) / 20, 1.0),
                    observations=len(levels),
                    last_observed=now,
                    metadata={"hour": hour, "avg_level": avg_level}
                )
                new_patterns.append(pattern)

        # Analyze daily patterns
        for day, levels in self.activity_tracker.daily_patterns.items():
            if len(levels) >= 3:
                avg_level = sum(levels) / len(levels)

                pattern = TemporalPattern(
                    pattern_type="daily_activity",
                    day_part=None,
                    day_of_week=day,
                    frequency=len(levels) / 30,
                    confidence=min(len(levels) / 10, 1.0),
                    observations=len(levels),
                    last_observed=now,
                    metadata={"day": day, "avg_level": avg_level}
                )
                new_patterns.append(pattern)

        # Analyze task duration patterns
        for task_type, durations in self.activity_tracker.task_completion_times.items():
            if len(durations) >= 3:
                avg_duration = sum(durations) / len(durations)
                std_dev = math.sqrt(sum((d - avg_duration) ** 2 for d in durations) / len(durations))

                pattern = TemporalPattern(
                    pattern_type="task_duration",
                    day_part=None,
                    day_of_week=None,
                    frequency=1.0,  # Always relevant
                    confidence=min(len(durations) / 10, 1.0),
                    observations=len(durations),
                    last_observed=now,
                    metadata={
                        "task_type": task_type,
                        "avg_duration": avg_duration,
                        "std_dev": std_dev
                    }
                )
                new_patterns.append(pattern)

        self.patterns = new_patterns
        self.last_pattern_analysis = now

        return self.patterns

    def get_optimal_execution_times(self, task_count: int = 5) -> List[datetime]:
        """Get optimal times for autonomous task execution."""
        best_hours = self.activity_tracker.get_best_hours(3)
        now = datetime.now()
        optimal_times = []

        for day_offset in range(7):
            for hour in best_hours:
                candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                candidate += timedelta(days=day_offset)
                if candidate > now:
                    optimal_times.append(candidate)

                if len(optimal_times) >= task_count:
                    break
            if len(optimal_times) >= task_count:
                break

        return optimal_times[:task_count]

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive temporal intelligence metrics."""
        return {
            "deadline_metrics": self.deadline_manager.get_metrics(),
            "pattern_count": len(self.patterns),
            "activity_log_size": len(self.activity_tracker.activity_log),
            "scheduled_tasks": len(self.schedule_optimizer.scheduled_tasks),
            "best_hours": self.activity_tracker.get_best_hours(3),
            "last_pattern_analysis": (
                self.last_pattern_analysis.isoformat()
                if self.last_pattern_analysis else None
            )
        }

    def _get_deadline_status(self, deadline: Deadline) -> str:
        """Get human-readable deadline status."""
        if deadline.completed:
            return "completed"
        if deadline.is_overdue:
            return "overdue"
        if deadline.is_escalation:
            return "escalation"
        if deadline.is_warning:
            return "warning"
        return "normal"

    def _hour_to_day_part(self, hour: int) -> DayPart:
        """Convert hour to day part."""
        if 5 <= hour < 8:
            return DayPart.EARLY_MORNING
        elif 8 <= hour < 12:
            return DayPart.MORNING
        elif 12 <= hour < 17:
            return DayPart.AFTERNOON
        elif 17 <= hour < 21:
            return DayPart.EVENING
        elif 21 <= hour <= 23:
            return DayPart.NIGHT
        else:
            return DayPart.LATE_NIGHT


# Convenience functions
def create_temporal_intelligence() -> TemporalIntelligence:
    """Create a new temporal intelligence instance."""
    return TemporalIntelligence()


def get_day_part(dt: datetime = None) -> DayPart:
    """Get the current day part."""
    dt = dt or datetime.now()
    hour = dt.hour

    if 5 <= hour < 8:
        return DayPart.EARLY_MORNING
    elif 8 <= hour < 12:
        return DayPart.MORNING
    elif 12 <= hour < 17:
        return DayPart.AFTERNOON
    elif 17 <= hour < 21:
        return DayPart.EVENING
    elif 21 <= hour <= 23:
        return DayPart.NIGHT
    else:
        return DayPart.LATE_NIGHT

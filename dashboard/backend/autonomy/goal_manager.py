#!/usr/bin/env python3
"""
Goal Manager for PM-1000

Manages the generation, decomposition, tracking, and completion of goals.
Goals are high-level objectives that decompose into tasks for execution.

Goal Hierarchy:
- MISSION   → Long-term purpose (e.g., "Improve codebase quality")
- OBJECTIVE → Medium-term goal (e.g., "Increase test coverage to 80%")
- GOAL      → Short-term target (e.g., "Add tests for auth module")
- TASK      → Atomic action (e.g., "Write test_login.py")

Goal Sources:
- User-directed goals (explicit requests)
- Opportunity-derived goals (from scanners)
- Self-generated goals (autonomous discovery)
- Recurring goals (scheduled/periodic)

Goal Lifecycle:
1. PROPOSED  → Goal suggested, awaiting approval
2. APPROVED  → Goal accepted for execution
3. ACTIVE    → Currently being worked on
4. BLOCKED   → Waiting on dependencies or resources
5. COMPLETED → Goal achieved
6. ABANDONED → Goal cancelled or superseded
"""

import uuid
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

from .config_manager import (
    AutonomyConfigManager,
    get_config_manager,
)
from .system_state_manager import (
    SystemStateManager,
    get_state_manager,
)
from .safety_controller import (
    SafetyController,
    SafetyLevel,
    Action,
    ActionType,
    get_safety_controller,
)

logger = get_logger("pm1000.autonomy.goals")


class GoalLevel(Enum):
    """Hierarchy level of a goal."""
    MISSION = 0     # Long-term purpose
    OBJECTIVE = 1   # Medium-term goal
    GOAL = 2        # Short-term target
    TASK = 3        # Atomic action


class GoalStatus(Enum):
    """Status of a goal in its lifecycle."""
    PROPOSED = "proposed"       # Awaiting approval
    APPROVED = "approved"       # Accepted for execution
    ACTIVE = "active"           # Currently being worked
    BLOCKED = "blocked"         # Waiting on something
    COMPLETED = "completed"     # Achieved
    ABANDONED = "abandoned"     # Cancelled


class GoalSource(Enum):
    """Source of goal creation."""
    USER = "user"               # Explicit user request
    OPPORTUNITY = "opportunity" # From opportunity scanners
    AUTONOMOUS = "autonomous"   # Self-generated
    RECURRING = "recurring"     # Scheduled/periodic
    DECOMPOSITION = "decomposition"  # From parent goal


class GoalPriority(Enum):
    """Priority level for goals."""
    CRITICAL = 0    # Must complete immediately
    HIGH = 1        # Complete soon
    MEDIUM = 2      # Normal priority
    LOW = 3         # When resources allow
    BACKGROUND = 4  # Only when idle


@dataclass
class GoalMetadata:
    """Metadata about a goal."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    source: GoalSource = GoalSource.AUTONOMOUS
    source_id: Optional[str] = None  # ID of opportunity, user request, etc.

    estimated_effort_hours: float = 1.0
    actual_effort_hours: float = 0.0

    confidence: float = 0.8  # Confidence in achievability
    value_score: float = 1.0  # Relative value/importance

    tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source": self.source.value,
            "source_id": self.source_id,
            "estimated_effort_hours": self.estimated_effort_hours,
            "actual_effort_hours": self.actual_effort_hours,
            "confidence": self.confidence,
            "value_score": self.value_score,
            "tags": self.tags,
            "notes": self.notes,
        }


@dataclass
class Goal:
    """A goal in the system."""
    id: str
    level: GoalLevel
    title: str
    description: str

    status: GoalStatus = GoalStatus.PROPOSED
    priority: GoalPriority = GoalPriority.MEDIUM

    # Hierarchy
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Goal IDs this depends on
    blocks: List[str] = field(default_factory=list)      # Goal IDs blocked by this

    # Progress tracking
    progress: float = 0.0  # 0-1 completion percentage
    success_criteria: List[str] = field(default_factory=list)
    criteria_met: List[bool] = field(default_factory=list)

    # Metadata
    metadata: GoalMetadata = field(default_factory=GoalMetadata)

    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None

    # For recurring goals
    recurrence_pattern: Optional[str] = None  # e.g., "daily", "weekly"
    last_occurrence: Optional[datetime] = None

    def __post_init__(self):
        if len(self.criteria_met) != len(self.success_criteria):
            self.criteria_met = [False] * len(self.success_criteria)

    @property
    def is_completed(self) -> bool:
        return self.status == GoalStatus.COMPLETED

    @property
    def is_active(self) -> bool:
        return self.status in (GoalStatus.APPROVED, GoalStatus.ACTIVE)

    @property
    def is_blocked(self) -> bool:
        return self.status == GoalStatus.BLOCKED

    @property
    def can_start(self) -> bool:
        """Check if goal can be started (dependencies met)."""
        return self.status == GoalStatus.APPROVED and len(self.depends_on) == 0

    def update_progress(self) -> float:
        """Calculate progress from success criteria."""
        if not self.success_criteria:
            return self.progress

        met_count = sum(1 for m in self.criteria_met if m)
        self.progress = met_count / len(self.success_criteria)
        return self.progress

    def mark_criterion_met(self, index: int, met: bool = True):
        """Mark a success criterion as met or not."""
        if 0 <= index < len(self.criteria_met):
            self.criteria_met[index] = met
            self.update_progress()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.name,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.name,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "progress": self.progress,
            "success_criteria": self.success_criteria,
            "criteria_met": self.criteria_met,
            "metadata": self.metadata.to_dict(),
            "context": self.context,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "recurrence_pattern": self.recurrence_pattern,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
        }


@dataclass
class GoalMetrics:
    """Metrics for goal management."""
    total_goals_created: int = 0
    goals_completed: int = 0
    goals_abandoned: int = 0
    goals_active: int = 0
    goals_blocked: int = 0
    goals_proposed: int = 0

    by_level: Dict[str, int] = field(default_factory=lambda: {l.name: 0 for l in GoalLevel})
    by_source: Dict[str, int] = field(default_factory=lambda: {s.value: 0 for s in GoalSource})

    avg_completion_time_hours: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        total = self.goals_completed + self.goals_abandoned
        return {
            "total_goals_created": self.total_goals_created,
            "goals_completed": self.goals_completed,
            "goals_abandoned": self.goals_abandoned,
            "goals_active": self.goals_active,
            "goals_blocked": self.goals_blocked,
            "goals_proposed": self.goals_proposed,
            "by_level": self.by_level,
            "by_source": self.by_source,
            "avg_completion_time_hours": self.avg_completion_time_hours,
            "success_rate": self.goals_completed / max(1, total),
        }


class GoalGenerator:
    """Generates goals from various sources."""

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default goal templates."""
        self._templates = {
            "improve_test_coverage": {
                "level": GoalLevel.OBJECTIVE,
                "title_template": "Improve test coverage for {target}",
                "description_template": "Increase test coverage for {target} to {coverage}%",
                "success_criteria": [
                    "Identify untested code paths",
                    "Write unit tests for core functions",
                    "Achieve target coverage percentage",
                    "All tests passing",
                ],
                "estimated_effort": 4.0,
            },
            "fix_bug": {
                "level": GoalLevel.GOAL,
                "title_template": "Fix bug: {bug_title}",
                "description_template": "Investigate and fix: {bug_description}",
                "success_criteria": [
                    "Reproduce the bug",
                    "Identify root cause",
                    "Implement fix",
                    "Add regression test",
                    "Verify fix works",
                ],
                "estimated_effort": 2.0,
            },
            "add_feature": {
                "level": GoalLevel.GOAL,
                "title_template": "Add feature: {feature_name}",
                "description_template": "Implement {feature_description}",
                "success_criteria": [
                    "Design implementation approach",
                    "Implement core functionality",
                    "Add tests",
                    "Update documentation",
                    "Code review",
                ],
                "estimated_effort": 8.0,
            },
            "refactor_module": {
                "level": GoalLevel.GOAL,
                "title_template": "Refactor {module_name}",
                "description_template": "Refactor {module_name} to improve {improvement_goal}",
                "success_criteria": [
                    "Analyze current implementation",
                    "Design improved architecture",
                    "Refactor incrementally",
                    "Ensure tests pass",
                    "Document changes",
                ],
                "estimated_effort": 6.0,
            },
            "code_review": {
                "level": GoalLevel.TASK,
                "title_template": "Review: {review_target}",
                "description_template": "Code review for {review_description}",
                "success_criteria": [
                    "Review code changes",
                    "Check for bugs/issues",
                    "Verify tests",
                    "Provide feedback",
                ],
                "estimated_effort": 1.0,
            },
        }

    def generate_from_template(
        self,
        template_name: str,
        params: Dict[str, Any],
        source: GoalSource = GoalSource.AUTONOMOUS,
        source_id: Optional[str] = None,
        priority: GoalPriority = GoalPriority.MEDIUM,
    ) -> Optional[Goal]:
        """Generate a goal from a template."""
        template = self._templates.get(template_name)
        if not template:
            logger.warning(f"Unknown goal template: {template_name}")
            return None

        try:
            goal = Goal(
                id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                level=template["level"],
                title=template["title_template"].format(**params),
                description=template["description_template"].format(**params),
                priority=priority,
                success_criteria=template.get("success_criteria", []).copy(),
                metadata=GoalMetadata(
                    source=source,
                    source_id=source_id,
                    estimated_effort_hours=template.get("estimated_effort", 1.0),
                ),
                context=params.copy(),
            )
            return goal
        except KeyError as e:
            logger.error(f"Missing template parameter: {e}")
            return None

    def generate_from_opportunity(
        self,
        opportunity_type: str,
        opportunity_description: str,
        opportunity_id: str,
        context: Dict[str, Any] = None,
    ) -> Optional[Goal]:
        """Generate a goal from a detected opportunity."""
        context = context or {}

        # Map opportunity types to templates
        type_mapping = {
            "todo_fix": "fix_bug",
            "test_add": "improve_test_coverage",
            "feature_implement": "add_feature",
            "refactor": "refactor_module",
            "code_review": "code_review",
        }

        template_name = type_mapping.get(opportunity_type)
        if template_name:
            # Create params from context
            params = {
                "target": context.get("target", "codebase"),
                "bug_title": opportunity_description[:50],
                "bug_description": opportunity_description,
                "feature_name": context.get("feature", "feature"),
                "feature_description": opportunity_description,
                "module_name": context.get("module", "module"),
                "improvement_goal": context.get("goal", "maintainability"),
                "review_target": context.get("target", "changes"),
                "review_description": opportunity_description,
                "coverage": context.get("coverage", 80),
            }
            return self.generate_from_template(
                template_name,
                params,
                source=GoalSource.OPPORTUNITY,
                source_id=opportunity_id,
            )

        # Generic goal for unknown types
        return Goal(
            id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            level=GoalLevel.GOAL,
            title=f"Address: {opportunity_description[:50]}",
            description=opportunity_description,
            metadata=GoalMetadata(
                source=GoalSource.OPPORTUNITY,
                source_id=opportunity_id,
            ),
            context=context or {},
        )


class GoalDecomposer:
    """Decomposes high-level goals into sub-goals and tasks."""

    def decompose(self, goal: Goal, max_depth: int = 2) -> List[Goal]:
        """Decompose a goal into sub-goals."""
        if goal.level.value >= GoalLevel.TASK.value:
            return []  # Tasks cannot be decomposed further

        sub_goals = []

        # Decompose based on success criteria
        for i, criterion in enumerate(goal.success_criteria):
            sub_goal = Goal(
                id=f"{goal.id}_sub_{i}",
                level=GoalLevel(min(goal.level.value + 1, GoalLevel.TASK.value)),
                title=criterion,
                description=f"Complete: {criterion}",
                parent_id=goal.id,
                priority=goal.priority,
                metadata=GoalMetadata(
                    source=GoalSource.DECOMPOSITION,
                    source_id=goal.id,
                    estimated_effort_hours=goal.metadata.estimated_effort_hours / max(1, len(goal.success_criteria)),
                ),
                context=goal.context.copy(),
            )
            sub_goals.append(sub_goal)

        # Set up dependencies (sequential by default)
        for i in range(1, len(sub_goals)):
            sub_goals[i].depends_on.append(sub_goals[i-1].id)
            sub_goals[i-1].blocks.append(sub_goals[i].id)

        # Update parent
        goal.child_ids = [sg.id for sg in sub_goals]

        return sub_goals


class GoalManager:
    """
    Manages the complete lifecycle of goals.

    Handles goal creation, approval, tracking, and completion.
    """

    def __init__(
        self,
        config_manager: Optional[AutonomyConfigManager] = None,
        state_manager: Optional[SystemStateManager] = None,
        safety_controller: Optional[SafetyController] = None,
    ):
        self._config = config_manager or get_config_manager()
        self._state = state_manager or get_state_manager()
        self._safety = safety_controller or get_safety_controller()

        # Goal storage
        self._goals: Dict[str, Goal] = {}
        self._lock = threading.RLock()

        # Components
        self._generator = GoalGenerator()
        self._decomposer = GoalDecomposer()

        # Metrics
        self._metrics = GoalMetrics()

        # Callbacks
        self._on_goal_created: List[Callable[[Goal], None]] = []
        self._on_goal_completed: List[Callable[[Goal], None]] = []
        self._on_goal_status_change: List[Callable[[Goal, GoalStatus, GoalStatus], None]] = []

        logger.info("GoalManager initialized")

    # =========================================================================
    # Goal Creation
    # =========================================================================

    def create_goal(
        self,
        title: str,
        description: str,
        level: GoalLevel = GoalLevel.GOAL,
        priority: GoalPriority = GoalPriority.MEDIUM,
        source: GoalSource = GoalSource.USER,
        success_criteria: List[str] = None,
        parent_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        auto_approve: bool = False,
    ) -> Goal:
        """Create a new goal."""
        with self._lock:
            goal = Goal(
                id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                level=level,
                title=title,
                description=description,
                priority=priority,
                parent_id=parent_id,
                success_criteria=success_criteria or [],
                metadata=GoalMetadata(source=source),
                context=context or {},
            )

            self._goals[goal.id] = goal

            # Update parent if exists
            if parent_id and parent_id in self._goals:
                self._goals[parent_id].child_ids.append(goal.id)

            # Update metrics
            self._metrics.total_goals_created += 1
            self._metrics.goals_proposed += 1
            self._metrics.by_level[level.name] = self._metrics.by_level.get(level.name, 0) + 1
            self._metrics.by_source[source.value] = self._metrics.by_source.get(source.value, 0) + 1

            # Notify callbacks
            for callback in self._on_goal_created:
                try:
                    callback(goal)
                except Exception as e:
                    logger.error(f"Goal created callback error: {e}")

            # Auto-approve if requested
            if auto_approve:
                self.approve_goal(goal.id)

            logger.info(f"Created goal {goal.id}: {title}")
            return goal

    def create_from_template(
        self,
        template_name: str,
        params: Dict[str, Any],
        auto_approve: bool = False,
    ) -> Optional[Goal]:
        """Create a goal from a template."""
        goal = self._generator.generate_from_template(template_name, params)
        if goal:
            with self._lock:
                self._goals[goal.id] = goal
                self._metrics.total_goals_created += 1
                self._metrics.goals_proposed += 1

                if auto_approve:
                    self.approve_goal(goal.id)

            logger.info(f"Created goal from template {template_name}: {goal.id}")
        return goal

    def create_from_opportunity(
        self,
        opportunity_type: str,
        opportunity_description: str,
        opportunity_id: str,
        context: Dict[str, Any] = None,
    ) -> Optional[Goal]:
        """Create a goal from an opportunity."""
        goal = self._generator.generate_from_opportunity(
            opportunity_type, opportunity_description, opportunity_id, context
        )
        if goal:
            with self._lock:
                self._goals[goal.id] = goal
                self._metrics.total_goals_created += 1
                self._metrics.goals_proposed += 1

            logger.info(f"Created goal from opportunity {opportunity_id}: {goal.id}")
        return goal

    # =========================================================================
    # Goal Lifecycle
    # =========================================================================

    def _set_status(self, goal: Goal, new_status: GoalStatus):
        """Set goal status and trigger callbacks."""
        old_status = goal.status
        goal.status = new_status
        goal.metadata.updated_at = datetime.now()

        # Update metrics
        status_counts = {
            GoalStatus.PROPOSED: "goals_proposed",
            GoalStatus.APPROVED: None,
            GoalStatus.ACTIVE: "goals_active",
            GoalStatus.BLOCKED: "goals_blocked",
            GoalStatus.COMPLETED: "goals_completed",
            GoalStatus.ABANDONED: "goals_abandoned",
        }

        # Decrement old status count
        old_attr = status_counts.get(old_status)
        if old_attr and hasattr(self._metrics, old_attr):
            setattr(self._metrics, old_attr, max(0, getattr(self._metrics, old_attr) - 1))

        # Increment new status count
        new_attr = status_counts.get(new_status)
        if new_attr and hasattr(self._metrics, new_attr):
            setattr(self._metrics, new_attr, getattr(self._metrics, new_attr) + 1)

        # Notify callbacks
        for callback in self._on_goal_status_change:
            try:
                callback(goal, old_status, new_status)
            except Exception as e:
                logger.error(f"Status change callback error: {e}")

        logger.debug(f"Goal {goal.id} status: {old_status.value} -> {new_status.value}")

    def approve_goal(self, goal_id: str) -> bool:
        """Approve a proposed goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status != GoalStatus.PROPOSED:
                logger.warning(f"Cannot approve goal {goal_id} in status {goal.status.value}")
                return False

            # Safety check
            dials = self._config.get_autonomy_dials()
            if goal.metadata.source == GoalSource.AUTONOMOUS:
                if dials.goal_setting.current_level < 0.3:
                    logger.warning(f"Goal {goal_id} blocked: autonomy level too low for autonomous goals")
                    return False

            goal.metadata.approved_at = datetime.now()
            self._set_status(goal, GoalStatus.APPROVED)

            logger.info(f"Approved goal {goal_id}")
            return True

    def start_goal(self, goal_id: str) -> bool:
        """Start working on an approved goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status != GoalStatus.APPROVED:
                logger.warning(f"Cannot start goal {goal_id} in status {goal.status.value}")
                return False

            # Check dependencies
            if not self._check_dependencies_met(goal):
                self._set_status(goal, GoalStatus.BLOCKED)
                return False

            goal.metadata.started_at = datetime.now()
            self._set_status(goal, GoalStatus.ACTIVE)

            logger.info(f"Started goal {goal_id}")
            return True

    def complete_goal(self, goal_id: str, result: Any = None) -> bool:
        """Mark a goal as completed."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status not in (GoalStatus.ACTIVE, GoalStatus.APPROVED):
                logger.warning(f"Cannot complete goal {goal_id} in status {goal.status.value}")
                return False

            goal.progress = 1.0
            goal.criteria_met = [True] * len(goal.success_criteria)
            goal.result = result
            goal.metadata.completed_at = datetime.now()

            # Calculate actual effort
            if goal.metadata.started_at:
                hours = (goal.metadata.completed_at - goal.metadata.started_at).total_seconds() / 3600
                goal.metadata.actual_effort_hours = hours

            self._set_status(goal, GoalStatus.COMPLETED)

            # Update blocked goals
            self._update_blocked_goals(goal)

            # Update parent progress
            self._update_parent_progress(goal)

            # Notify callbacks
            for callback in self._on_goal_completed:
                try:
                    callback(goal)
                except Exception as e:
                    logger.error(f"Goal completed callback error: {e}")

            logger.info(f"Completed goal {goal_id}")
            return True

    def abandon_goal(self, goal_id: str, reason: str = None) -> bool:
        """Abandon a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
                return False

            goal.error = reason
            goal.metadata.completed_at = datetime.now()
            self._set_status(goal, GoalStatus.ABANDONED)

            # Also abandon child goals
            for child_id in goal.child_ids:
                self.abandon_goal(child_id, f"Parent goal {goal_id} abandoned")

            logger.info(f"Abandoned goal {goal_id}: {reason}")
            return True

    def block_goal(self, goal_id: str, reason: str = None) -> bool:
        """Mark a goal as blocked."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status not in (GoalStatus.APPROVED, GoalStatus.ACTIVE):
                return False

            if reason:
                goal.metadata.notes.append(f"Blocked: {reason}")
            self._set_status(goal, GoalStatus.BLOCKED)

            logger.info(f"Blocked goal {goal_id}: {reason}")
            return True

    def unblock_goal(self, goal_id: str) -> bool:
        """Unblock a goal and resume."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            if goal.status != GoalStatus.BLOCKED:
                return False

            if goal.metadata.started_at:
                self._set_status(goal, GoalStatus.ACTIVE)
            else:
                self._set_status(goal, GoalStatus.APPROVED)

            logger.info(f"Unblocked goal {goal_id}")
            return True

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update goal progress (0-1)."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            goal.progress = max(0.0, min(1.0, progress))
            goal.metadata.updated_at = datetime.now()

            # Auto-complete if progress reaches 100%
            if goal.progress >= 1.0 and goal.status == GoalStatus.ACTIVE:
                self.complete_goal(goal_id)

            self._update_parent_progress(goal)
            return True

    def mark_criterion_met(self, goal_id: str, criterion_index: int, met: bool = True) -> bool:
        """Mark a success criterion as met."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False

            goal.mark_criterion_met(criterion_index, met)
            goal.metadata.updated_at = datetime.now()

            # Auto-complete if all criteria met
            if all(goal.criteria_met) and goal.status == GoalStatus.ACTIVE:
                self.complete_goal(goal_id)

            self._update_parent_progress(goal)
            return True

    def _update_parent_progress(self, goal: Goal):
        """Update parent goal progress based on children."""
        if not goal.parent_id:
            return

        parent = self._goals.get(goal.parent_id)
        if not parent:
            return

        # Calculate progress from children
        children = [self._goals.get(cid) for cid in parent.child_ids]
        children = [c for c in children if c]

        if children:
            parent.progress = sum(c.progress for c in children) / len(children)

    def _check_dependencies_met(self, goal: Goal) -> bool:
        """Check if all dependencies are met."""
        for dep_id in goal.depends_on:
            dep = self._goals.get(dep_id)
            if not dep or dep.status != GoalStatus.COMPLETED:
                return False
        return True

    def _update_blocked_goals(self, completed_goal: Goal):
        """Update goals that were blocked by the completed goal."""
        for blocked_id in completed_goal.blocks:
            blocked = self._goals.get(blocked_id)
            if not blocked:
                continue

            # Remove completed goal from dependencies
            if completed_goal.id in blocked.depends_on:
                blocked.depends_on.remove(completed_goal.id)

            # Check if now unblocked
            if blocked.status == GoalStatus.BLOCKED and self._check_dependencies_met(blocked):
                self.unblock_goal(blocked_id)

    # =========================================================================
    # Goal Decomposition
    # =========================================================================

    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """Decompose a goal into sub-goals."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return []

            sub_goals = self._decomposer.decompose(goal)

            # Store sub-goals
            for sg in sub_goals:
                self._goals[sg.id] = sg
                self._metrics.total_goals_created += 1
                self._metrics.goals_proposed += 1
                self._metrics.by_level[sg.level.name] = self._metrics.by_level.get(sg.level.name, 0) + 1
                self._metrics.by_source[GoalSource.DECOMPOSITION.value] = (
                    self._metrics.by_source.get(GoalSource.DECOMPOSITION.value, 0) + 1
                )

            logger.info(f"Decomposed goal {goal_id} into {len(sub_goals)} sub-goals")
            return sub_goals

    # =========================================================================
    # Goal Queries
    # =========================================================================

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with a given status."""
        with self._lock:
            return [g for g in self._goals.values() if g.status == status]

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return self.get_goals_by_status(GoalStatus.ACTIVE)

    def get_ready_goals(self) -> List[Goal]:
        """Get goals that are approved and ready to start."""
        with self._lock:
            return [
                g for g in self._goals.values()
                if g.status == GoalStatus.APPROVED and self._check_dependencies_met(g)
            ]

    def get_goals_by_priority(self, priority: GoalPriority) -> List[Goal]:
        """Get all goals with a given priority."""
        with self._lock:
            return [g for g in self._goals.values() if g.priority == priority]

    def get_goals_by_level(self, level: GoalLevel) -> List[Goal]:
        """Get all goals at a given level."""
        with self._lock:
            return [g for g in self._goals.values() if g.level == level]

    def get_child_goals(self, parent_id: str) -> List[Goal]:
        """Get all child goals of a parent."""
        parent = self._goals.get(parent_id)
        if not parent:
            return []
        return [self._goals.get(cid) for cid in parent.child_ids if cid in self._goals]

    def get_goal_tree(self, root_id: str) -> Dict[str, Any]:
        """Get a goal and all its descendants as a tree."""
        root = self._goals.get(root_id)
        if not root:
            return {}

        def build_tree(goal: Goal) -> Dict[str, Any]:
            return {
                "goal": goal.to_dict(),
                "children": [
                    build_tree(self._goals[cid])
                    for cid in goal.child_ids
                    if cid in self._goals
                ],
            }

        return build_tree(root)

    def get_next_task_goal(self) -> Optional[Goal]:
        """Get the next task-level goal to execute."""
        with self._lock:
            ready = self.get_ready_goals()
            tasks = [g for g in ready if g.level == GoalLevel.TASK]

            if not tasks:
                return None

            # Sort by priority, then by creation time
            tasks.sort(key=lambda g: (g.priority.value, g.metadata.created_at))
            return tasks[0]

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_goal_created(self, callback: Callable[[Goal], None]):
        """Register callback for goal creation."""
        self._on_goal_created.append(callback)

    def on_goal_completed(self, callback: Callable[[Goal], None]):
        """Register callback for goal completion."""
        self._on_goal_completed.append(callback)

    def on_goal_status_change(self, callback: Callable[[Goal, GoalStatus, GoalStatus], None]):
        """Register callback for goal status changes."""
        self._on_goal_status_change.append(callback)

    # =========================================================================
    # Metrics and Status
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get goal management metrics."""
        with self._lock:
            return self._metrics.to_dict()

    def get_status(self) -> Dict[str, Any]:
        """Get overall goal management status."""
        with self._lock:
            return {
                "total_goals": len(self._goals),
                "active_goals": len(self.get_active_goals()),
                "ready_goals": len(self.get_ready_goals()),
                "blocked_goals": len(self.get_goals_by_status(GoalStatus.BLOCKED)),
                "proposed_goals": len(self.get_goals_by_status(GoalStatus.PROPOSED)),
                "metrics": self._metrics.to_dict(),
            }

    def get_all_goals(self) -> List[Dict[str, Any]]:
        """Get all goals as dictionaries."""
        with self._lock:
            return [g.to_dict() for g in self._goals.values()]


# Global instance
_goal_manager: Optional[GoalManager] = None
_manager_lock = threading.Lock()


def get_goal_manager() -> GoalManager:
    """Get the global goal manager instance."""
    global _goal_manager
    with _manager_lock:
        if _goal_manager is None:
            _goal_manager = GoalManager()
        return _goal_manager


def init_goal_manager(**kwargs) -> GoalManager:
    """Initialize the global goal manager."""
    global _goal_manager
    with _manager_lock:
        _goal_manager = GoalManager(**kwargs)
        return _goal_manager

#!/usr/bin/env python3
"""
Safety Controller for PM-1000 Autonomous Operation

The MOST CRITICAL component - everything depends on this working correctly.

Provides:
- Multiple independent kill switches
- Constraint validation (immutable rules)
- Pre-execution safety checks
- Post-execution auditing
- Emergency stop procedures
- Defense-in-depth safety layers
"""

import os
import signal
import threading
import json
import hashlib
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

logger = get_logger("pm1000.autonomy.safety")


class SafetyLevel(Enum):
    """Safety operation levels."""
    NORMAL = 0          # Full capabilities
    CAUTIOUS = 1        # Extra validation required
    RESTRICTED = 2      # Limited operations only
    READ_ONLY = 3       # No modifications allowed
    EMERGENCY_STOP = 4  # All operations blocked


class ConstraintSeverity(Enum):
    """Severity of constraint violations."""
    CRITICAL = "critical"    # Must block immediately
    HIGH = "high"           # Should block, may warn
    MEDIUM = "medium"       # Warn and log
    LOW = "low"             # Log only


class KillSwitchType(Enum):
    """Types of kill switches."""
    FILE_BASED = "file"             # Check for stop file
    TIME_BASED = "time"             # Max operation duration
    BUDGET_BASED = "budget"         # Resource exhaustion
    FAILURE_BASED = "failure"       # Consecutive failures
    API_BASED = "api"               # API endpoint trigger
    SIGNAL_BASED = "signal"         # Unix signals
    HEARTBEAT_BASED = "heartbeat"   # Missing heartbeat


class ActionType(Enum):
    """Types of actions that can be validated."""
    FILE_CREATE = "file_create"
    FILE_MODIFY = "file_modify"
    FILE_DELETE = "file_delete"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
    API_CALL = "api_call"
    EXTERNAL_COMMUNICATION = "external_communication"
    SYSTEM_COMMAND = "system_command"
    DATABASE_WRITE = "database_write"
    TASK_EXECUTION = "task_execution"


@dataclass
class Action:
    """Represents an action to be validated."""
    action_type: ActionType
    description: str
    target: str  # File path, API endpoint, etc.
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "description": self.description,
            "target": self.target,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    constraint_name: str
    severity: ConstraintSeverity
    message: str
    action: Action
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "severity": self.severity.value,
            "message": self.message,
            "action": self.action.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked,
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    passed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "safety_level": self.safety_level.name,
        }


@dataclass
class KillSwitchStatus:
    """Status of a kill switch."""
    switch_type: KillSwitchType
    triggered: bool
    reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    can_reset: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "switch_type": self.switch_type.value,
            "triggered": self.triggered,
            "reason": self.reason,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "can_reset": self.can_reset,
        }


class Constraint:
    """Base class for safety constraints."""

    def __init__(
        self,
        name: str,
        description: str,
        severity: ConstraintSeverity = ConstraintSeverity.HIGH,
        enabled: bool = True,
        immutable: bool = False
    ):
        self.name = name
        self.description = description
        self.severity = severity
        self.enabled = enabled
        self.immutable = immutable  # Cannot be disabled

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        """Validate an action. Return violation if constraint is violated."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "immutable": self.immutable,
        }


class NoDataDestructionConstraint(Constraint):
    """Never delete databases, files, or data without backup."""

    # Patterns for dangerous file operations
    DANGEROUS_PATTERNS = [
        r".*\.db$",           # Database files
        r".*\.sqlite$",       # SQLite files
        r".*\.sql$",          # SQL files
        r".*backup.*",        # Backup files
        r".*\.env$",          # Environment files
        r".*credentials.*",   # Credential files
        r".*secret.*",        # Secret files
    ]

    def __init__(self):
        super().__init__(
            name="no_data_destruction",
            description="Never delete databases, files, or data without backup",
            severity=ConstraintSeverity.CRITICAL,
            enabled=True,
            immutable=True  # Cannot be disabled
        )

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type != ActionType.FILE_DELETE:
            return None

        target = action.target.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.match(pattern, target):
                return ConstraintViolation(
                    constraint_name=self.name,
                    severity=self.severity,
                    message=f"Attempted to delete protected file: {action.target}",
                    action=action,
                    blocked=True
                )
        return None


class TestsMustPassConstraint(Constraint):
    """Never commit code that breaks existing tests."""

    def __init__(self):
        super().__init__(
            name="tests_must_pass",
            description="Never commit code that breaks existing tests",
            severity=ConstraintSeverity.CRITICAL,
            enabled=True,
            immutable=True
        )

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type != ActionType.GIT_COMMIT:
            return None

        # Check metadata for test results
        test_passed = action.metadata.get("tests_passed", True)
        if not test_passed:
            return ConstraintViolation(
                constraint_name=self.name,
                severity=self.severity,
                message="Attempted to commit with failing tests",
                action=action,
                blocked=True
            )
        return None


class NoSecretsInCodeConstraint(Constraint):
    """Never commit API keys, passwords, or credentials."""

    # Patterns that look like secrets
    SECRET_PATTERNS = [
        r"['\"]?(?:api[_-]?key|apikey)['\"]?\s*[=:]\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
        r"['\"]?(?:secret|password|passwd|pwd)['\"]?\s*[=:]\s*['\"][^'\"]{8,}['\"]",
        r"['\"]?(?:token|auth)['\"]?\s*[=:]\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
        r"sk-[a-zA-Z0-9]{40,}",  # OpenAI API keys
        r"ghp_[a-zA-Z0-9]{36,}",  # GitHub tokens
        r"AKIA[A-Z0-9]{16}",  # AWS access keys
    ]

    def __init__(self):
        super().__init__(
            name="no_secrets_in_code",
            description="Never commit API keys, passwords, or credentials",
            severity=ConstraintSeverity.CRITICAL,
            enabled=True,
            immutable=True
        )
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.SECRET_PATTERNS]

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type not in [ActionType.FILE_CREATE, ActionType.FILE_MODIFY, ActionType.GIT_COMMIT]:
            return None

        content = action.metadata.get("content", "")
        if not content:
            return None

        for pattern in self._compiled_patterns:
            if pattern.search(content):
                return ConstraintViolation(
                    constraint_name=self.name,
                    severity=self.severity,
                    message="Detected potential secret/credential in code",
                    action=action,
                    blocked=True
                )
        return None


class BudgetLimitsConstraint(Constraint):
    """Never exceed daily API budget limits."""

    def __init__(self, daily_limit: float = 50.0):
        super().__init__(
            name="budget_limits",
            description="Never exceed daily API budget limits",
            severity=ConstraintSeverity.CRITICAL,
            enabled=True,
            immutable=True
        )
        self.daily_limit = daily_limit
        self.current_spend = 0.0
        self.last_reset = datetime.now().date()

    def reset_if_new_day(self):
        """Reset spend counter if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset:
            self.current_spend = 0.0
            self.last_reset = today

    def record_spend(self, amount: float):
        """Record API spend."""
        self.reset_if_new_day()
        self.current_spend += amount

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        self.reset_if_new_day()
        return max(0, self.daily_limit - self.current_spend)

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type != ActionType.API_CALL:
            return None

        self.reset_if_new_day()
        estimated_cost = action.metadata.get("estimated_cost", 0.0)

        if self.current_spend + estimated_cost > self.daily_limit:
            return ConstraintViolation(
                constraint_name=self.name,
                severity=self.severity,
                message=f"Would exceed daily budget (${self.current_spend:.2f} + ${estimated_cost:.2f} > ${self.daily_limit:.2f})",
                action=action,
                blocked=True
            )
        return None


class MaxFileChangesConstraint(Constraint):
    """Limit the number of files changed per task."""

    def __init__(self, max_files: int = 20):
        super().__init__(
            name="max_file_changes",
            description=f"Limit file changes to {max_files} per task",
            severity=ConstraintSeverity.MEDIUM,
            enabled=True,
            immutable=False
        )
        self.max_files = max_files
        self._task_changes: Dict[str, int] = {}

    def record_change(self, task_id: str):
        """Record a file change for a task."""
        self._task_changes[task_id] = self._task_changes.get(task_id, 0) + 1

    def reset_task(self, task_id: str):
        """Reset change count for a task."""
        self._task_changes.pop(task_id, None)

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type not in [ActionType.FILE_CREATE, ActionType.FILE_MODIFY, ActionType.FILE_DELETE]:
            return None

        task_id = action.task_id
        if not task_id:
            return None

        current_count = self._task_changes.get(task_id, 0)
        if current_count >= self.max_files:
            return ConstraintViolation(
                constraint_name=self.name,
                severity=self.severity,
                message=f"Task {task_id} has already changed {current_count} files (limit: {self.max_files})",
                action=action,
                blocked=self.severity == ConstraintSeverity.CRITICAL
            )
        return None


class NoProductionDeployConstraint(Constraint):
    """Never deploy directly to production."""

    PRODUCTION_INDICATORS = [
        "production",
        "prod",
        "live",
        "main",
        "master",
    ]

    def __init__(self):
        super().__init__(
            name="no_production_deploy",
            description="Never deploy directly to production",
            severity=ConstraintSeverity.HIGH,
            enabled=True,
            immutable=False
        )

    def validate(self, action: Action) -> Optional[ConstraintViolation]:
        if action.action_type != ActionType.GIT_PUSH:
            return None

        target = action.target.lower()
        for indicator in self.PRODUCTION_INDICATORS:
            if indicator in target:
                return ConstraintViolation(
                    constraint_name=self.name,
                    severity=self.severity,
                    message=f"Attempted to push to production branch: {action.target}",
                    action=action,
                    blocked=True
                )
        return None


class KillSwitch:
    """Base class for kill switches."""

    def __init__(self, switch_type: KillSwitchType, can_reset: bool = True):
        self.switch_type = switch_type
        self.can_reset = can_reset
        self._triggered = False
        self._reason: Optional[str] = None
        self._triggered_at: Optional[datetime] = None

    def check(self) -> KillSwitchStatus:
        """Check if the kill switch should be triggered."""
        raise NotImplementedError

    def trigger(self, reason: str):
        """Manually trigger the kill switch."""
        self._triggered = True
        self._reason = reason
        self._triggered_at = datetime.now()
        logger.critical(f"Kill switch {self.switch_type.value} triggered: {reason}")

    def reset(self) -> bool:
        """Reset the kill switch if allowed."""
        if not self.can_reset:
            logger.warning(f"Cannot reset kill switch {self.switch_type.value}")
            return False
        self._triggered = False
        self._reason = None
        self._triggered_at = None
        logger.info(f"Kill switch {self.switch_type.value} reset")
        return True

    def get_status(self) -> KillSwitchStatus:
        """Get current status."""
        return KillSwitchStatus(
            switch_type=self.switch_type,
            triggered=self._triggered,
            reason=self._reason,
            triggered_at=self._triggered_at,
            can_reset=self.can_reset
        )


class FileBasedKillSwitch(KillSwitch):
    """Kill switch triggered by presence of a file."""

    def __init__(self, file_path: str = "/tmp/PM1000_EMERGENCY_STOP"):
        super().__init__(KillSwitchType.FILE_BASED, can_reset=True)
        self.file_path = Path(file_path)

    def check(self) -> KillSwitchStatus:
        if self.file_path.exists():
            if not self._triggered:
                # Read reason from file if available
                try:
                    reason = self.file_path.read_text().strip() or "Stop file detected"
                except Exception:
                    reason = "Stop file detected"
                self.trigger(reason)
        return self.get_status()

    def reset(self) -> bool:
        if super().reset():
            # Remove the stop file
            try:
                if self.file_path.exists():
                    self.file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove stop file: {e}")
            return True
        return False


class TimeBasedKillSwitch(KillSwitch):
    """Kill switch triggered after max operation duration."""

    def __init__(self, max_hours: int = 24):
        super().__init__(KillSwitchType.TIME_BASED, can_reset=True)
        self.max_hours = max_hours
        self.start_time = datetime.now()

    def check(self) -> KillSwitchStatus:
        if not self._triggered:
            elapsed = datetime.now() - self.start_time
            if elapsed > timedelta(hours=self.max_hours):
                self.trigger(f"Max operation duration exceeded ({self.max_hours} hours)")
        return self.get_status()

    def reset(self) -> bool:
        if super().reset():
            self.start_time = datetime.now()
            return True
        return False


class BudgetBasedKillSwitch(KillSwitch):
    """Kill switch triggered by budget exhaustion."""

    def __init__(self, threshold: float = 0.95):
        super().__init__(KillSwitchType.BUDGET_BASED, can_reset=True)
        self.threshold = threshold
        self.current_usage = 0.0
        self.budget_limit = 1.0

    def set_budget_status(self, current: float, limit: float):
        """Update budget status."""
        self.current_usage = current
        self.budget_limit = limit

    def check(self) -> KillSwitchStatus:
        if not self._triggered and self.budget_limit > 0:
            usage_ratio = self.current_usage / self.budget_limit
            if usage_ratio >= self.threshold:
                self.trigger(f"Budget threshold exceeded ({usage_ratio:.1%} of ${self.budget_limit:.2f})")
        return self.get_status()


class FailureBasedKillSwitch(KillSwitch):
    """Kill switch triggered by consecutive failures."""

    def __init__(self, max_failures: int = 5):
        super().__init__(KillSwitchType.FAILURE_BASED, can_reset=True)
        self.max_failures = max_failures
        self.consecutive_failures = 0

    def record_failure(self):
        """Record a failure."""
        self.consecutive_failures += 1

    def record_success(self):
        """Record a success (resets counter)."""
        self.consecutive_failures = 0

    def check(self) -> KillSwitchStatus:
        if not self._triggered:
            if self.consecutive_failures >= self.max_failures:
                self.trigger(f"Max consecutive failures ({self.max_failures})")
        return self.get_status()

    def reset(self) -> bool:
        if super().reset():
            self.consecutive_failures = 0
            return True
        return False


class HeartbeatBasedKillSwitch(KillSwitch):
    """Kill switch triggered by missing heartbeat."""

    def __init__(self, timeout_seconds: int = 300):
        super().__init__(KillSwitchType.HEARTBEAT_BASED, can_reset=True)
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = datetime.now()

    def heartbeat(self):
        """Record a heartbeat."""
        self.last_heartbeat = datetime.now()

    def check(self) -> KillSwitchStatus:
        if not self._triggered:
            elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
            if elapsed > self.timeout_seconds:
                self.trigger(f"Heartbeat timeout ({elapsed:.0f}s > {self.timeout_seconds}s)")
        return self.get_status()


class SafetyController:
    """
    Central safety controller for PM-1000.

    This is the MOST CRITICAL component. It provides:
    - Multiple independent kill switches
    - Constraint validation
    - Pre/post execution safety checks
    - Emergency stop procedures
    - Defense-in-depth safety layers
    """

    def __init__(
        self,
        kill_file_path: str = "/tmp/PM1000_EMERGENCY_STOP",
        max_operation_hours: int = 24,
        budget_threshold: float = 0.95,
        max_consecutive_failures: int = 5,
        heartbeat_timeout: int = 300
    ):
        self._lock = threading.RLock()
        self._safety_level = SafetyLevel.NORMAL
        self._violations: List[ConstraintViolation] = []
        self._audit_log: List[Dict[str, Any]] = []

        # Initialize kill switches
        self._kill_switches: Dict[KillSwitchType, KillSwitch] = {
            KillSwitchType.FILE_BASED: FileBasedKillSwitch(kill_file_path),
            KillSwitchType.TIME_BASED: TimeBasedKillSwitch(max_operation_hours),
            KillSwitchType.BUDGET_BASED: BudgetBasedKillSwitch(budget_threshold),
            KillSwitchType.FAILURE_BASED: FailureBasedKillSwitch(max_consecutive_failures),
            KillSwitchType.HEARTBEAT_BASED: HeartbeatBasedKillSwitch(heartbeat_timeout),
        }

        # Initialize constraints
        self._constraints: Dict[str, Constraint] = {}
        self._init_default_constraints()

        # Signal handlers
        self._setup_signal_handlers()

        logger.info("SafetyController initialized")

    def _init_default_constraints(self):
        """Initialize default safety constraints."""
        self.add_constraint(NoDataDestructionConstraint())
        self.add_constraint(TestsMustPassConstraint())
        self.add_constraint(NoSecretsInCodeConstraint())
        self.add_constraint(BudgetLimitsConstraint())
        self.add_constraint(MaxFileChangesConstraint())
        self.add_constraint(NoProductionDeployConstraint())

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"Received signal {signal_name}, triggering safety stop")
            self.emergency_stop(f"Signal received: {signal_name}")

        # Only set up in main thread
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Not in main thread
            pass

    # =========================================================================
    # Constraint Management
    # =========================================================================

    def add_constraint(self, constraint: Constraint):
        """Add a safety constraint."""
        with self._lock:
            self._constraints[constraint.name] = constraint
            logger.debug(f"Added constraint: {constraint.name}")

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint (if not immutable)."""
        with self._lock:
            if name in self._constraints:
                if self._constraints[name].immutable:
                    logger.warning(f"Cannot remove immutable constraint: {name}")
                    return False
                del self._constraints[name]
                logger.info(f"Removed constraint: {name}")
                return True
            return False

    def enable_constraint(self, name: str) -> bool:
        """Enable a constraint."""
        with self._lock:
            if name in self._constraints:
                self._constraints[name].enabled = True
                return True
            return False

    def disable_constraint(self, name: str) -> bool:
        """Disable a constraint (if not immutable)."""
        with self._lock:
            if name in self._constraints:
                if self._constraints[name].immutable:
                    logger.warning(f"Cannot disable immutable constraint: {name}")
                    return False
                self._constraints[name].enabled = False
                logger.info(f"Disabled constraint: {name}")
                return True
            return False

    def get_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get all constraints."""
        with self._lock:
            return {name: c.to_dict() for name, c in self._constraints.items()}

    # =========================================================================
    # Kill Switch Management
    # =========================================================================

    def check_kill_switches(self) -> List[KillSwitchStatus]:
        """Check all kill switches."""
        statuses = []
        with self._lock:
            for switch in self._kill_switches.values():
                status = switch.check()
                statuses.append(status)

                if status.triggered:
                    self._safety_level = SafetyLevel.EMERGENCY_STOP
                    logger.critical(f"Kill switch triggered: {status.switch_type.value} - {status.reason}")

        return statuses

    def is_any_kill_switch_triggered(self) -> Tuple[bool, Optional[str]]:
        """Check if any kill switch is triggered."""
        statuses = self.check_kill_switches()
        for status in statuses:
            if status.triggered:
                return True, status.reason
        return False, None

    def trigger_kill_switch(self, switch_type: KillSwitchType, reason: str):
        """Manually trigger a specific kill switch."""
        with self._lock:
            if switch_type in self._kill_switches:
                self._kill_switches[switch_type].trigger(reason)
                self._safety_level = SafetyLevel.EMERGENCY_STOP

    def reset_kill_switch(self, switch_type: KillSwitchType) -> bool:
        """Reset a specific kill switch."""
        with self._lock:
            if switch_type in self._kill_switches:
                return self._kill_switches[switch_type].reset()
            return False

    def reset_all_kill_switches(self) -> Dict[KillSwitchType, bool]:
        """Reset all kill switches that can be reset."""
        results = {}
        with self._lock:
            for switch_type, switch in self._kill_switches.items():
                results[switch_type] = switch.reset()
            # Only reset safety level if all critical switches are reset
            if all(not s.get_status().triggered for s in self._kill_switches.values()):
                self._safety_level = SafetyLevel.NORMAL
        return results

    def get_kill_switch_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all kill switches."""
        with self._lock:
            return {
                switch_type.value: switch.get_status().to_dict()
                for switch_type, switch in self._kill_switches.items()
            }

    # =========================================================================
    # Safety Checks
    # =========================================================================

    def validate_action(self, action: Action) -> SafetyCheckResult:
        """
        Validate an action against all constraints.

        This is the MAIN safety gate. Every action must pass through this.
        """
        violations = []
        warnings = []

        with self._lock:
            # First check kill switches
            kill_triggered, kill_reason = self.is_any_kill_switch_triggered()
            if kill_triggered:
                return SafetyCheckResult(
                    passed=False,
                    violations=[ConstraintViolation(
                        constraint_name="kill_switch",
                        severity=ConstraintSeverity.CRITICAL,
                        message=f"Kill switch active: {kill_reason}",
                        action=action,
                        blocked=True
                    )],
                    safety_level=SafetyLevel.EMERGENCY_STOP
                )

            # Check safety level
            if self._safety_level == SafetyLevel.EMERGENCY_STOP:
                return SafetyCheckResult(
                    passed=False,
                    violations=[ConstraintViolation(
                        constraint_name="emergency_stop",
                        severity=ConstraintSeverity.CRITICAL,
                        message="System is in emergency stop mode",
                        action=action,
                        blocked=True
                    )],
                    safety_level=self._safety_level
                )

            if self._safety_level == SafetyLevel.READ_ONLY:
                if action.action_type in [ActionType.FILE_CREATE, ActionType.FILE_MODIFY,
                                         ActionType.FILE_DELETE, ActionType.GIT_COMMIT,
                                         ActionType.GIT_PUSH, ActionType.DATABASE_WRITE]:
                    return SafetyCheckResult(
                        passed=False,
                        violations=[ConstraintViolation(
                            constraint_name="read_only_mode",
                            severity=ConstraintSeverity.HIGH,
                            message="System is in read-only mode",
                            action=action,
                            blocked=True
                        )],
                        safety_level=self._safety_level
                    )

            # Check all constraints
            for constraint in self._constraints.values():
                if not constraint.enabled:
                    continue

                violation = constraint.validate(action)
                if violation:
                    violations.append(violation)
                    self._violations.append(violation)

                    if violation.blocked:
                        logger.warning(
                            f"Action blocked by constraint {constraint.name}: {violation.message}"
                        )

            # Determine if action passes
            critical_violations = [v for v in violations if v.blocked]
            passed = len(critical_violations) == 0

            # Add warnings for non-blocking violations
            for v in violations:
                if not v.blocked:
                    warnings.append(f"{v.constraint_name}: {v.message}")

        # Audit the check
        self._audit_action(action, violations, passed)

        return SafetyCheckResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            safety_level=self._safety_level
        )

    def pre_execution_check(self, action: Action) -> SafetyCheckResult:
        """
        Pre-execution safety check.
        Call this BEFORE executing any action.
        """
        return self.validate_action(action)

    def post_execution_audit(self, action: Action, success: bool, result: Any = None):
        """
        Post-execution audit.
        Call this AFTER executing an action.
        """
        with self._lock:
            # Update failure tracking
            failure_switch = self._kill_switches.get(KillSwitchType.FAILURE_BASED)
            if isinstance(failure_switch, FailureBasedKillSwitch):
                if success:
                    failure_switch.record_success()
                else:
                    failure_switch.record_failure()

            # Update file change tracking
            file_changes_constraint = self._constraints.get("max_file_changes")
            if isinstance(file_changes_constraint, MaxFileChangesConstraint):
                if action.action_type in [ActionType.FILE_CREATE, ActionType.FILE_MODIFY, ActionType.FILE_DELETE]:
                    if action.task_id:
                        file_changes_constraint.record_change(action.task_id)

            # Audit
            self._audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": action.to_dict(),
                "success": success,
                "type": "post_execution",
            })

    def _audit_action(self, action: Action, violations: List[ConstraintViolation], passed: bool):
        """Record action in audit log."""
        with self._lock:
            self._audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": action.to_dict(),
                "violations": [v.to_dict() for v in violations],
                "passed": passed,
                "type": "pre_execution",
            })

            # Keep last 1000 entries
            self._audit_log = self._audit_log[-1000:]

    # =========================================================================
    # Emergency Operations
    # =========================================================================

    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """
        EMERGENCY STOP - Halt all operations immediately.
        """
        with self._lock:
            self._safety_level = SafetyLevel.EMERGENCY_STOP

            # Trigger all kill switches
            for switch_type in [KillSwitchType.FILE_BASED]:
                if switch_type in self._kill_switches:
                    self._kill_switches[switch_type].trigger(reason)

            # Create stop file
            try:
                Path("/tmp/PM1000_EMERGENCY_STOP").write_text(reason)
            except Exception as e:
                logger.error(f"Failed to create stop file: {e}")

            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

    def set_safety_level(self, level: SafetyLevel):
        """Set the safety level."""
        with self._lock:
            old_level = self._safety_level
            self._safety_level = level
            logger.info(f"Safety level changed: {old_level.name} -> {level.name}")

    def heartbeat(self):
        """Record a heartbeat (call this regularly to prevent timeout)."""
        with self._lock:
            heartbeat_switch = self._kill_switches.get(KillSwitchType.HEARTBEAT_BASED)
            if isinstance(heartbeat_switch, HeartbeatBasedKillSwitch):
                heartbeat_switch.heartbeat()

    def update_budget_status(self, current_spend: float, daily_limit: float):
        """Update budget status for kill switch."""
        with self._lock:
            budget_switch = self._kill_switches.get(KillSwitchType.BUDGET_BASED)
            if isinstance(budget_switch, BudgetBasedKillSwitch):
                budget_switch.set_budget_status(current_spend, daily_limit)

            budget_constraint = self._constraints.get("budget_limits")
            if isinstance(budget_constraint, BudgetLimitsConstraint):
                budget_constraint.current_spend = current_spend
                budget_constraint.daily_limit = daily_limit

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    @property
    def safety_level(self) -> SafetyLevel:
        """Get current safety level."""
        with self._lock:
            return self._safety_level

    def is_safe_to_operate(self) -> Tuple[bool, str]:
        """Check if it's safe to operate."""
        with self._lock:
            # Check kill switches
            kill_triggered, reason = self.is_any_kill_switch_triggered()
            if kill_triggered:
                return False, f"Kill switch triggered: {reason}"

            # Check safety level
            if self._safety_level in [SafetyLevel.EMERGENCY_STOP, SafetyLevel.READ_ONLY]:
                return False, f"Safety level: {self._safety_level.name}"

            return True, "Safe to operate"

    def get_recent_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent constraint violations."""
        with self._lock:
            return [v.to_dict() for v in self._violations[-limit:]]

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        with self._lock:
            return self._audit_log[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status."""
        with self._lock:
            safe, reason = self.is_safe_to_operate()
            return {
                "safe_to_operate": safe,
                "reason": reason,
                "safety_level": self._safety_level.name,
                "kill_switches": self.get_kill_switch_status(),
                "constraints": self.get_constraints(),
                "recent_violations_count": len(self._violations),
                "audit_log_entries": len(self._audit_log),
            }


# Global instance
_safety_controller: Optional[SafetyController] = None
_safety_controller_lock = threading.Lock()


def get_safety_controller() -> SafetyController:
    """Get the global safety controller."""
    global _safety_controller
    with _safety_controller_lock:
        if _safety_controller is None:
            _safety_controller = SafetyController()
        return _safety_controller


def init_safety_controller(**kwargs) -> SafetyController:
    """Initialize the global safety controller."""
    global _safety_controller
    with _safety_controller_lock:
        _safety_controller = SafetyController(**kwargs)
        return _safety_controller


# Convenience functions
def is_safe_to_operate() -> Tuple[bool, str]:
    """Check if it's safe to operate."""
    return get_safety_controller().is_safe_to_operate()


def validate_action(action: Action) -> SafetyCheckResult:
    """Validate an action."""
    return get_safety_controller().validate_action(action)


def emergency_stop(reason: str = "Manual emergency stop"):
    """Trigger emergency stop."""
    get_safety_controller().emergency_stop(reason)


def heartbeat():
    """Record a heartbeat."""
    get_safety_controller().heartbeat()


def safety_check(action_type: ActionType, target: str, **metadata):
    """
    Decorator for safety-checked functions.

    Example:
        @safety_check(ActionType.FILE_MODIFY, "/path/to/file")
        def modify_file():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            action = Action(
                action_type=action_type,
                description=func.__name__,
                target=target,
                metadata=metadata
            )

            result = get_safety_controller().pre_execution_check(action)
            if not result.passed:
                raise SafetyConstraintViolation(
                    f"Safety check failed: {[v.message for v in result.violations]}"
                )

            try:
                return_value = func(*args, **kwargs)
                get_safety_controller().post_execution_audit(action, success=True, result=return_value)
                return return_value
            except Exception as e:
                get_safety_controller().post_execution_audit(action, success=False, result=str(e))
                raise

        return wrapper
    return decorator


class SafetyConstraintViolation(Exception):
    """Exception raised when a safety constraint is violated."""
    pass

#!/usr/bin/env python3
"""
Comprehensive Error Handling Framework for PM-1000 Autonomous Operation

Provides:
- Error taxonomy with categorization
- Recovery strategies per error type
- Error correlation and aggregation
- Graceful degradation modes
- Error reporting and escalation
- Circuit breaker integration
"""

import traceback
import threading
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from enum import Enum
from collections import defaultdict
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

logger = get_logger("pm1000.autonomy.errors")


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = 0       # Informational, no action needed
    INFO = 1        # Minor issue, auto-recovered
    WARNING = 2     # Potential problem, monitoring needed
    ERROR = 3       # Significant error, may need intervention
    CRITICAL = 4    # System-threatening, immediate action required
    FATAL = 5       # System must stop


class ErrorCategory(Enum):
    """Categories of errors for routing recovery strategies."""
    NETWORK = "network"           # Network connectivity issues
    API = "api"                   # External API errors
    RESOURCE = "resource"         # Resource exhaustion (memory, disk, budget)
    VALIDATION = "validation"     # Input/output validation failures
    EXECUTION = "execution"       # Task execution failures
    STATE = "state"               # State corruption or inconsistency
    SECURITY = "security"         # Security violations
    TIMEOUT = "timeout"           # Operation timeouts
    DEPENDENCY = "dependency"     # External dependency failures
    INTERNAL = "internal"         # Internal logic errors
    UNKNOWN = "unknown"           # Unclassified errors


class RecoveryAction(Enum):
    """Actions to take for error recovery."""
    RETRY = "retry"               # Retry the operation
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    FALLBACK = "fallback"         # Use fallback/alternative approach
    SKIP = "skip"                 # Skip this operation, continue
    ROLLBACK = "rollback"         # Rollback to previous state
    DEGRADE = "degrade"           # Enter degraded mode
    ESCALATE = "escalate"         # Escalate to human
    STOP = "stop"                 # Stop current task
    SHUTDOWN = "shutdown"         # Shutdown the system
    LOG_ONLY = "log_only"         # Just log, no action


class DegradationLevel(Enum):
    """System degradation levels."""
    NORMAL = 0          # Full capabilities
    REDUCED = 1         # Some features disabled
    MINIMAL = 2         # Essential features only
    SAFE_MODE = 3       # Read-only, no modifications
    EMERGENCY = 4       # Emergency stop pending


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str                     # What operation was being performed
    component: str                     # Which component raised the error
    task_id: Optional[str] = None      # Related task ID if applicable
    session_id: Optional[str] = None   # Related session ID if applicable
    user_action: Optional[str] = None  # User action that triggered this
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    context: ErrorContext
    recovery_action: Optional[RecoveryAction] = None
    recovery_result: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback[:2000],  # Truncate long tracebacks
            "context": self.context.to_dict(),
            "recovery_action": self.recovery_action.value if self.recovery_action else None,
            "recovery_result": self.recovery_result,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
        }

    @property
    def fingerprint(self) -> str:
        """Generate error fingerprint for deduplication."""
        data = f"{self.category.value}:{self.exception_type}:{self.context.operation}:{self.context.component}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error."""
    action: RecoveryAction
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    fallback_handler: Optional[Callable] = None
    escalation_threshold: int = 3  # Escalate after this many occurrences
    cooldown_period: float = 60.0  # Seconds before resetting retry count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "escalation_threshold": self.escalation_threshold,
            "cooldown_period": self.cooldown_period,
        }


class PM1000Error(Exception):
    """Base exception for PM-1000 errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(operation="unknown", component="unknown")
        self.cause = cause
        self.timestamp = datetime.now()


# Specific error types
class NetworkError(PM1000Error):
    """Network connectivity error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class APIError(PM1000Error):
    """External API error."""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.API, **kwargs)
        self.status_code = status_code


class ResourceExhaustedError(PM1000Error):
    """Resource exhaustion error."""
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, severity=ErrorSeverity.CRITICAL, **kwargs)
        self.resource_type = resource_type


class BudgetExceededError(ResourceExhaustedError):
    """Budget limit exceeded."""
    def __init__(self, message: str, budget_type: str = "api", current: float = 0, limit: float = 0, **kwargs):
        super().__init__(message, resource_type=budget_type, **kwargs)
        self.current = current
        self.limit = limit


class ValidationError(PM1000Error):
    """Validation failure error."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.field = field


class ExecutionError(PM1000Error):
    """Task execution error."""
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.EXECUTION, **kwargs)
        self.task_id = task_id


class StateCorruptionError(PM1000Error):
    """State corruption or inconsistency error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.STATE, severity=ErrorSeverity.CRITICAL, **kwargs)


class SecurityViolationError(PM1000Error):
    """Security violation error."""
    def __init__(self, message: str, violation_type: str = "unknown", **kwargs):
        super().__init__(message, category=ErrorCategory.SECURITY, severity=ErrorSeverity.CRITICAL, **kwargs)
        self.violation_type = violation_type


class TimeoutError(PM1000Error):
    """Operation timeout error."""
    def __init__(self, message: str, timeout_seconds: float = 0, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)
        self.timeout_seconds = timeout_seconds


class DependencyError(PM1000Error):
    """External dependency failure."""
    def __init__(self, message: str, dependency_name: str = "unknown", **kwargs):
        super().__init__(message, category=ErrorCategory.DEPENDENCY, **kwargs)
        self.dependency_name = dependency_name


class SafetyConstraintViolation(PM1000Error):
    """Safety constraint violation error."""
    def __init__(self, message: str, constraint_name: str = "unknown", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.constraint_name = constraint_name


class ErrorAggregator:
    """Aggregates and correlates errors for pattern detection."""

    def __init__(self, window_size: timedelta = timedelta(hours=1)):
        self._errors: List[ErrorRecord] = []
        self._window_size = window_size
        self._lock = threading.Lock()
        self._fingerprint_counts: Dict[str, int] = defaultdict(int)
        self._category_counts: Dict[ErrorCategory, int] = defaultdict(int)

    def add_error(self, record: ErrorRecord):
        """Add an error record."""
        with self._lock:
            self._errors.append(record)
            self._fingerprint_counts[record.fingerprint] += 1
            self._category_counts[record.category] += 1
            self._cleanup_old_errors()

    def _cleanup_old_errors(self):
        """Remove errors outside the window."""
        cutoff = datetime.now() - self._window_size
        old_count = len(self._errors)
        self._errors = [e for e in self._errors if e.timestamp > cutoff]

        # Recalculate counts if significant cleanup occurred
        if old_count - len(self._errors) > 10:
            self._recalculate_counts()

    def _recalculate_counts(self):
        """Recalculate fingerprint and category counts."""
        self._fingerprint_counts.clear()
        self._category_counts.clear()
        for error in self._errors:
            self._fingerprint_counts[error.fingerprint] += 1
            self._category_counts[error.category] += 1

    def get_error_rate(self, category: Optional[ErrorCategory] = None) -> float:
        """Get error rate per minute."""
        with self._lock:
            self._cleanup_old_errors()
            if category:
                count = self._category_counts.get(category, 0)
            else:
                count = len(self._errors)
            window_minutes = self._window_size.total_seconds() / 60
            return count / max(window_minutes, 1)

    def get_frequent_errors(self, threshold: int = 3) -> List[Tuple[str, int]]:
        """Get errors occurring more than threshold times."""
        with self._lock:
            self._cleanup_old_errors()
            return [(fp, count) for fp, count in self._fingerprint_counts.items() if count >= threshold]

    def is_error_spike(self, category: ErrorCategory, threshold: float = 5.0) -> bool:
        """Check if there's an error spike in a category."""
        return self.get_error_rate(category) > threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        with self._lock:
            self._cleanup_old_errors()
            return {
                "total_errors": len(self._errors),
                "window_size_hours": self._window_size.total_seconds() / 3600,
                "errors_by_category": {cat.value: count for cat, count in self._category_counts.items()},
                "unique_error_types": len(self._fingerprint_counts),
                "error_rate_per_minute": self.get_error_rate(),
            }


class ErrorHandler:
    """
    Central error handling system for PM-1000.

    Features:
    - Error categorization and routing
    - Recovery strategy execution
    - Error aggregation and pattern detection
    - Graceful degradation management
    - Escalation to human operators
    """

    # Default recovery strategies by category
    DEFAULT_STRATEGIES: Dict[ErrorCategory, RecoveryStrategy] = {
        ErrorCategory.NETWORK: RecoveryStrategy(
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            max_retries=5,
            backoff_base=2.0,
        ),
        ErrorCategory.API: RecoveryStrategy(
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            max_retries=3,
            backoff_base=2.0,
            escalation_threshold=5,
        ),
        ErrorCategory.RESOURCE: RecoveryStrategy(
            action=RecoveryAction.DEGRADE,
            max_retries=1,
            escalation_threshold=2,
        ),
        ErrorCategory.VALIDATION: RecoveryStrategy(
            action=RecoveryAction.SKIP,
            max_retries=0,
        ),
        ErrorCategory.EXECUTION: RecoveryStrategy(
            action=RecoveryAction.RETRY,
            max_retries=2,
            escalation_threshold=3,
        ),
        ErrorCategory.STATE: RecoveryStrategy(
            action=RecoveryAction.ROLLBACK,
            max_retries=1,
            escalation_threshold=1,
        ),
        ErrorCategory.SECURITY: RecoveryStrategy(
            action=RecoveryAction.STOP,
            max_retries=0,
            escalation_threshold=1,
        ),
        ErrorCategory.TIMEOUT: RecoveryStrategy(
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            max_retries=3,
        ),
        ErrorCategory.DEPENDENCY: RecoveryStrategy(
            action=RecoveryAction.FALLBACK,
            max_retries=2,
        ),
        ErrorCategory.INTERNAL: RecoveryStrategy(
            action=RecoveryAction.ESCALATE,
            max_retries=0,
        ),
        ErrorCategory.UNKNOWN: RecoveryStrategy(
            action=RecoveryAction.LOG_ONLY,
            max_retries=0,
        ),
    }

    def __init__(self):
        self._strategies = dict(self.DEFAULT_STRATEGIES)
        self._aggregator = ErrorAggregator()
        self._error_history: List[ErrorRecord] = []
        self._lock = threading.Lock()
        self._degradation_level = DegradationLevel.NORMAL
        self._retry_counts: Dict[str, Tuple[int, datetime]] = {}  # fingerprint -> (count, last_time)
        self._escalation_handlers: List[Callable[[ErrorRecord], None]] = []
        self._degradation_handlers: List[Callable[[DegradationLevel], None]] = []

    def set_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy):
        """Set recovery strategy for an error category."""
        with self._lock:
            self._strategies[category] = strategy

    def get_strategy(self, category: ErrorCategory) -> RecoveryStrategy:
        """Get recovery strategy for an error category."""
        with self._lock:
            return self._strategies.get(category, self._strategies[ErrorCategory.UNKNOWN])

    def add_escalation_handler(self, handler: Callable[[ErrorRecord], None]):
        """Add handler for escalated errors."""
        self._escalation_handlers.append(handler)

    def add_degradation_handler(self, handler: Callable[[DegradationLevel], None]):
        """Add handler for degradation level changes."""
        self._degradation_handlers.append(handler)

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"err_{uuid.uuid4().hex[:12]}"

    def _classify_exception(self, exc: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify a standard exception into category and severity."""
        exc_type = type(exc).__name__

        # Network errors
        if any(name in exc_type for name in ["Connection", "Network", "Socket", "DNS"]):
            return ErrorCategory.NETWORK, ErrorSeverity.ERROR

        # Timeout errors
        if "Timeout" in exc_type:
            return ErrorCategory.TIMEOUT, ErrorSeverity.WARNING

        # Validation errors
        if any(name in exc_type for name in ["ValueError", "TypeError", "Validation"]):
            return ErrorCategory.VALIDATION, ErrorSeverity.WARNING

        # Resource errors
        if any(name in exc_type for name in ["Memory", "Disk", "Resource", "Quota"]):
            return ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL

        # Security errors
        if any(name in exc_type for name in ["Permission", "Auth", "Security", "Access"]):
            return ErrorCategory.SECURITY, ErrorSeverity.CRITICAL

        # State errors
        if any(name in exc_type for name in ["State", "Integrity", "Corruption"]):
            return ErrorCategory.STATE, ErrorSeverity.CRITICAL

        return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR

    def handle_error(
        self,
        exc: Exception,
        context: Optional[ErrorContext] = None,
        recovery_handler: Optional[Callable] = None
    ) -> ErrorRecord:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            exc: The exception to handle
            context: Error context information
            recovery_handler: Optional custom recovery handler

        Returns:
            ErrorRecord with handling details
        """
        # Determine category and severity
        if isinstance(exc, PM1000Error):
            category = exc.category
            severity = exc.severity
            context = context or exc.context
        else:
            category, severity = self._classify_exception(exc)

        context = context or ErrorContext(operation="unknown", component="unknown")

        # Create error record
        record = ErrorRecord(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(exc),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            context=context,
        )

        # Log the error
        log_method = {
            ErrorSeverity.DEBUG: logger.debug,
            ErrorSeverity.INFO: logger.info,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
            ErrorSeverity.FATAL: logger.critical,
        }.get(severity, logger.error)

        log_method(f"[{record.error_id}] {category.value}: {exc}")

        # Add to aggregator
        self._aggregator.add_error(record)

        # Store in history
        with self._lock:
            self._error_history.append(record)
            # Keep last 1000 errors
            self._error_history = self._error_history[-1000:]

        # Get and execute recovery strategy
        strategy = self.get_strategy(category)
        record.recovery_action = strategy.action

        # Check if we should escalate
        fingerprint = record.fingerprint
        with self._lock:
            retry_info = self._retry_counts.get(fingerprint)
            current_time = datetime.now()

            if retry_info:
                count, last_time = retry_info
                # Reset count if cooldown has passed
                if (current_time - last_time).total_seconds() > strategy.cooldown_period:
                    count = 0
                count += 1
            else:
                count = 1

            self._retry_counts[fingerprint] = (count, current_time)

        # Escalate if threshold reached
        if count >= strategy.escalation_threshold:
            self._escalate(record)

        # Execute recovery
        try:
            if recovery_handler:
                recovery_handler(record)
            elif strategy.fallback_handler:
                strategy.fallback_handler(record)
            else:
                self._execute_recovery(record, strategy)

            record.recovery_result = "success"
            record.resolved = True
            record.resolution_time = datetime.now()

        except Exception as recovery_error:
            record.recovery_result = f"failed: {recovery_error}"
            logger.error(f"Recovery failed for {record.error_id}: {recovery_error}")

        # Check for degradation
        self._check_degradation()

        return record

    def _execute_recovery(self, record: ErrorRecord, strategy: RecoveryStrategy):
        """Execute the recovery strategy."""
        action = strategy.action

        if action == RecoveryAction.LOG_ONLY:
            pass  # Already logged

        elif action == RecoveryAction.SKIP:
            logger.info(f"[{record.error_id}] Skipping operation")

        elif action == RecoveryAction.DEGRADE:
            self._increase_degradation()

        elif action == RecoveryAction.STOP:
            logger.warning(f"[{record.error_id}] Stopping current task")
            raise record  # Re-raise to stop

        elif action == RecoveryAction.SHUTDOWN:
            logger.critical(f"[{record.error_id}] Initiating shutdown")
            self._set_degradation(DegradationLevel.EMERGENCY)

        elif action == RecoveryAction.ESCALATE:
            self._escalate(record)

    def _escalate(self, record: ErrorRecord):
        """Escalate error to human operators."""
        logger.warning(f"[{record.error_id}] ESCALATING to human operators")

        for handler in self._escalation_handlers:
            try:
                handler(record)
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

    def _check_degradation(self):
        """Check if system should enter degraded mode."""
        stats = self._aggregator.get_stats()

        # Check for error spikes
        if self._aggregator.is_error_spike(ErrorCategory.API, threshold=10):
            self._increase_degradation()
        elif self._aggregator.is_error_spike(ErrorCategory.RESOURCE, threshold=3):
            self._set_degradation(DegradationLevel.SAFE_MODE)

        # Check overall error rate
        if stats["error_rate_per_minute"] > 20:
            self._increase_degradation()

    def _increase_degradation(self):
        """Increase degradation level by one step."""
        current = self._degradation_level.value
        if current < DegradationLevel.EMERGENCY.value:
            new_level = DegradationLevel(current + 1)
            self._set_degradation(new_level)

    def _set_degradation(self, level: DegradationLevel):
        """Set degradation level."""
        if level != self._degradation_level:
            old_level = self._degradation_level
            self._degradation_level = level
            logger.warning(f"Degradation level changed: {old_level.name} -> {level.name}")

            for handler in self._degradation_handlers:
                try:
                    handler(level)
                except Exception as e:
                    logger.error(f"Degradation handler error: {e}")

    def reset_degradation(self):
        """Reset degradation level to normal."""
        self._set_degradation(DegradationLevel.NORMAL)

    @property
    def degradation_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._degradation_level

    def is_degraded(self) -> bool:
        """Check if system is in any degraded state."""
        return self._degradation_level != DegradationLevel.NORMAL

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error records."""
        with self._lock:
            return [e.to_dict() for e in self._error_history[-limit:]]

    def get_error_by_id(self, error_id: str) -> Optional[ErrorRecord]:
        """Get error record by ID."""
        with self._lock:
            for error in self._error_history:
                if error.error_id == error_id:
                    return error
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics."""
        aggregator_stats = self._aggregator.get_stats()
        with self._lock:
            return {
                "degradation_level": self._degradation_level.name,
                "total_errors_handled": len(self._error_history),
                "unique_error_types": len(self._retry_counts),
                "escalation_handlers": len(self._escalation_handlers),
                **aggregator_stats,
            }

    def clear_history(self):
        """Clear error history (for testing)."""
        with self._lock:
            self._error_history.clear()
            self._retry_counts.clear()


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None
_error_handler_lock = threading.Lock()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    global _error_handler
    with _error_handler_lock:
        if _error_handler is None:
            _error_handler = ErrorHandler()
        return _error_handler


def handle_error(
    exc: Exception,
    context: Optional[ErrorContext] = None,
    recovery_handler: Optional[Callable] = None
) -> ErrorRecord:
    """Convenience function to handle an error."""
    return get_error_handler().handle_error(exc, context, recovery_handler)


def error_context(operation: str, component: str, **kwargs) -> ErrorContext:
    """Create an error context."""
    return ErrorContext(operation=operation, component=component, additional_data=kwargs)


def with_error_handling(
    operation: str = "unknown",
    component: str = "unknown",
    fallback_value: Any = None,
    reraise: bool = False
):
    """
    Decorator for automatic error handling.

    Args:
        operation: Name of the operation
        component: Name of the component
        fallback_value: Value to return on error
        reraise: Whether to re-raise the exception after handling

    Example:
        @with_error_handling(operation="fetch_data", component="api_client")
        def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=operation or func.__name__,
                    component=component,
                )
                record = handle_error(e, context)

                if reraise:
                    raise
                return fallback_value

        return wrapper
    return decorator


@contextmanager
def error_boundary(
    operation: str,
    component: str,
    fallback_handler: Optional[Callable[[Exception], Any]] = None
):
    """
    Context manager for error boundary.

    Example:
        with error_boundary("process_task", "worker"):
            process_task()
    """
    try:
        yield
    except Exception as e:
        context = ErrorContext(operation=operation, component=component)
        record = handle_error(e, context)

        if fallback_handler:
            fallback_handler(e)
        else:
            raise


def is_degraded() -> bool:
    """Check if system is in degraded state."""
    return get_error_handler().is_degraded()


def get_degradation_level() -> DegradationLevel:
    """Get current degradation level."""
    return get_error_handler().degradation_level

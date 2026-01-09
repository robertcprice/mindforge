#!/usr/bin/env python3
"""
Autonomous Loop for PM-1000

The core sense-think-act loop that drives autonomous operation.

Loop Phases:
1. SENSE  - Observe environment, check for opportunities/issues
2. THINK  - Evaluate options, make decisions
3. DECIDE - Select best action based on constraints
4. ACT    - Execute selected action with safety checks
5. LEARN  - Record outcomes, update knowledge

Safety Integration:
- Pre-loop safety checks (kill switches, budget, constraints)
- Per-action validation through SafetyController
- Post-action auditing and state checkpointing
- Graceful degradation on errors

State Management:
- Full state persistence via SystemStateManager
- Checkpoint before risky operations
- Transaction rollback on failures
- Crash recovery from last good state
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

from .safety_controller import (
    SafetyController,
    SafetyLevel,
    Action,
    ActionType,
    get_safety_controller,
    is_safe_to_operate,
)
from .system_state_manager import (
    SystemStateManager,
    CheckpointType,
    get_state_manager,
)
from .error_framework import (
    ErrorHandler,
    ErrorContext,
    DegradationLevel,
    PM1000Error,
    get_error_handler,
)
from .config_manager import (
    AutonomyConfigManager,
    get_config_manager,
)

logger = get_logger("pm1000.autonomy.loop")


class LoopPhase(Enum):
    """Phases of the autonomous loop."""
    IDLE = "idle"
    SENSING = "sensing"
    THINKING = "thinking"
    DECIDING = "deciding"
    ACTING = "acting"
    LEARNING = "learning"
    WAITING = "waiting"
    ERROR_RECOVERY = "error_recovery"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0    # Must do immediately
    HIGH = 1        # Do soon
    MEDIUM = 2      # Normal priority
    LOW = 3         # When nothing else
    BACKGROUND = 4  # Only if idle


@dataclass
class Opportunity:
    """Represents a detected opportunity for action."""
    id: str
    type: str  # e.g., "todo_fix", "test_add", "doc_update"
    description: str
    source: str  # What detected this opportunity
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_effort: float = 1.0  # Hours
    estimated_value: float = 1.0   # Relative value
    confidence: float = 0.8        # 0-1 confidence in opportunity
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @property
    def value_score(self) -> float:
        """Calculate value/effort ratio adjusted by confidence."""
        if self.estimated_effort <= 0:
            return 0.0
        return (self.estimated_value / self.estimated_effort) * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "priority": self.priority.name,
            "estimated_effort": self.estimated_effort,
            "estimated_value": self.estimated_value,
            "confidence": self.confidence,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "value_score": self.value_score,
        }


@dataclass
class Task:
    """A task to be executed."""
    id: str
    type: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    opportunity_id: Optional[str] = None  # Link to source opportunity
    estimated_duration: float = 60.0  # Seconds
    max_retries: int = 2
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "priority": self.priority.name,
            "opportunity_id": self.opportunity_id,
            "estimated_duration": self.estimated_duration,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "status": self.status,
        }


@dataclass
class LoopMetrics:
    """Metrics for the autonomous loop."""
    iterations: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    opportunities_detected: int = 0
    opportunities_acted_on: int = 0
    total_execution_time: float = 0.0
    last_iteration_time: float = 0.0
    consecutive_failures: int = 0
    last_successful_iteration: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "opportunities_detected": self.opportunities_detected,
            "opportunities_acted_on": self.opportunities_acted_on,
            "total_execution_time": self.total_execution_time,
            "last_iteration_time": self.last_iteration_time,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_iteration": self.last_successful_iteration.isoformat() if self.last_successful_iteration else None,
            "success_rate": self.tasks_completed / max(1, self.tasks_completed + self.tasks_failed),
        }


class OpportunityScanner:
    """Base class for opportunity scanners."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.scan_interval = 300  # Seconds between scans
        self.last_scan: Optional[datetime] = None

    def scan(self) -> List[Opportunity]:
        """Scan for opportunities. Override in subclasses."""
        raise NotImplementedError

    def should_scan(self) -> bool:
        """Check if enough time has passed since last scan."""
        if self.last_scan is None:
            return True
        elapsed = (datetime.now() - self.last_scan).total_seconds()
        return elapsed >= self.scan_interval


class TaskExecutor:
    """Base class for task executors."""

    def __init__(self, name: str):
        self.name = name
        self.supported_task_types: List[str] = []

    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle the task."""
        return task.type in self.supported_task_types

    def execute(self, task: Task) -> Tuple[bool, Any]:
        """Execute the task. Returns (success, result)."""
        raise NotImplementedError


class AutonomousLoop:
    """
    The main autonomous loop controller.

    Implements the sense-think-act cycle with full safety integration.
    """

    def __init__(
        self,
        safety_controller: Optional[SafetyController] = None,
        state_manager: Optional[SystemStateManager] = None,
        error_handler: Optional[ErrorHandler] = None,
        config_manager: Optional[AutonomyConfigManager] = None,
        min_iteration_interval: float = 10.0,  # Minimum seconds between iterations
        max_consecutive_failures: int = 5,
    ):
        self._safety = safety_controller or get_safety_controller()
        self._state = state_manager or get_state_manager()
        self._errors = error_handler or get_error_handler()
        self._config = config_manager or get_config_manager()

        self._min_interval = min_iteration_interval
        self._max_failures = max_consecutive_failures

        # Loop state
        self._running = False
        self._phase = LoopPhase.IDLE
        self._metrics = LoopMetrics()
        self._lock = threading.RLock()

        # Task management
        self._task_queue: List[Task] = []
        self._current_task: Optional[Task] = None
        self._completed_tasks: List[Task] = []

        # Opportunity management
        self._scanners: List[OpportunityScanner] = []
        self._opportunities: List[Opportunity] = []

        # Executors
        self._executors: List[TaskExecutor] = []

        # Callbacks
        self._on_phase_change: List[Callable[[LoopPhase, LoopPhase], None]] = []
        self._on_task_complete: List[Callable[[Task, bool], None]] = []
        self._on_opportunity_detected: List[Callable[[Opportunity], None]] = []

        logger.info("AutonomousLoop initialized")

    # =========================================================================
    # Configuration
    # =========================================================================

    def register_scanner(self, scanner: OpportunityScanner):
        """Register an opportunity scanner."""
        with self._lock:
            self._scanners.append(scanner)
            logger.debug(f"Registered scanner: {scanner.name}")

    def register_executor(self, executor: TaskExecutor):
        """Register a task executor."""
        with self._lock:
            self._executors.append(executor)
            logger.debug(f"Registered executor: {executor.name}")

    def on_phase_change(self, callback: Callable[[LoopPhase, LoopPhase], None]):
        """Register callback for phase changes."""
        self._on_phase_change.append(callback)

    def on_task_complete(self, callback: Callable[[Task, bool], None]):
        """Register callback for task completion."""
        self._on_task_complete.append(callback)

    def on_opportunity_detected(self, callback: Callable[[Opportunity], None]):
        """Register callback for new opportunities."""
        self._on_opportunity_detected.append(callback)

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_phase(self, phase: LoopPhase):
        """Set the current loop phase."""
        old_phase = self._phase
        self._phase = phase

        # Update state manager
        self._state.update_loop_state(phase=phase.value)

        # Notify callbacks
        for callback in self._on_phase_change:
            try:
                callback(old_phase, phase)
            except Exception as e:
                logger.error(f"Phase change callback error: {e}")

        logger.debug(f"Phase: {old_phase.value} -> {phase.value}")

    def _sync_state_to_manager(self):
        """Sync current loop state to the state manager."""
        with self._lock:
            self._state.update_loop_state(
                phase=self._phase.value,
                iteration=self._metrics.iterations,
                current_task_id=self._current_task.id if self._current_task else None,
                consecutive_failures=self._metrics.consecutive_failures,
                total_tasks_completed=self._metrics.tasks_completed,
            )

    # =========================================================================
    # Safety Checks
    # =========================================================================

    def _check_safety(self) -> Tuple[bool, str]:
        """Perform safety checks before iteration."""
        # Check kill switches
        safe, reason = self._safety.is_safe_to_operate()
        if not safe:
            return False, reason

        # Check degradation level
        if self._errors.degradation_level.value >= DegradationLevel.EMERGENCY.value:
            return False, "Emergency degradation level"

        # Check consecutive failures
        if self._metrics.consecutive_failures >= self._max_failures:
            return False, f"Max consecutive failures ({self._max_failures}) reached"

        # Check budget
        budget = self._config.get_resource_budget()
        resource_state = self._state.resource_state
        if resource_state.api_spend_today >= budget.daily_api_budget:
            return False, "Daily API budget exhausted"

        return True, "Safe to operate"

    def _validate_task_action(self, task: Task) -> Tuple[bool, str]:
        """Validate a task action through safety controller."""
        action = Action(
            action_type=ActionType.TASK_EXECUTION,
            description=f"Execute task: {task.description}",
            target=task.type,
            task_id=task.id,
            metadata={
                "task_type": task.type,
                "priority": task.priority.name,
                "context": task.context,
            }
        )

        result = self._safety.validate_action(action)
        if result.passed:
            return True, "Action validated"
        else:
            violations = [v.message for v in result.violations]
            return False, f"Safety violations: {violations}"

    # =========================================================================
    # SENSE Phase - Observe Environment
    # =========================================================================

    def _sense(self) -> List[Opportunity]:
        """Sense phase: scan for opportunities."""
        self._set_phase(LoopPhase.SENSING)

        opportunities = []

        for scanner in self._scanners:
            if not scanner.enabled or not scanner.should_scan():
                continue

            try:
                scanner.last_scan = datetime.now()
                found = scanner.scan()
                opportunities.extend(found)

                self._metrics.opportunities_detected += len(found)
                logger.debug(f"Scanner {scanner.name} found {len(found)} opportunities")

            except Exception as e:
                logger.error(f"Scanner {scanner.name} error: {e}")
                ctx = ErrorContext(operation="scan", component=scanner.name)
                self._errors.handle_error(e, context=ctx)

        # Notify callbacks
        for opp in opportunities:
            for callback in self._on_opportunity_detected:
                try:
                    callback(opp)
                except Exception as e:
                    logger.error(f"Opportunity callback error: {e}")

        # Add to opportunity pool
        with self._lock:
            self._opportunities.extend(opportunities)
            # Remove expired opportunities
            now = datetime.now()
            self._opportunities = [
                o for o in self._opportunities
                if o.expires_at is None or o.expires_at > now
            ]

        return opportunities

    # =========================================================================
    # THINK Phase - Evaluate Options
    # =========================================================================

    def _think(self) -> List[Task]:
        """Think phase: convert opportunities to tasks."""
        self._set_phase(LoopPhase.THINKING)

        tasks = []

        with self._lock:
            # Sort opportunities by value score
            sorted_opps = sorted(
                self._opportunities,
                key=lambda o: (o.priority.value, -o.value_score)
            )

            # Convert top opportunities to tasks
            # Limit to prevent overwhelming the queue
            autonomy_dials = self._config.get_autonomy_dials()
            max_new_tasks = int(5 * autonomy_dials.execution.current_level) + 1

            for opp in sorted_opps[:max_new_tasks]:
                # Check if we already have a task for this opportunity
                existing = any(
                    t.opportunity_id == opp.id
                    for t in self._task_queue + ([self._current_task] if self._current_task else [])
                )
                if existing:
                    continue

                task = Task(
                    id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{opp.id[:8]}",
                    type=opp.type,
                    description=opp.description,
                    priority=opp.priority,
                    opportunity_id=opp.id,
                    estimated_duration=opp.estimated_effort * 3600,  # Convert hours to seconds
                    context=opp.context.copy(),
                )
                tasks.append(task)

                logger.debug(f"Created task {task.id} from opportunity {opp.id}")

        return tasks

    # =========================================================================
    # DECIDE Phase - Select Best Action
    # =========================================================================

    def _decide(self, new_tasks: List[Task]) -> Optional[Task]:
        """Decide phase: select the best task to execute."""
        self._set_phase(LoopPhase.DECIDING)

        # Add new tasks to queue
        with self._lock:
            self._task_queue.extend(new_tasks)

            # Sort queue by priority
            self._task_queue.sort(key=lambda t: (t.priority.value, t.created_at))

            # Check if we have tasks
            if not self._task_queue:
                return None

            # Get autonomy level
            autonomy_dials = self._config.get_autonomy_dials()
            autonomy_level = autonomy_dials.execution.current_level

            # At low autonomy, only execute critical tasks
            if autonomy_level < 0.3:
                critical_tasks = [t for t in self._task_queue if t.priority == TaskPriority.CRITICAL]
                if not critical_tasks:
                    return None
                selected = critical_tasks[0]
            else:
                selected = self._task_queue[0]

            # Validate through safety controller
            valid, reason = self._validate_task_action(selected)
            if not valid:
                logger.warning(f"Task {selected.id} blocked by safety: {reason}")
                selected.status = "cancelled"
                selected.error = reason
                self._task_queue.remove(selected)
                return None

            # Remove from queue
            self._task_queue.remove(selected)
            return selected

    # =========================================================================
    # ACT Phase - Execute Task
    # =========================================================================

    def _act(self, task: Task) -> Tuple[bool, Any]:
        """Act phase: execute the selected task."""
        self._set_phase(LoopPhase.ACTING)

        self._current_task = task
        task.status = "running"
        task.started_at = datetime.now()

        # Sync state
        self._sync_state_to_manager()

        # Create checkpoint before execution
        checkpoint = self._state.create_checkpoint(CheckpointType.AUTO)
        logger.debug(f"Created checkpoint {checkpoint.checkpoint_id} before task {task.id}")

        # Record heartbeat
        self._safety.heartbeat()

        # Find executor
        executor = None
        for ex in self._executors:
            if ex.can_execute(task):
                executor = ex
                break

        if not executor:
            logger.error(f"No executor found for task type: {task.type}")
            return False, f"No executor for task type: {task.type}"

        # Execute with error handling
        try:
            with self._state.transaction(f"task_{task.id}"):
                success, result = executor.execute(task)

                if success:
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.now()

                    # Post-execution audit
                    action = Action(
                        action_type=ActionType.TASK_EXECUTION,
                        description=f"Completed: {task.description}",
                        target=task.type,
                        task_id=task.id,
                    )
                    self._safety.post_execution_audit(action, success=True, result=result)

                    return True, result
                else:
                    raise Exception(f"Task failed: {result}")

        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")

            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()

            # Post-execution audit
            action = Action(
                action_type=ActionType.TASK_EXECUTION,
                description=f"Failed: {task.description}",
                target=task.type,
                task_id=task.id,
            )
            self._safety.post_execution_audit(action, success=False, result=str(e))

            # Handle retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                task.error = None
                with self._lock:
                    self._task_queue.append(task)
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")

            return False, str(e)

        finally:
            self._current_task = None
            self._sync_state_to_manager()

    # =========================================================================
    # LEARN Phase - Record Outcomes
    # =========================================================================

    def _learn(self, task: Task, success: bool, result: Any):
        """Learn phase: record outcomes and update knowledge."""
        self._set_phase(LoopPhase.LEARNING)

        # Update metrics
        with self._lock:
            if success:
                self._metrics.tasks_completed += 1
                self._metrics.consecutive_failures = 0
                self._metrics.last_successful_iteration = datetime.now()
                self._metrics.opportunities_acted_on += 1
            else:
                self._metrics.tasks_failed += 1
                self._metrics.consecutive_failures += 1

            # Store completed task
            self._completed_tasks.append(task)
            # Keep last 100 tasks
            self._completed_tasks = self._completed_tasks[-100:]

            # Remove opportunity if task completed
            if success and task.opportunity_id:
                self._opportunities = [
                    o for o in self._opportunities
                    if o.id != task.opportunity_id
                ]

        # Update learning state
        self._state.update_learning_state(
            tasks_completed_today=self._metrics.tasks_completed,
        )

        # Notify callbacks
        for callback in self._on_task_complete:
            try:
                callback(task, success)
            except Exception as e:
                logger.error(f"Task complete callback error: {e}")

        logger.info(
            f"Task {task.id} {'completed' if success else 'failed'}: "
            f"{self._metrics.tasks_completed} completed, {self._metrics.tasks_failed} failed"
        )

    # =========================================================================
    # Main Loop
    # =========================================================================

    def _run_iteration(self) -> bool:
        """Run a single iteration of the autonomous loop."""
        iteration_start = time.time()

        try:
            # 1. Safety check
            safe, reason = self._check_safety()
            if not safe:
                logger.warning(f"Safety check failed: {reason}")
                self._set_phase(LoopPhase.WAITING)
                return False

            # 2. SENSE - Observe
            opportunities = self._sense()

            # 3. THINK - Evaluate
            new_tasks = self._think()

            # 4. DECIDE - Select
            task = self._decide(new_tasks)

            if task:
                # 5. ACT - Execute
                success, result = self._act(task)

                # 6. LEARN - Record
                self._learn(task, success, result)
            else:
                # No task to execute
                self._set_phase(LoopPhase.WAITING)

            # Update metrics
            iteration_time = time.time() - iteration_start
            self._metrics.iterations += 1
            self._metrics.last_iteration_time = iteration_time
            self._metrics.total_execution_time += iteration_time

            # Sync state
            self._sync_state_to_manager()

            return True

        except Exception as e:
            logger.error(f"Iteration error: {e}")
            self._set_phase(LoopPhase.ERROR_RECOVERY)

            ctx = ErrorContext(
                operation="iteration",
                component="autonomous_loop",
                additional_context={"iteration": self._metrics.iterations}
            )
            self._errors.handle_error(e, context=ctx)

            self._metrics.consecutive_failures += 1
            return False

    def run(self, max_iterations: Optional[int] = None):
        """Run the autonomous loop."""
        logger.info("Starting autonomous loop")

        self._running = True
        iteration = 0

        try:
            while self._running:
                # Check iteration limit
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                # Run iteration
                success = self._run_iteration()
                iteration += 1

                # Wait between iterations
                if self._running:
                    self._set_phase(LoopPhase.WAITING)
                    time.sleep(self._min_interval)

        except KeyboardInterrupt:
            logger.info("Autonomous loop interrupted by user")
        finally:
            self._running = False
            self._set_phase(LoopPhase.SHUTDOWN)
            logger.info(f"Autonomous loop stopped after {iteration} iterations")

    async def run_async(self, max_iterations: Optional[int] = None):
        """Run the autonomous loop asynchronously."""
        logger.info("Starting async autonomous loop")

        self._running = True
        iteration = 0

        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                success = self._run_iteration()
                iteration += 1

                if self._running:
                    self._set_phase(LoopPhase.WAITING)
                    await asyncio.sleep(self._min_interval)

        except asyncio.CancelledError:
            logger.info("Autonomous loop cancelled")
        finally:
            self._running = False
            self._set_phase(LoopPhase.SHUTDOWN)
            logger.info(f"Async autonomous loop stopped after {iteration} iterations")

    def stop(self):
        """Stop the autonomous loop gracefully."""
        logger.info("Stopping autonomous loop")
        self._running = False

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    @property
    def phase(self) -> LoopPhase:
        """Get current loop phase."""
        return self._phase

    @property
    def is_running(self) -> bool:
        """Check if loop is running."""
        return self._running

    @property
    def metrics(self) -> LoopMetrics:
        """Get loop metrics."""
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive loop status."""
        with self._lock:
            return {
                "running": self._running,
                "phase": self._phase.value,
                "metrics": self._metrics.to_dict(),
                "current_task": self._current_task.to_dict() if self._current_task else None,
                "queue_size": len(self._task_queue),
                "opportunities_pending": len(self._opportunities),
                "scanners": [s.name for s in self._scanners],
                "executors": [e.name for e in self._executors],
            }

    def get_queue(self) -> List[Dict[str, Any]]:
        """Get current task queue."""
        with self._lock:
            return [t.to_dict() for t in self._task_queue]

    def get_opportunities(self) -> List[Dict[str, Any]]:
        """Get pending opportunities."""
        with self._lock:
            return [o.to_dict() for o in self._opportunities]

    def get_completed_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently completed tasks."""
        with self._lock:
            return [t.to_dict() for t in self._completed_tasks[-limit:]]


# Global instance
_autonomous_loop: Optional[AutonomousLoop] = None
_loop_lock = threading.Lock()


def get_autonomous_loop() -> AutonomousLoop:
    """Get the global autonomous loop instance."""
    global _autonomous_loop
    with _loop_lock:
        if _autonomous_loop is None:
            _autonomous_loop = AutonomousLoop()
        return _autonomous_loop


def init_autonomous_loop(**kwargs) -> AutonomousLoop:
    """Initialize the global autonomous loop."""
    global _autonomous_loop
    with _loop_lock:
        _autonomous_loop = AutonomousLoop(**kwargs)
        return _autonomous_loop

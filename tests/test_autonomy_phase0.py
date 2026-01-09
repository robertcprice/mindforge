#!/usr/bin/env python3
"""
Integration Tests for PM-1000 Autonomy Phase 0 Components

Tests the four foundation components:
1. ConfigManager - Configuration with autonomy dials
2. ErrorFramework - Error handling with graceful degradation
3. SystemStateManager - Checkpointing and recovery
4. SafetyController - Kill switches and constraints (CRITICAL)

Run with: pytest tests/test_autonomy_phase0.py -v
"""

import pytest
import tempfile
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard" / "backend"))

from autonomy import (
    # Config Manager
    AutonomyConfigManager,
    AutonomyConfig,
    ResourceBudget,
    AutonomyDial,
    AutonomyDialsConfig,
    SafetyConstraint,

    # Error Framework
    ErrorCategory,
    RecoveryAction,
    DegradationLevel,
    PM1000Error,
    NetworkError,
    APIError,
    BudgetExceededError,
    ErrorHandler,
    ErrorContext,
    with_error_handling,

    # System State Manager
    AutonomousLoopState,
    ResourceUsageState,
    SystemStateManager,
    CheckpointType,

    # Safety Controller - THE MOST CRITICAL
    SafetyLevel,
    KillSwitchType,
    ActionType,
    Action,
    SafetyController,
    is_safe_to_operate,
    validate_action,
    emergency_stop,
    heartbeat,
    safety_check,
    SafetyConstraintViolation,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fresh_config_manager(temp_dir):
    """Create a fresh config manager with temp directory."""
    config_path = temp_dir / "autonomy_config.json"
    manager = AutonomyConfigManager(str(config_path))
    return manager


@pytest.fixture
def fresh_state_manager(temp_dir):
    """Create a fresh state manager with temp directory."""
    db_path = temp_dir / "state.db"
    manager = SystemStateManager(str(db_path))
    return manager


@pytest.fixture
def fresh_safety_controller(temp_dir):
    """Create a fresh safety controller with temp directory."""
    kill_file = temp_dir / "EMERGENCY_STOP"
    controller = SafetyController(
        kill_file_path=str(kill_file),
        max_operation_hours=1,
        budget_threshold=0.95,
        max_consecutive_failures=3,
        heartbeat_timeout=60
    )
    return controller


@pytest.fixture
def fresh_error_handler():
    """Create a fresh error handler."""
    return ErrorHandler()


# =============================================================================
# Config Manager Tests
# =============================================================================

class TestConfigManager:
    """Tests for AutonomyConfigManager."""

    def test_default_config_creation(self, fresh_config_manager):
        """Test that default configuration is created correctly."""
        config = fresh_config_manager.config
        assert config is not None
        assert config.environment == "development"

    def test_resource_budget_defaults(self, fresh_config_manager):
        """Test resource budget defaults."""
        budget = fresh_config_manager.get_resource_budget()
        assert budget.daily_api_budget == 50.0
        assert budget.max_concurrent_sessions == 3

    def test_autonomy_dials_exist(self, fresh_config_manager):
        """Test autonomy dials are properly configured."""
        dials = fresh_config_manager.get_autonomy_dials()
        assert dials is not None
        assert hasattr(dials, 'goal_generation')
        assert hasattr(dials, 'execution')
        assert hasattr(dials, 'learning')

    def test_safety_constraints_exist(self, fresh_config_manager):
        """Test safety constraints configuration."""
        constraints = fresh_config_manager.get_safety_constraints()
        assert constraints is not None

    def test_config_persistence(self, temp_dir):
        """Test that config persists across manager instances."""
        config_path = temp_dir / "autonomy_config.json"

        # Create first manager
        manager1 = AutonomyConfigManager(str(config_path))
        config1 = manager1.config

        # Create second manager and verify it loads same config
        manager2 = AutonomyConfigManager(str(config_path))
        config2 = manager2.config

        assert config1.environment == config2.environment

    def test_set_autonomy_dial(self, fresh_config_manager):
        """Test setting an autonomy dial."""
        result = fresh_config_manager.set_autonomy_dial("goal_generation", 0.5)
        assert result is True
        dials = fresh_config_manager.get_autonomy_dials()
        assert dials.goal_generation.current_level == 0.5


# =============================================================================
# Error Framework Tests
# =============================================================================

class TestErrorFramework:
    """Tests for the error handling framework."""

    def test_error_handler_creation(self, fresh_error_handler):
        """Test that error handler can be created."""
        assert fresh_error_handler is not None

    def test_custom_error_types(self):
        """Test custom PM1000 error types."""
        error = BudgetExceededError("Daily budget exceeded")
        assert isinstance(error, PM1000Error)

        error2 = NetworkError("Connection failed")
        assert isinstance(error2, PM1000Error)

        error3 = APIError("API call failed")
        assert isinstance(error3, PM1000Error)

    def test_error_handler_tracking(self, fresh_error_handler):
        """Test that error handler tracks errors."""
        error = NetworkError("Test network error")
        context = ErrorContext(operation="test_op", component="test_component")
        fresh_error_handler.handle_error(error, context=context)

        stats = fresh_error_handler.get_stats()
        assert stats["total_errors"] >= 1

    def test_degradation_level_property(self, fresh_error_handler):
        """Test that degradation level property works."""
        level = fresh_error_handler.degradation_level
        assert level is not None
        assert isinstance(level, DegradationLevel)

    def test_with_error_handling_decorator_success(self):
        """Test the error handling decorator on success."""
        @with_error_handling(operation="test_op", component="test")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"


# =============================================================================
# System State Manager Tests
# =============================================================================

class TestSystemStateManager:
    """Tests for SystemStateManager."""

    def test_state_initialization(self, fresh_state_manager):
        """Test initial state creation."""
        loop_state = fresh_state_manager.loop_state
        assert loop_state is not None
        assert loop_state.phase == "idle"
        assert loop_state.iteration == 0

    def test_state_update(self, fresh_state_manager):
        """Test state updates."""
        fresh_state_manager.update_loop_state(phase="thinking", iteration=1)
        loop_state = fresh_state_manager.loop_state

        assert loop_state.phase == "thinking"
        assert loop_state.iteration == 1

    def test_checkpoint_creation(self, fresh_state_manager):
        """Test checkpoint creation."""
        fresh_state_manager.update_loop_state(phase="acting", iteration=5)

        checkpoint = fresh_state_manager.create_checkpoint(CheckpointType.MANUAL)
        assert checkpoint is not None
        assert checkpoint.checkpoint_id is not None

    def test_checkpoint_restore(self, fresh_state_manager):
        """Test checkpoint restoration."""
        # Set initial state
        fresh_state_manager.update_loop_state(phase="initial", iteration=10)
        checkpoint = fresh_state_manager.create_checkpoint(CheckpointType.MANUAL)

        # Modify state
        fresh_state_manager.update_loop_state(phase="modified", iteration=99)

        # Restore checkpoint
        success = fresh_state_manager.restore_checkpoint(checkpoint.checkpoint_id)
        assert success

        loop_state = fresh_state_manager.loop_state
        assert loop_state.phase == "initial"
        assert loop_state.iteration == 10

    def test_transaction_support(self, fresh_state_manager):
        """Test transaction support."""
        # Successful transaction
        with fresh_state_manager.transaction("test_success"):
            fresh_state_manager.update_loop_state(phase="in_transaction", iteration=50)

        loop_state = fresh_state_manager.loop_state
        assert loop_state.phase == "in_transaction"

    def test_transaction_rollback(self, fresh_state_manager):
        """Test transaction rollback on failure."""
        fresh_state_manager.update_loop_state(phase="before", iteration=1)

        try:
            with fresh_state_manager.transaction("test_rollback"):
                fresh_state_manager.update_loop_state(phase="during", iteration=100)
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # State should be rolled back
        loop_state = fresh_state_manager.loop_state
        assert loop_state.phase == "before"

    def test_resource_usage_tracking(self, fresh_state_manager):
        """Test resource usage tracking."""
        fresh_state_manager.increment_resource_usage(api_spend=5.0, api_calls=1)
        fresh_state_manager.increment_resource_usage(api_spend=3.0, api_calls=1)

        resource_state = fresh_state_manager.resource_state
        assert resource_state.api_spend_today == 8.0
        assert resource_state.api_calls_today == 2


# =============================================================================
# Safety Controller Tests (THE MOST CRITICAL)
# =============================================================================

class TestSafetyController:
    """Tests for SafetyController - THE MOST CRITICAL COMPONENT."""

    def test_initialization(self, fresh_safety_controller):
        """Test safety controller initialization."""
        status = fresh_safety_controller.get_status()

        assert status["safe_to_operate"] is True
        assert status["safety_level"] == "NORMAL"

    def test_immutable_constraints_cannot_be_disabled(self, fresh_safety_controller):
        """Test that immutable constraints cannot be disabled."""
        result = fresh_safety_controller.disable_constraint("no_data_destruction")
        assert result is False

        constraints = fresh_safety_controller.get_constraints()
        assert constraints["no_data_destruction"]["enabled"] is True

    def test_file_deletion_protection(self, fresh_safety_controller):
        """Test that database files are protected from deletion."""
        action = Action(
            action_type=ActionType.FILE_DELETE,
            description="Delete database",
            target="/path/to/database.db"
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is False
        assert any("no_data_destruction" in v.constraint_name for v in result.violations)

    def test_secret_detection(self, fresh_safety_controller):
        """Test that secrets are detected in code."""
        action = Action(
            action_type=ActionType.FILE_MODIFY,
            description="Modify file with API key",
            target="/path/to/file.py",
            metadata={"content": "api_key = 'sk-abcdefghijklmnopqrstuvwxyz123456789012345678'"}
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is False
        assert any("no_secrets_in_code" in v.constraint_name for v in result.violations)

    def test_budget_kill_switch(self, fresh_safety_controller):
        """Test budget-based kill switch triggers when threshold exceeded."""
        # Set budget near threshold (95% of limit triggers kill switch)
        fresh_safety_controller.update_budget_status(48.0, 50.0)  # 96% usage

        # The kill switch should trigger
        safe, reason = fresh_safety_controller.is_safe_to_operate()
        assert safe is False

    def test_file_based_kill_switch(self, temp_dir):
        """Test file-based kill switch."""
        kill_file = temp_dir / "TEST_EMERGENCY_STOP"
        controller = SafetyController(kill_file_path=str(kill_file))

        # Initially safe
        safe, _ = controller.is_safe_to_operate()
        assert safe is True

        # Create kill file
        kill_file.write_text("Test emergency stop")

        # Should trigger kill switch
        safe, reason = controller.is_safe_to_operate()
        assert safe is False

    def test_emergency_stop(self, fresh_safety_controller):
        """Test emergency stop procedure."""
        fresh_safety_controller.emergency_stop("Test emergency")

        assert fresh_safety_controller.safety_level == SafetyLevel.EMERGENCY_STOP
        safe, _ = fresh_safety_controller.is_safe_to_operate()
        assert safe is False

    def test_heartbeat_keeps_alive(self, temp_dir):
        """Test that heartbeat prevents timeout kill switch."""
        controller = SafetyController(
            kill_file_path=str(temp_dir / "STOP"),
            heartbeat_timeout=60
        )

        controller.heartbeat()
        safe, _ = controller.is_safe_to_operate()
        assert safe is True

    def test_failure_tracking(self, fresh_safety_controller):
        """Test consecutive failure tracking."""
        action = Action(
            action_type=ActionType.TASK_EXECUTION,
            description="Test task",
            target="test"
        )

        # Record failures up to the max
        for _ in range(3):
            fresh_safety_controller.post_execution_audit(action, success=False)

        # Should trigger failure-based kill switch
        safe, _ = fresh_safety_controller.is_safe_to_operate()
        assert safe is False

    def test_production_deploy_protection(self, fresh_safety_controller):
        """Test that production deployments are blocked."""
        action = Action(
            action_type=ActionType.GIT_PUSH,
            description="Push to production",
            target="origin/main"
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is False

    def test_read_only_mode(self, fresh_safety_controller):
        """Test read-only mode blocks write operations."""
        fresh_safety_controller.set_safety_level(SafetyLevel.READ_ONLY)

        action = Action(
            action_type=ActionType.FILE_MODIFY,
            description="Modify file",
            target="/path/to/file.txt"
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is False
        assert result.safety_level == SafetyLevel.READ_ONLY

    def test_audit_logging(self, fresh_safety_controller):
        """Test that actions are audited."""
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="Create file",
            target="/path/to/safe_file.txt"
        )

        fresh_safety_controller.validate_action(action)

        log = fresh_safety_controller.get_audit_log()
        assert len(log) >= 1

    def test_kill_switch_reset(self, fresh_safety_controller):
        """Test kill switch reset capability."""
        fresh_safety_controller.emergency_stop("Test")
        safe, _ = fresh_safety_controller.is_safe_to_operate()
        assert safe is False

        fresh_safety_controller.reset_all_kill_switches()
        safe, _ = fresh_safety_controller.is_safe_to_operate()
        assert safe is True

    def test_safe_file_operations_allowed(self, fresh_safety_controller, temp_dir):
        """Test that safe file operations are allowed."""
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="Create safe file",
            target=str(temp_dir / "output.txt")
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is True

    def test_env_file_protected(self, fresh_safety_controller):
        """Test that .env files are protected from deletion."""
        action = Action(
            action_type=ActionType.FILE_DELETE,
            description="Delete env file",
            target="/project/.env"
        )

        result = fresh_safety_controller.validate_action(action)
        assert result.passed is False


# =============================================================================
# Integration Tests - Components Working Together
# =============================================================================

class TestPhase0Integration:
    """Integration tests for all Phase 0 components working together."""

    def test_full_autonomous_cycle(self, temp_dir):
        """Test a complete autonomous operation cycle."""
        # Initialize all components
        config_manager = AutonomyConfigManager(str(temp_dir / "config.json"))
        state_manager = SystemStateManager(str(temp_dir / "state.db"))
        safety_controller = SafetyController(kill_file_path=str(temp_dir / "STOP"))
        error_handler = ErrorHandler()

        # 1. Check if safe to start
        safe, reason = safety_controller.is_safe_to_operate()
        assert safe, f"Not safe to start: {reason}"

        # 2. Get config
        config = config_manager.config
        assert config.environment == "development"

        # 3. Begin autonomous loop
        state_manager.update_loop_state(phase="sensing", iteration=1)

        # 4. Create checkpoint before risky operation
        checkpoint = state_manager.create_checkpoint(CheckpointType.MANUAL)

        # 5. Validate action before execution
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="Create output file",
            target=str(temp_dir / "output.txt")
        )
        result = safety_controller.validate_action(action)
        assert result.passed

        # 6. Execute the operation
        (temp_dir / "output.txt").write_text("Hello, autonomous world!")

        # 7. Record heartbeat
        safety_controller.heartbeat()

        # 8. Update state
        state_manager.update_loop_state(phase="completed", iteration=1)
        state_manager.update_loop_state(total_tasks_completed=1)

        # 9. Verify final state
        loop_state = state_manager.loop_state
        assert loop_state.phase == "completed"
        assert loop_state.total_tasks_completed >= 1

    def test_error_recovery_with_state(self, temp_dir):
        """Test error recovery using state checkpoints."""
        state_manager = SystemStateManager(str(temp_dir / "state.db"))
        error_handler = ErrorHandler()

        # Set up initial state
        state_manager.update_loop_state(phase="working", iteration=10)
        checkpoint = state_manager.create_checkpoint(CheckpointType.MANUAL)

        # Simulate error during operation
        try:
            with state_manager.transaction("risky_operation"):
                state_manager.update_loop_state(phase="risky", iteration=99)
                raise NetworkError("Connection lost")
        except NetworkError as e:
            context = ErrorContext(operation="risky_op", component="test")
            error_handler.handle_error(e, context=context)

        # State should be restored to checkpoint
        loop_state = state_manager.loop_state
        assert loop_state.phase == "working"
        assert loop_state.iteration == 10

    def test_budget_constraint_across_components(self, temp_dir):
        """Test budget tracking across state and safety components."""
        state_manager = SystemStateManager(str(temp_dir / "state.db"))
        safety_controller = SafetyController(
            kill_file_path=str(temp_dir / "STOP"),
            budget_threshold=0.90  # Lower threshold for this test
        )

        # Record spending
        state_manager.increment_resource_usage(api_spend=45.0, api_calls=10)

        # Sync with safety controller
        resource_state = state_manager.resource_state
        safety_controller.update_budget_status(
            resource_state.api_spend_today,
            50.0  # Daily limit
        )

        # Should be unsafe (90% threshold exceeded)
        safe, _ = safety_controller.is_safe_to_operate()
        assert safe is False

    def test_multi_threaded_safety(self, temp_dir):
        """Test thread safety of components."""
        safety_controller = SafetyController(kill_file_path=str(temp_dir / "STOP"))
        state_manager = SystemStateManager(str(temp_dir / "state.db"))

        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    safety_controller.heartbeat()
                    state_manager.update_loop_state(phase=f"thread_{thread_id}", iteration=i)
                    action = Action(
                        action_type=ActionType.API_CALL,
                        description=f"Thread {thread_id} call {i}",
                        target="api"
                    )
                    safety_controller.validate_action(action)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# =============================================================================
# Stress Tests
# =============================================================================

class TestPhase0StressTests:
    """Stress tests for Phase 0 components."""

    def test_rapid_state_updates(self, fresh_state_manager):
        """Test rapid state updates don't cause issues."""
        for i in range(100):
            fresh_state_manager.update_loop_state(
                phase=f"iteration_{i % 5}",
                iteration=i
            )

        loop_state = fresh_state_manager.loop_state
        assert loop_state.iteration == 99

    def test_many_checkpoints(self, fresh_state_manager):
        """Test creating many checkpoints."""
        checkpoint_ids = []
        for i in range(50):
            fresh_state_manager.update_loop_state(phase=f"phase_{i}", iteration=i)
            checkpoint = fresh_state_manager.create_checkpoint(CheckpointType.AUTO)
            checkpoint_ids.append(checkpoint.checkpoint_id)

        # Should be able to restore any checkpoint
        success = fresh_state_manager.restore_checkpoint(checkpoint_ids[25])
        assert success

        loop_state = fresh_state_manager.loop_state
        assert loop_state.iteration == 25

    def test_many_safety_validations(self, fresh_safety_controller):
        """Test many safety validations don't degrade performance."""
        start = time.time()
        for i in range(500):
            action = Action(
                action_type=ActionType.FILE_CREATE,
                description=f"Create file {i}",
                target=f"/tmp/test_{i}.txt"
            )
            fresh_safety_controller.validate_action(action)

        elapsed = time.time() - start
        assert elapsed < 5.0, f"Safety validations too slow: {elapsed:.2f}s"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_action_target(self, fresh_safety_controller):
        """Test handling of empty action target."""
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="Create file",
            target=""
        )
        result = fresh_safety_controller.validate_action(action)
        assert result is not None

    def test_very_long_description(self, fresh_safety_controller):
        """Test handling of very long description."""
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="A" * 10000,
            target="/tmp/test.txt"
        )
        result = fresh_safety_controller.validate_action(action)
        assert result is not None

    def test_unicode_in_action(self, fresh_safety_controller):
        """Test handling of unicode in actions."""
        action = Action(
            action_type=ActionType.FILE_CREATE,
            description="Create file with émojis 🎉",
            target="/tmp/tëst_fïlé.txt"
        )
        result = fresh_safety_controller.validate_action(action)
        assert result.passed is True

    def test_state_manager_restart_recovery(self, temp_dir):
        """Test state manager can recover after restart."""
        db_path = temp_dir / "state.db"

        # Create first manager and set state
        manager1 = SystemStateManager(str(db_path))
        manager1.update_loop_state(phase="running", iteration=42)
        manager1.increment_resource_usage(api_spend=10.5, api_calls=5)

        # Simulate restart by creating new manager
        manager2 = SystemStateManager(str(db_path))
        loop_state = manager2.loop_state

        # State should be preserved
        assert loop_state.phase == "running"
        assert loop_state.iteration == 42

    def test_safety_controller_concurrent_emergency_stop(self, temp_dir):
        """Test concurrent emergency stop calls."""
        controller = SafetyController(kill_file_path=str(temp_dir / "STOP"))
        errors = []

        def stop_worker():
            try:
                controller.emergency_stop("Concurrent stop")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stop_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert controller.safety_level == SafetyLevel.EMERGENCY_STOP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

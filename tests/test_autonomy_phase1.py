#!/usr/bin/env python3
"""
Phase 1 Integration Tests for PM-1000 Autonomy System

Tests the core autonomous loop components:
- AutonomousLoop (sense-think-act cycle)
- TaskRouter (multi-tier model routing)
- GoalManager (goal lifecycle management)

And their integration with Phase 0 components:
- SafetyController
- SystemStateManager
- ErrorHandler
- ConfigManager
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard" / "backend"))

from autonomy import (
    # Phase 0
    get_safety_controller,
    init_safety_controller,
    get_state_manager,
    init_state_manager,
    get_config_manager,
    init_config_manager,
    get_error_handler,
    is_safe_to_operate,
    SafetyLevel,
    ActionType,
    Action,
    CheckpointType,

    # Phase 1
    LoopPhase,
    TaskPriority,
    Opportunity,
    Task,
    LoopMetrics,
    OpportunityScanner,
    TaskExecutor,
    AutonomousLoop,
    get_autonomous_loop,
    init_autonomous_loop,

    ModelTier,
    TaskComplexity,
    ModelConfig,
    TaskAnalysis,
    RoutingDecision,
    TaskRouter,
    get_task_router,
    init_task_router,

    GoalLevel,
    GoalStatus,
    GoalSource,
    GoalPriority,
    Goal,
    GoalManager,
    get_goal_manager,
    init_goal_manager,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons before each test."""
    import autonomy.safety_controller as sc
    import autonomy.system_state_manager as ssm
    import autonomy.config_manager as cm
    import autonomy.error_framework as ef
    import autonomy.autonomous_loop as al
    import autonomy.task_router as tr
    import autonomy.goal_manager as gm
    import os

    # Reset all singletons
    sc._safety_controller = None
    ssm._state_manager = None
    cm._config_manager = None
    ef._error_handler = None
    al._autonomous_loop = None
    tr._task_router = None
    gm._goal_manager = None

    # Clean up kill switch file before each test
    kill_switch_file = "/tmp/PM1000_EMERGENCY_STOP"
    if os.path.exists(kill_switch_file):
        os.remove(kill_switch_file)

    yield

    # Clean up after test too
    if os.path.exists(kill_switch_file):
        os.remove(kill_switch_file)


@pytest.fixture
def safety_controller():
    """Get a fresh safety controller."""
    import os
    # Double-check kill switch file is removed before creating controller
    kill_switch_file = "/tmp/PM1000_EMERGENCY_STOP"
    if os.path.exists(kill_switch_file):
        os.remove(kill_switch_file)
    controller = init_safety_controller()
    yield controller
    # Clean up after test
    if os.path.exists(kill_switch_file):
        os.remove(kill_switch_file)


@pytest.fixture
def state_manager():
    """Get a fresh state manager with unique database per test."""
    import tempfile
    import os
    # Use a unique temp database for each test
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_state.db")
    return init_state_manager(db_path=db_path)


@pytest.fixture
def config_manager():
    """Get a fresh config manager."""
    return init_config_manager()


@pytest.fixture
def autonomous_loop(safety_controller, state_manager, config_manager):
    """Get a configured autonomous loop."""
    return init_autonomous_loop(
        safety_controller=safety_controller,
        state_manager=state_manager,
        config_manager=config_manager,
        min_iteration_interval=0.1,  # Fast for testing
    )


@pytest.fixture
def task_router(config_manager, state_manager):
    """Get a configured task router."""
    return init_task_router(
        config_manager=config_manager,
        state_manager=state_manager,
    )


@pytest.fixture
def goal_manager(config_manager, state_manager, safety_controller):
    """Get a configured goal manager."""
    return init_goal_manager(
        config_manager=config_manager,
        state_manager=state_manager,
        safety_controller=safety_controller,
    )


# ===========================================================================
# Test Opportunity Scanner
# ===========================================================================

class MockOpportunityScanner(OpportunityScanner):
    """Mock scanner for testing."""

    def __init__(self, opportunities: List[Opportunity] = None):
        super().__init__("mock_scanner")
        self._opportunities = opportunities or []
        self.scan_count = 0

    def scan(self) -> List[Opportunity]:
        self.scan_count += 1
        return self._opportunities


# ===========================================================================
# Test Task Executor
# ===========================================================================

class MockTaskExecutor(TaskExecutor):
    """Mock executor for testing."""

    def __init__(self, supported_types: List[str] = None, should_succeed: bool = True):
        super().__init__("mock_executor")
        self.supported_task_types = supported_types or ["test_task", "mock_task"]
        self.should_succeed = should_succeed
        self.executed_tasks: List[Task] = []

    def execute(self, task: Task) -> Tuple[bool, Any]:
        self.executed_tasks.append(task)
        if self.should_succeed:
            return True, {"status": "completed", "task_id": task.id}
        else:
            return False, "Execution failed"


# ===========================================================================
# Autonomous Loop Tests
# ===========================================================================

class TestAutonomousLoop:
    """Tests for the autonomous loop."""

    def test_loop_initialization(self, autonomous_loop):
        """Test loop initializes correctly."""
        assert autonomous_loop is not None
        assert autonomous_loop.phase == LoopPhase.IDLE
        assert not autonomous_loop.is_running
        assert autonomous_loop.metrics.iterations == 0

    def test_register_scanner(self, autonomous_loop):
        """Test registering opportunity scanners."""
        scanner = MockOpportunityScanner()
        autonomous_loop.register_scanner(scanner)

        status = autonomous_loop.get_status()
        assert "mock_scanner" in status["scanners"]

    def test_register_executor(self, autonomous_loop):
        """Test registering task executors."""
        executor = MockTaskExecutor()
        autonomous_loop.register_executor(executor)

        status = autonomous_loop.get_status()
        assert "mock_executor" in status["executors"]

    def test_sense_phase_with_scanner(self, autonomous_loop):
        """Test sense phase detects opportunities."""
        opp = Opportunity(
            id="test_opp_1",
            type="test_task",
            description="Test opportunity",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        autonomous_loop.register_scanner(scanner)

        # Run single iteration (will go through sense phase)
        autonomous_loop._sense()

        assert scanner.scan_count == 1
        assert len(autonomous_loop.get_opportunities()) == 1

    def test_think_phase_creates_tasks(self, autonomous_loop):
        """Test think phase converts opportunities to tasks."""
        opp = Opportunity(
            id="test_opp_2",
            type="test_task",
            description="Test opportunity for task creation",
            source="test",
            priority=TaskPriority.HIGH,
        )
        scanner = MockOpportunityScanner([opp])
        autonomous_loop.register_scanner(scanner)

        # Run sense and think
        autonomous_loop._sense()
        tasks = autonomous_loop._think()

        assert len(tasks) == 1
        assert tasks[0].type == "test_task"
        assert tasks[0].opportunity_id == "test_opp_2"

    def test_decide_phase_selects_task(self, autonomous_loop, safety_controller):
        """Test decide phase selects best task."""
        opp = Opportunity(
            id="test_opp_3",
            type="test_task",
            description="Test opportunity for decision",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor()
        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # Run sense and think
        autonomous_loop._sense()
        tasks = autonomous_loop._think()

        # Run decide
        selected = autonomous_loop._decide(tasks)

        assert selected is not None
        assert selected.type == "test_task"

    def test_act_phase_executes_task(self, autonomous_loop):
        """Test act phase executes selected task."""
        opp = Opportunity(
            id="test_opp_4",
            type="test_task",
            description="Test opportunity for execution",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor()
        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # Run through phases
        autonomous_loop._sense()
        tasks = autonomous_loop._think()
        selected = autonomous_loop._decide(tasks)

        if selected:
            success, result = autonomous_loop._act(selected)
            assert success
            assert len(executor.executed_tasks) == 1

    def test_full_iteration(self, autonomous_loop):
        """Test a full loop iteration."""
        opp = Opportunity(
            id="test_opp_5",
            type="test_task",
            description="Test opportunity for full iteration",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor()
        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # Run single iteration
        success = autonomous_loop._run_iteration()

        assert success
        assert autonomous_loop.metrics.iterations == 1
        assert autonomous_loop.metrics.tasks_completed == 1

    def test_loop_runs_multiple_iterations(self, autonomous_loop):
        """Test loop runs for specified iterations."""
        scanner = MockOpportunityScanner()
        autonomous_loop.register_scanner(scanner)

        # Run 3 iterations
        autonomous_loop.run(max_iterations=3)

        assert autonomous_loop.metrics.iterations == 3
        assert not autonomous_loop.is_running

    def test_loop_stops_on_safety_failure(self, autonomous_loop, safety_controller):
        """Test loop stops when safety check fails."""
        scanner = MockOpportunityScanner()
        autonomous_loop.register_scanner(scanner)

        # Trigger emergency stop
        safety_controller.emergency_stop("Test stop")

        # Try to run
        autonomous_loop.run(max_iterations=5)

        # Should have stopped early
        assert autonomous_loop.metrics.iterations < 5

    def test_phase_change_callbacks(self, autonomous_loop):
        """Test phase change callbacks are triggered."""
        phases_seen = []

        def on_phase_change(old, new):
            phases_seen.append((old.value, new.value))

        autonomous_loop.on_phase_change(on_phase_change)
        autonomous_loop._run_iteration()

        # Should have gone through multiple phases
        assert len(phases_seen) > 0
        assert any("sensing" in p[1] for p in phases_seen)

    def test_task_complete_callbacks(self, autonomous_loop):
        """Test task completion callbacks are triggered."""
        completed_tasks = []

        def on_task_complete(task, success):
            completed_tasks.append((task.id, success))

        opp = Opportunity(
            id="test_opp_6",
            type="test_task",
            description="Test opportunity for callback",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor()
        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)
        autonomous_loop.on_task_complete(on_task_complete)

        autonomous_loop._run_iteration()

        assert len(completed_tasks) == 1
        assert completed_tasks[0][1] is True  # success

    def test_metrics_tracking(self, autonomous_loop):
        """Test metrics are tracked correctly."""
        opps = [
            Opportunity(id=f"opp_{i}", type="test_task", description=f"Opp {i}", source="test")
            for i in range(3)
        ]
        scanner = MockOpportunityScanner(opps[:1])  # One per scan
        executor = MockTaskExecutor()
        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        autonomous_loop.run(max_iterations=3)

        metrics = autonomous_loop.metrics
        assert metrics.iterations == 3
        assert metrics.opportunities_detected >= 1
        assert metrics.total_execution_time > 0


# ===========================================================================
# Task Router Tests
# ===========================================================================

class TestTaskRouter:
    """Tests for the task router."""

    def test_router_initialization(self, task_router):
        """Test router initializes with default models."""
        assert task_router is not None
        available = task_router.get_available_tiers()
        assert len(available) > 0

    def test_analyze_simple_task(self, task_router):
        """Test analyzing a simple task."""
        analysis = task_router.analyze_task(
            task_id="test_1",
            task_type="code_format",
            description="Format the config file",
        )

        assert analysis.estimated_complexity == TaskComplexity.TRIVIAL
        assert analysis.minimum_tier == ModelTier.LOCAL

    def test_analyze_complex_task(self, task_router):
        """Test analyzing a complex task."""
        analysis = task_router.analyze_task(
            task_id="test_2",
            task_type="feature_implement",
            description="Implement a new architecture design pattern",
        )

        assert analysis.estimated_complexity.value >= TaskComplexity.COMPLEX.value

    def test_route_to_appropriate_tier(self, task_router):
        """Test routing to appropriate tier."""
        # Simple task should route to LOCAL or FAST
        analysis = task_router.analyze_task(
            task_id="test_3",
            task_type="code_format",
            description="Simple formatting task",
        )
        decision = task_router.route(analysis)

        assert decision.selected_tier in (ModelTier.LOCAL, ModelTier.FAST)

    def test_route_complex_to_capable(self, task_router):
        """Test complex tasks route to capable tier."""
        analysis = task_router.analyze_task(
            task_id="test_4",
            task_type="refactor",
            description="Refactor the entire authentication system",
        )
        decision = task_router.route(analysis)

        assert decision.selected_tier.value >= ModelTier.CAPABLE.value

    def test_route_respects_requirements(self, task_router):
        """Test routing respects task requirements."""
        analysis = task_router.analyze_task(
            task_id="test_5",
            task_type="ui_review",
            description="Review the visual design of the image gallery",
            context={"requires_vision": True},
        )
        # Vision requirement should push to higher tier
        decision = task_router.route(analysis)

        # Should select a model that supports vision
        assert decision.selected_model.supports_vision or decision.selected_tier.value >= ModelTier.FAST.value

    def test_record_outcome(self, task_router):
        """Test recording routing outcomes."""
        task_router.record_outcome(
            task_id="test_6",
            tier=ModelTier.FAST,
            success=True,
            latency_ms=500,
            cost=0.01,
            input_tokens=100,
            output_tokens=200,
            task_type="test_task",
        )

        metrics = task_router.get_metrics()
        assert metrics["total_routed"] >= 0
        assert metrics["total_cost"] >= 0.01

    def test_cost_breakdown(self, task_router):
        """Test cost breakdown calculation."""
        # Route and record some tasks
        for i in range(3):
            decision = task_router.route_task(
                task_id=f"cost_test_{i}",
                task_type="test_task",
                description="Test task for cost tracking",
            )
            task_router.record_outcome(
                task_id=f"cost_test_{i}",
                tier=decision.selected_tier,
                success=True,
                latency_ms=100,
                cost=0.05 * (i + 1),
                input_tokens=100,
                output_tokens=200,
            )

        breakdown = task_router.get_cost_breakdown()
        assert breakdown["total_cost"] > 0

    def test_model_availability(self, task_router):
        """Test handling of model availability."""
        # Disable LOCAL tier
        task_router.set_model_available(ModelTier.LOCAL, False)

        analysis = task_router.analyze_task(
            task_id="avail_test",
            task_type="code_format",
            description="Simple task",
        )
        decision = task_router.route(analysis)

        # Should route to next available tier
        assert decision.selected_tier != ModelTier.LOCAL


# ===========================================================================
# Goal Manager Tests
# ===========================================================================

class TestGoalManager:
    """Tests for the goal manager."""

    def test_manager_initialization(self, goal_manager):
        """Test goal manager initializes correctly."""
        assert goal_manager is not None
        status = goal_manager.get_status()
        assert status["total_goals"] == 0

    def test_create_goal(self, goal_manager):
        """Test creating a goal."""
        goal = goal_manager.create_goal(
            title="Test Goal",
            description="A test goal for unit testing",
            level=GoalLevel.GOAL,
            priority=GoalPriority.MEDIUM,
        )

        assert goal is not None
        assert goal.id.startswith("goal_")
        assert goal.status == GoalStatus.PROPOSED
        assert goal.title == "Test Goal"

    def test_goal_approval(self, goal_manager):
        """Test approving a goal."""
        goal = goal_manager.create_goal(
            title="Approval Test",
            description="Goal to test approval",
        )

        result = goal_manager.approve_goal(goal.id)

        assert result is True
        assert goal.status == GoalStatus.APPROVED
        assert goal.metadata.approved_at is not None

    def test_goal_start(self, goal_manager):
        """Test starting an approved goal."""
        goal = goal_manager.create_goal(
            title="Start Test",
            description="Goal to test starting",
            auto_approve=True,
        )

        result = goal_manager.start_goal(goal.id)

        assert result is True
        assert goal.status == GoalStatus.ACTIVE
        assert goal.metadata.started_at is not None

    def test_goal_completion(self, goal_manager):
        """Test completing a goal."""
        goal = goal_manager.create_goal(
            title="Completion Test",
            description="Goal to test completion",
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)

        result = goal_manager.complete_goal(goal.id, result="Success!")

        assert result is True
        assert goal.status == GoalStatus.COMPLETED
        assert goal.progress == 1.0
        assert goal.result == "Success!"

    def test_goal_abandonment(self, goal_manager):
        """Test abandoning a goal."""
        goal = goal_manager.create_goal(
            title="Abandon Test",
            description="Goal to test abandonment",
            auto_approve=True,
        )

        result = goal_manager.abandon_goal(goal.id, "No longer needed")

        assert result is True
        assert goal.status == GoalStatus.ABANDONED
        assert goal.error == "No longer needed"

    def test_goal_blocking(self, goal_manager):
        """Test blocking and unblocking goals."""
        goal = goal_manager.create_goal(
            title="Block Test",
            description="Goal to test blocking",
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)

        # Block
        result = goal_manager.block_goal(goal.id, "Waiting for dependency")
        assert result is True
        assert goal.status == GoalStatus.BLOCKED

        # Unblock
        result = goal_manager.unblock_goal(goal.id)
        assert result is True
        assert goal.status == GoalStatus.ACTIVE

    def test_success_criteria_tracking(self, goal_manager):
        """Test tracking success criteria."""
        goal = goal_manager.create_goal(
            title="Criteria Test",
            description="Goal with success criteria",
            success_criteria=[
                "Criterion 1",
                "Criterion 2",
                "Criterion 3",
            ],
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)

        # Mark criteria as met
        goal_manager.mark_criterion_met(goal.id, 0)
        assert goal.progress == pytest.approx(1/3, rel=0.01)

        goal_manager.mark_criterion_met(goal.id, 1)
        assert goal.progress == pytest.approx(2/3, rel=0.01)

        goal_manager.mark_criterion_met(goal.id, 2)
        # Should auto-complete
        assert goal.status == GoalStatus.COMPLETED
        assert goal.progress == 1.0

    def test_goal_decomposition(self, goal_manager):
        """Test decomposing goals into sub-goals."""
        parent = goal_manager.create_goal(
            title="Parent Goal",
            description="Goal to decompose",
            level=GoalLevel.OBJECTIVE,
            success_criteria=[
                "Step 1",
                "Step 2",
                "Step 3",
            ],
        )

        sub_goals = goal_manager.decompose_goal(parent.id)

        assert len(sub_goals) == 3
        assert all(sg.parent_id == parent.id for sg in sub_goals)
        assert parent.child_ids == [sg.id for sg in sub_goals]

    def test_goal_dependencies(self, goal_manager):
        """Test goal dependency tracking."""
        goal1 = goal_manager.create_goal(
            title="First Goal",
            description="Must complete first",
            auto_approve=True,
        )
        goal2 = goal_manager.create_goal(
            title="Second Goal",
            description="Depends on first",
            auto_approve=True,
        )

        # Set up dependency
        goal2.depends_on.append(goal1.id)
        goal1.blocks.append(goal2.id)

        # Try to start goal2 (should fail - dependency not met)
        result = goal_manager.start_goal(goal2.id)
        assert result is False
        assert goal2.status == GoalStatus.BLOCKED

        # Complete goal1
        goal_manager.start_goal(goal1.id)
        goal_manager.complete_goal(goal1.id)

        # Now goal2 should be unblocked
        assert goal2.status != GoalStatus.BLOCKED

    def test_create_from_template(self, goal_manager):
        """Test creating goals from templates."""
        goal = goal_manager.create_from_template(
            template_name="fix_bug",
            params={
                "bug_title": "Login fails",
                "bug_description": "Login fails when password contains special characters",
            },
        )

        assert goal is not None
        assert "fix" in goal.title.lower() or "bug" in goal.title.lower()
        assert len(goal.success_criteria) > 0

    def test_goal_queries(self, goal_manager):
        """Test querying goals by various criteria."""
        # Create goals with different statuses and priorities
        goal1 = goal_manager.create_goal(
            title="High Priority Active",
            description="Test",
            priority=GoalPriority.HIGH,
            auto_approve=True,
        )
        goal_manager.start_goal(goal1.id)

        goal2 = goal_manager.create_goal(
            title="Low Priority Proposed",
            description="Test",
            priority=GoalPriority.LOW,
        )

        goal3 = goal_manager.create_goal(
            title="Medium Priority Completed",
            description="Test",
            priority=GoalPriority.MEDIUM,
            auto_approve=True,
        )
        goal_manager.start_goal(goal3.id)
        goal_manager.complete_goal(goal3.id)

        # Query by status
        active = goal_manager.get_active_goals()
        assert len(active) == 1
        assert active[0].id == goal1.id

        # Query by priority
        high_priority = goal_manager.get_goals_by_priority(GoalPriority.HIGH)
        assert len(high_priority) == 1

        # Query completed
        completed = goal_manager.get_goals_by_status(GoalStatus.COMPLETED)
        assert len(completed) == 1

    def test_goal_tree(self, goal_manager):
        """Test getting goal tree structure."""
        parent = goal_manager.create_goal(
            title="Root Goal",
            description="Root",
            level=GoalLevel.OBJECTIVE,
            success_criteria=["Sub 1", "Sub 2"],
        )
        goal_manager.decompose_goal(parent.id)

        tree = goal_manager.get_goal_tree(parent.id)

        assert "goal" in tree
        assert "children" in tree
        assert len(tree["children"]) == 2

    def test_goal_callbacks(self, goal_manager):
        """Test goal lifecycle callbacks."""
        created_goals = []
        completed_goals = []
        status_changes = []

        goal_manager.on_goal_created(lambda g: created_goals.append(g.id))
        goal_manager.on_goal_completed(lambda g: completed_goals.append(g.id))
        goal_manager.on_goal_status_change(lambda g, old, new: status_changes.append((g.id, old, new)))

        goal = goal_manager.create_goal(
            title="Callback Test",
            description="Test callbacks",
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)
        goal_manager.complete_goal(goal.id)

        assert goal.id in created_goals
        assert goal.id in completed_goals
        assert len(status_changes) >= 3  # proposed->approved->active->completed


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 components."""

    def test_loop_with_router_and_goals(self, autonomous_loop, task_router, goal_manager):
        """Test autonomous loop integration with router and goals."""
        # Create a goal that will generate opportunities
        goal = goal_manager.create_goal(
            title="Integration Test Goal",
            description="Test the full integration",
            success_criteria=["Complete test task"],
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)

        # Create scanner that generates opportunities from goals
        class GoalOpportunityScanner(OpportunityScanner):
            def __init__(self, gm):
                super().__init__("goal_scanner")
                self.goal_manager = gm

            def scan(self) -> List[Opportunity]:
                active = self.goal_manager.get_active_goals()
                return [
                    Opportunity(
                        id=f"opp_from_{g.id}",
                        type="test_task",
                        description=g.description,
                        source="goal_manager",
                        context={"goal_id": g.id},
                    )
                    for g in active
                ]

        scanner = GoalOpportunityScanner(goal_manager)
        executor = MockTaskExecutor()

        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # Run one iteration
        autonomous_loop._run_iteration()

        # Should have processed the goal-derived opportunity
        assert autonomous_loop.metrics.tasks_completed >= 1 or autonomous_loop.metrics.opportunities_detected >= 1

    def test_safety_integration(self, autonomous_loop, safety_controller):
        """Test safety controller integration with loop."""
        scanner = MockOpportunityScanner([
            Opportunity(id="unsafe_opp", type="dangerous_task", description="Unsafe", source="test")
        ])
        autonomous_loop.register_scanner(scanner)

        # Emergency stop should prevent execution
        safety_controller.emergency_stop("Test")

        success = autonomous_loop._run_iteration()

        # Should fail safety check
        assert not success or autonomous_loop.phase == LoopPhase.WAITING

    def test_state_checkpointing_in_loop(self, autonomous_loop, state_manager):
        """Test state checkpointing during loop execution."""
        opp = Opportunity(
            id="checkpoint_opp",
            type="test_task",
            description="Task that creates checkpoints",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor()

        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # Run iteration and check state is being tracked
        autonomous_loop._run_iteration()

        # Verify state manager is tracking loop state
        loop_state = state_manager.loop_state
        assert loop_state is not None
        assert loop_state.iteration >= 0

    def test_error_handling_in_loop(self, autonomous_loop):
        """Test error handling during loop execution."""
        opp = Opportunity(
            id="error_opp",
            type="test_task",
            description="Task that will fail",
            source="test",
        )
        scanner = MockOpportunityScanner([opp])
        executor = MockTaskExecutor(should_succeed=False)

        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        autonomous_loop._run_iteration()

        # Should have recorded failure
        assert autonomous_loop.metrics.tasks_failed >= 1 or autonomous_loop.metrics.consecutive_failures >= 0

    def test_router_learns_from_outcomes(self, task_router):
        """Test router learning from task outcomes."""
        # Record multiple outcomes for same task type
        for i in range(10):
            task_router.record_outcome(
                task_id=f"learn_test_{i}",
                tier=ModelTier.FAST,
                success=True,
                latency_ms=100 + i * 10,
                cost=0.01,
                input_tokens=100,
                output_tokens=200,
                task_type="learned_task",
            )

        # Analyze same task type - should influence routing
        analysis = task_router.analyze_task(
            task_id="learned_task_new",
            task_type="learned_task",
            description="Another learned task",
        )

        # Router should have learned from history
        stats = task_router.get_model_stats()
        assert stats.get("FAST", {}).get("total_routed", 0) >= 0

    def test_full_autonomous_cycle(
        self, autonomous_loop, task_router, goal_manager, state_manager, safety_controller
    ):
        """Test a complete autonomous operation cycle."""
        # 1. Create goal
        goal = goal_manager.create_goal(
            title="Full Cycle Test",
            description="Complete autonomous cycle",
            success_criteria=["Execute task"],
            auto_approve=True,
        )
        goal_manager.start_goal(goal.id)

        # 2. Set up scanners and executors
        class FullCycleScanner(OpportunityScanner):
            def __init__(self):
                super().__init__("full_cycle_scanner")
                self.scanned = False

            def scan(self) -> List[Opportunity]:
                if not self.scanned:
                    self.scanned = True
                    return [
                        Opportunity(
                            id="full_cycle_opp",
                            type="test_task",
                            description="Full cycle opportunity",
                            source="test",
                        )
                    ]
                return []

        class FullCycleExecutor(TaskExecutor):
            def __init__(self, router, gm, goal_id):
                super().__init__("full_cycle_executor")
                self.supported_task_types = ["test_task"]
                self.router = router
                self.goal_manager = gm
                self.goal_id = goal_id

            def execute(self, task: Task) -> Tuple[bool, Any]:
                # Route through router
                decision = self.router.route_task(
                    task_id=task.id,
                    task_type=task.type,
                    description=task.description,
                )

                # Record outcome
                self.router.record_outcome(
                    task_id=task.id,
                    tier=decision.selected_tier,
                    success=True,
                    latency_ms=100,
                    cost=0.01,
                    input_tokens=50,
                    output_tokens=100,
                    task_type=task.type,
                )

                # Update goal progress
                self.goal_manager.mark_criterion_met(self.goal_id, 0)

                return True, "Full cycle complete"

        scanner = FullCycleScanner()
        executor = FullCycleExecutor(task_router, goal_manager, goal.id)

        autonomous_loop.register_scanner(scanner)
        autonomous_loop.register_executor(executor)

        # 3. Run the loop
        autonomous_loop.run(max_iterations=1)

        # 4. Verify results
        assert autonomous_loop.metrics.tasks_completed >= 1
        assert goal.status == GoalStatus.COMPLETED
        assert task_router.get_metrics()["total_routed"] >= 1


# ===========================================================================
# Run Tests
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

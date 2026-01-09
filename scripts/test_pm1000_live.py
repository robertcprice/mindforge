#!/usr/bin/env python3
"""
Live PM-1000 Autonomy System Test

This script runs the PM-1000 autonomous loop through realistic scenarios:
1. Code quality scanning (TODO detection, test coverage)
2. Goal-driven task execution
3. Multi-tier model routing
4. Safety constraint enforcement
5. State persistence and recovery

Run with: python3 scripts/test_pm1000_live.py
"""

import sys
import os
import time
import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard" / "backend"))

from autonomy import (
    # Phase 0 - Foundation
    init_safety_controller,
    init_state_manager,
    init_config_manager,
    get_error_handler,
    SafetyLevel,
    ActionType,
    Action,
    CheckpointType,
    is_safe_to_operate,
    emergency_stop,

    # Phase 1 - Core Loop
    LoopPhase,
    TaskPriority,
    Opportunity,
    Task,
    OpportunityScanner,
    TaskExecutor,
    init_autonomous_loop,

    ModelTier,
    TaskComplexity,
    init_task_router,

    GoalLevel,
    GoalStatus,
    GoalSource,
    GoalPriority,
    init_goal_manager,
)


# =============================================================================
# Test Configuration
# =============================================================================

TEST_CONFIG = {
    "loop_iterations": 10,
    "iteration_interval": 1.0,  # seconds
    "verbose": True,
    "test_safety_stops": True,
    "test_goal_decomposition": True,
    "test_model_routing": True,
}


# =============================================================================
# Custom Opportunity Scanners
# =============================================================================

class CodeQualityScanner(OpportunityScanner):
    """Scans for code quality improvement opportunities."""

    def __init__(self, project_path: str = "."):
        super().__init__("code_quality_scanner")
        self.project_path = Path(project_path)
        self.scan_interval = 60  # Scan every 60 seconds
        self._scanned_files = set()

    def scan(self) -> List[Opportunity]:
        opportunities = []

        # Simulate finding TODO comments
        todo_files = [
            ("src/auth.py", "TODO: Add rate limiting"),
            ("src/api.py", "TODO: Implement caching"),
            ("tests/test_auth.py", "TODO: Add edge case tests"),
        ]

        for file_path, todo_text in todo_files:
            if file_path not in self._scanned_files:
                self._scanned_files.add(file_path)
                opportunities.append(Opportunity(
                    id=f"todo_{hash(file_path) % 10000}",
                    type="todo_fix",
                    description=f"Fix TODO in {file_path}: {todo_text}",
                    source=self.name,
                    priority=TaskPriority.MEDIUM,
                    estimated_effort=0.5,
                    estimated_value=2.0,
                    confidence=0.9,
                    context={
                        "file": file_path,
                        "todo_text": todo_text,
                    }
                ))

        return opportunities


class TestCoverageScanner(OpportunityScanner):
    """Scans for test coverage improvement opportunities."""

    def __init__(self):
        super().__init__("test_coverage_scanner")
        self.scan_interval = 120

    def scan(self) -> List[Opportunity]:
        # Simulate finding untested modules
        untested_modules = [
            ("auth_module", 45),  # 45% coverage
            ("api_module", 60),   # 60% coverage
        ]

        opportunities = []
        for module, coverage in untested_modules:
            if coverage < 80:
                opportunities.append(Opportunity(
                    id=f"coverage_{module}",
                    type="test_add",
                    description=f"Improve test coverage for {module} (currently {coverage}%)",
                    source=self.name,
                    priority=TaskPriority.HIGH if coverage < 50 else TaskPriority.MEDIUM,
                    estimated_effort=2.0,
                    estimated_value=3.0,
                    confidence=0.85,
                    context={
                        "module": module,
                        "current_coverage": coverage,
                        "target_coverage": 80,
                    }
                ))

        return opportunities


class GoalDrivenScanner(OpportunityScanner):
    """Creates opportunities from active goals."""

    def __init__(self, goal_manager):
        super().__init__("goal_driven_scanner")
        self.goal_manager = goal_manager
        self.scan_interval = 30

    def scan(self) -> List[Opportunity]:
        opportunities = []
        active_goals = self.goal_manager.get_active_goals()

        for goal in active_goals:
            if goal.level == GoalLevel.TASK and goal.progress < 1.0:
                opportunities.append(Opportunity(
                    id=f"goal_{goal.id}",
                    type="goal_task",
                    description=goal.description,
                    source=self.name,
                    priority=TaskPriority(min(goal.priority.value, 4)),
                    estimated_effort=goal.metadata.estimated_effort_hours,
                    estimated_value=goal.metadata.value_score,
                    confidence=goal.metadata.confidence,
                    context={
                        "goal_id": goal.id,
                        "goal_title": goal.title,
                    }
                ))

        return opportunities


# =============================================================================
# Custom Task Executors
# =============================================================================

class SimulatedCodeExecutor(TaskExecutor):
    """Simulates code-related task execution."""

    def __init__(self, task_router, success_rate: float = 0.9):
        super().__init__("simulated_code_executor")
        self.supported_task_types = ["todo_fix", "test_add", "goal_task", "code_review"]
        self.task_router = task_router
        self.success_rate = success_rate
        self.execution_log = []

    def execute(self, task: Task) -> Tuple[bool, Any]:
        start_time = time.time()

        # Route task through the router
        decision = self.task_router.route_task(
            task_id=task.id,
            task_type=task.type,
            description=task.description,
            context=task.context,
        )

        # Simulate execution time based on tier
        tier_delays = {
            ModelTier.LOCAL: 0.1,
            ModelTier.FAST: 0.2,
            ModelTier.CAPABLE: 0.4,
            ModelTier.EXPERT: 0.8,
        }
        time.sleep(tier_delays.get(decision.selected_tier, 0.2))

        # Simulate success/failure
        import random
        success = random.random() < self.success_rate

        elapsed = time.time() - start_time
        result = {
            "task_id": task.id,
            "task_type": task.type,
            "routed_to": decision.selected_tier.name,
            "model": decision.selected_model.name,
            "success": success,
            "elapsed_time": elapsed,
        }

        # Record outcome
        self.task_router.record_outcome(
            task_id=task.id,
            tier=decision.selected_tier,
            success=success,
            latency_ms=elapsed * 1000,
            cost=decision.estimated_cost,
            input_tokens=500,
            output_tokens=1000,
            task_type=task.type,
        )

        self.execution_log.append(result)

        if success:
            return True, result
        else:
            return False, f"Simulated failure for task {task.id}"


# =============================================================================
# Test Harness
# =============================================================================

class PM1000TestHarness:
    """Comprehensive test harness for PM-1000 autonomy system."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or TEST_CONFIG
        self.results = {
            "start_time": None,
            "end_time": None,
            "iterations": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "goals_created": 0,
            "goals_completed": 0,
            "routing_decisions": [],
            "phase_transitions": [],
            "safety_events": [],
            "errors": [],
        }

        # Create temp directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_state.db")

        # Remove any existing kill switch
        kill_switch = "/tmp/PM1000_EMERGENCY_STOP"
        if os.path.exists(kill_switch):
            os.remove(kill_switch)

        self._init_components()

    def _init_components(self):
        """Initialize all PM-1000 components."""
        print("\n" + "="*60)
        print("Initializing PM-1000 Autonomy System")
        print("="*60)

        # Initialize Phase 0 components
        print("\n[1/6] Initializing Config Manager...")
        self.config_mgr = init_config_manager()

        print("[2/6] Initializing State Manager...")
        self.state_mgr = init_state_manager(db_path=self.db_path)

        print("[3/6] Initializing Safety Controller...")
        self.safety = init_safety_controller()

        print("[4/6] Initializing Error Handler...")
        self.error_handler = get_error_handler()

        # Initialize Phase 1 components
        print("[5/6] Initializing Task Router...")
        self.router = init_task_router(
            config_manager=self.config_mgr,
            state_manager=self.state_mgr,
        )

        print("[6/6] Initializing Goal Manager...")
        self.goal_mgr = init_goal_manager(
            config_manager=self.config_mgr,
            state_manager=self.state_mgr,
            safety_controller=self.safety,
        )

        # Initialize Autonomous Loop
        print("\nInitializing Autonomous Loop...")
        self.loop = init_autonomous_loop(
            safety_controller=self.safety,
            state_manager=self.state_mgr,
            error_handler=self.error_handler,
            config_manager=self.config_mgr,
            min_iteration_interval=self.config["iteration_interval"],
        )

        # Register scanners
        print("\nRegistering Opportunity Scanners...")
        self.code_scanner = CodeQualityScanner()
        self.coverage_scanner = TestCoverageScanner()
        self.goal_scanner = GoalDrivenScanner(self.goal_mgr)

        self.loop.register_scanner(self.code_scanner)
        self.loop.register_scanner(self.coverage_scanner)
        self.loop.register_scanner(self.goal_scanner)
        print(f"  - Registered {len([self.code_scanner, self.coverage_scanner, self.goal_scanner])} scanners")

        # Register executor
        print("\nRegistering Task Executors...")
        self.executor = SimulatedCodeExecutor(self.router)
        self.loop.register_executor(self.executor)
        print(f"  - Registered executor with {len(self.executor.supported_task_types)} supported task types")

        # Register callbacks
        self._setup_callbacks()

        print("\n✓ PM-1000 Autonomy System initialized")

    def _setup_callbacks(self):
        """Set up monitoring callbacks."""
        def on_phase_change(old, new):
            self.results["phase_transitions"].append({
                "timestamp": datetime.now().isoformat(),
                "from": old.value,
                "to": new.value,
            })
            if self.config["verbose"]:
                print(f"  Phase: {old.value} → {new.value}")

        def on_task_complete(task, success):
            if success:
                self.results["tasks_completed"] += 1
            else:
                self.results["tasks_failed"] += 1
            if self.config["verbose"]:
                status = "✓" if success else "✗"
                print(f"  {status} Task {task.id[:20]}... ({task.type})")

        def on_opportunity(opp):
            if self.config["verbose"]:
                print(f"  📡 Detected: {opp.description[:50]}...")

        def on_goal_created(goal):
            self.results["goals_created"] += 1
            if self.config["verbose"]:
                print(f"  🎯 Goal created: {goal.title}")

        def on_goal_completed(goal):
            self.results["goals_completed"] += 1
            if self.config["verbose"]:
                print(f"  ✅ Goal completed: {goal.title}")

        self.loop.on_phase_change(on_phase_change)
        self.loop.on_task_complete(on_task_complete)
        self.loop.on_opportunity_detected(on_opportunity)
        self.goal_mgr.on_goal_created(on_goal_created)
        self.goal_mgr.on_goal_completed(on_goal_completed)

    def test_safety_system(self):
        """Test the safety system."""
        print("\n" + "-"*60)
        print("Testing Safety System")
        print("-"*60)

        # Test safety check
        safe, reason = is_safe_to_operate()
        print(f"\n[Safety Check] Safe to operate: {safe}")
        if not safe:
            print(f"  Reason: {reason}")

        # Test action validation
        test_action = Action(
            action_type=ActionType.FILE_MODIFY,
            description="Modify test file",
            target="/tmp/test.txt",
        )
        result = self.safety.validate_action(test_action)
        print(f"\n[Action Validation] File modify action: {'✓ Passed' if result.passed else '✗ Failed'}")

        # Test dangerous action
        dangerous_action = Action(
            action_type=ActionType.FILE_DELETE,
            description="Delete important file",
            target="/etc/passwd",
        )
        result = self.safety.validate_action(dangerous_action)
        print(f"[Action Validation] Dangerous action: {'✓ Blocked' if not result.passed else '✗ Allowed (BAD!)'}")

        if self.config["test_safety_stops"]:
            print("\n[Safety Test] Testing emergency stop and recovery...")

            # Trigger emergency stop
            self.safety.emergency_stop("Test emergency stop")
            safe, reason = is_safe_to_operate()
            print(f"  After emergency stop: Safe={safe}")
            self.results["safety_events"].append({
                "type": "emergency_stop",
                "timestamp": datetime.now().isoformat(),
                "reason": "Test emergency stop",
            })

            # Reset for continued testing
            kill_switch = "/tmp/PM1000_EMERGENCY_STOP"
            if os.path.exists(kill_switch):
                os.remove(kill_switch)
            # Also reset the internal kill switch state
            reset_results = self.safety.reset_all_kill_switches()
            print(f"  Reset kill switches: {sum(reset_results.values())}/{len(reset_results)} reset")

        print("\n✓ Safety system tests complete")

    def test_goal_management(self):
        """Test goal creation and decomposition."""
        print("\n" + "-"*60)
        print("Testing Goal Management")
        print("-"*60)

        # Create a high-level objective
        objective = self.goal_mgr.create_goal(
            title="Improve Code Quality",
            description="Systematically improve codebase quality through testing and refactoring",
            level=GoalLevel.OBJECTIVE,
            priority=GoalPriority.HIGH,
            source=GoalSource.USER,
            success_criteria=[
                "Achieve 80% test coverage",
                "Fix all TODO comments",
                "Pass code review",
            ],
            auto_approve=True,
        )
        print(f"\n[Goal Created] {objective.title} ({objective.id})")
        print(f"  Status: {objective.status.value}")
        print(f"  Criteria: {len(objective.success_criteria)} success criteria")

        if self.config["test_goal_decomposition"]:
            print("\n[Goal Decomposition] Breaking down objective into sub-goals...")
            sub_goals = self.goal_mgr.decompose_goal(objective.id)
            for sg in sub_goals:
                print(f"  → {sg.title} ({sg.level.name})")
                self.goal_mgr.approve_goal(sg.id)

        # Create a task-level goal
        task_goal = self.goal_mgr.create_goal(
            title="Add unit tests for auth module",
            description="Write comprehensive unit tests for the authentication module",
            level=GoalLevel.TASK,
            priority=GoalPriority.MEDIUM,
            parent_id=objective.id,
            auto_approve=True,
        )
        self.goal_mgr.start_goal(task_goal.id)
        print(f"\n[Task Goal] {task_goal.title}")
        print(f"  Status: {task_goal.status.value}")

        print("\n✓ Goal management tests complete")

    def test_task_routing(self):
        """Test multi-tier task routing."""
        print("\n" + "-"*60)
        print("Testing Task Router")
        print("-"*60)

        test_cases = [
            ("code_format", "Format config file", TaskComplexity.TRIVIAL),
            ("bug_fix", "Fix authentication bug", TaskComplexity.MODERATE),
            ("refactor", "Refactor entire auth system", TaskComplexity.COMPLEX),
            ("architecture", "Design new microservices architecture", TaskComplexity.EXPERT),
        ]

        print("\n[Routing Tests]")
        for task_type, description, expected_complexity in test_cases:
            analysis = self.router.analyze_task(
                task_id=f"test_{task_type}",
                task_type=task_type,
                description=description,
            )
            decision = self.router.route(analysis)

            complexity_match = "✓" if analysis.estimated_complexity == expected_complexity else "~"
            print(f"  {complexity_match} {task_type}: {analysis.estimated_complexity.name} → {decision.selected_tier.name}")

            self.results["routing_decisions"].append({
                "task_type": task_type,
                "complexity": analysis.estimated_complexity.name,
                "tier": decision.selected_tier.name,
                "model": decision.selected_model.name,
            })

        # Show model stats
        print("\n[Model Statistics]")
        stats = self.router.get_model_stats()
        for tier, info in stats.items():
            print(f"  {tier}: {info['name']} (success: {info['success_rate']:.0%})")

        print("\n✓ Task routing tests complete")

    def run_autonomous_loop(self):
        """Run the autonomous loop for configured iterations."""
        print("\n" + "-"*60)
        print(f"Running Autonomous Loop ({self.config['loop_iterations']} iterations)")
        print("-"*60)

        self.results["start_time"] = datetime.now().isoformat()

        print("\n[Loop Starting]")
        print(f"  Iteration interval: {self.config['iteration_interval']}s")
        print(f"  Scanners: {len(self.loop._scanners)}")
        print(f"  Executors: {len(self.loop._executors)}")

        try:
            for i in range(self.config["loop_iterations"]):
                print(f"\n--- Iteration {i+1}/{self.config['loop_iterations']} ---")

                success = self.loop._run_iteration()
                self.results["iterations"] += 1

                if not success:
                    print("  ⚠ Iteration did not complete successfully")

                # Brief pause between iterations
                if i < self.config["loop_iterations"] - 1:
                    time.sleep(self.config["iteration_interval"])

        except KeyboardInterrupt:
            print("\n\n[Interrupted] Stopping loop gracefully...")
            self.loop.stop()
        except Exception as e:
            self.results["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"\n[Error] {type(e).__name__}: {e}")

        self.results["end_time"] = datetime.now().isoformat()

        # Print summary
        print("\n" + "="*60)
        print("Loop Execution Summary")
        print("="*60)

        metrics = self.loop.metrics
        print(f"\n  Iterations: {metrics.iterations}")
        print(f"  Tasks Completed: {metrics.tasks_completed}")
        print(f"  Tasks Failed: {metrics.tasks_failed}")
        print(f"  Opportunities Detected: {metrics.opportunities_detected}")
        print(f"  Success Rate: {metrics.tasks_completed / max(1, metrics.tasks_completed + metrics.tasks_failed):.0%}")
        print(f"  Total Execution Time: {metrics.total_execution_time:.2f}s")

        if metrics.consecutive_failures > 0:
            print(f"  ⚠ Consecutive Failures: {metrics.consecutive_failures}")

    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("PM-1000 Test Report")
        print("="*60)

        # Loop metrics
        loop_metrics = self.loop.metrics.to_dict()
        print("\n[Autonomous Loop Metrics]")
        for key, value in loop_metrics.items():
            print(f"  {key}: {value}")

        # Router metrics
        router_metrics = self.router.get_metrics()
        print("\n[Task Router Metrics]")
        for key, value in router_metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Goal metrics
        goal_status = self.goal_mgr.get_status()
        print("\n[Goal Manager Status]")
        for key, value in goal_status.items():
            if key != "metrics":
                print(f"  {key}: {value}")

        # Execution log
        print(f"\n[Execution Log] {len(self.executor.execution_log)} tasks executed")
        for entry in self.executor.execution_log[-5:]:
            print(f"  - {entry['task_id'][:25]}... → {entry['routed_to']} ({'✓' if entry['success'] else '✗'})")

        # Save full report
        report = {
            "config": self.config,
            "results": self.results,
            "loop_metrics": loop_metrics,
            "router_metrics": router_metrics,
            "goal_status": goal_status,
            "execution_log": self.executor.execution_log,
        }

        report_path = Path(self.temp_dir) / "pm1000_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[Report saved] {report_path}")

        return report

    def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "="*60)
        print("PM-1000 COMPREHENSIVE TEST SUITE")
        print("="*60)
        print(f"Started: {datetime.now().isoformat()}")

        try:
            # Phase 1: Safety tests
            self.test_safety_system()

            # Phase 2: Goal management tests
            self.test_goal_management()

            # Phase 3: Task routing tests
            self.test_task_routing()

            # Phase 4: Run autonomous loop
            self.run_autonomous_loop()

            # Generate report
            report = self.generate_report()

            print("\n" + "="*60)
            print("TEST SUITE COMPLETE")
            print("="*60)
            print(f"\nTotal Tasks Executed: {self.results['tasks_completed'] + self.results['tasks_failed']}")
            print(f"Success Rate: {self.results['tasks_completed'] / max(1, self.results['tasks_completed'] + self.results['tasks_failed']):.0%}")
            print(f"Goals Created: {self.results['goals_created']}")
            print(f"Goals Completed: {self.results['goals_completed']}")
            print(f"Errors: {len(self.results['errors'])}")

            return True

        except Exception as e:
            print(f"\n[FATAL ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    harness = PM1000TestHarness(TEST_CONFIG)
    success = harness.run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Anti-Goal Generator for PM-1000

Generate goals that PREVENT bad outcomes, not just achieve good ones.
Prevention is more valuable than correction - it's cheaper to stop tech debt
from accumulating than to pay it down later.

Anti-Goal Philosophy:
- Proactive defense beats reactive fixes
- Small consistent prevention > large periodic cleanup
- Early detection = cheaper resolution
- Constraint satisfaction > feature addition

Anti-Goal Types:
- PREVENT_TECH_DEBT_ACCUMULATION: Stop before it compounds
- PREVENT_TEST_BRITTLENESS: Replace fragile tests before they break
- PREVENT_DOC_DRIFT: Sync comments with code before confusion
- PREVENT_DEPENDENCY_HELL: Update before conflicts arise
- PREVENT_ARCHITECTURE_EROSION: Enforce boundaries before blur
- PREVENT_SECURITY_REGRESSION: Block before exposure
- PREVENT_PERFORMANCE_DEGRADATION: Catch before users notice
"""

import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

from .autonomous_loop import Opportunity, TaskPriority
from .goal_manager import Goal, GoalLevel, GoalStatus, GoalSource, GoalPriority

logger = get_logger("pm1000.autonomy.antigoal")


class AntiGoalType(Enum):
    """Types of anti-goals (things to prevent)."""
    PREVENT_TECH_DEBT = "prevent_tech_debt"
    PREVENT_TEST_BRITTLENESS = "prevent_test_brittleness"
    PREVENT_DOC_DRIFT = "prevent_doc_drift"
    PREVENT_DEPENDENCY_HELL = "prevent_dependency_hell"
    PREVENT_ARCHITECTURE_EROSION = "prevent_architecture_erosion"
    PREVENT_SECURITY_REGRESSION = "prevent_security_regression"
    PREVENT_PERFORMANCE_DEGRADATION = "prevent_performance_degradation"
    PREVENT_CODE_COMPLEXITY = "prevent_code_complexity"
    PREVENT_COUPLING = "prevent_coupling"
    PREVENT_KNOWLEDGE_LOSS = "prevent_knowledge_loss"


class TriggerSeverity(Enum):
    """Severity of an anti-goal trigger."""
    INFO = 0       # Worth noting, no action needed yet
    WARNING = 1    # Should address soon
    CRITICAL = 2   # Must address immediately
    EMERGENCY = 3  # System at risk


@dataclass
class AntiGoalTrigger:
    """Evidence that an anti-goal condition is emerging."""
    anti_goal_type: AntiGoalType
    severity: TriggerSeverity
    description: str
    evidence: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    prevention_action: Optional[str] = None
    estimated_cost_if_ignored: float = 0.0  # Cost in hours if not addressed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anti_goal_type": self.anti_goal_type.value,
            "severity": self.severity.name,
            "description": self.description,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
            "prevention_action": self.prevention_action,
            "estimated_cost_if_ignored": self.estimated_cost_if_ignored,
        }


@dataclass
class AntiGoal:
    """
    An anti-goal represents something to PREVENT from happening.

    Unlike regular goals that aim to achieve something, anti-goals
    aim to maintain a desired state by preventing degradation.
    """
    id: str
    type: AntiGoalType
    name: str
    description: str

    # Detection configuration
    detection_method: str  # How to detect this threat
    detection_threshold: float = 0.5  # 0-1, when to trigger
    check_interval_seconds: int = 300  # How often to check

    # Prevention configuration
    prevention_actions: List[str] = field(default_factory=list)
    auto_prevent: bool = False  # Can PM-1000 auto-fix?
    max_auto_severity: TriggerSeverity = TriggerSeverity.WARNING

    # Tracking
    enabled: bool = True
    last_checked: Optional[datetime] = None
    current_severity: TriggerSeverity = TriggerSeverity.INFO
    trigger_history: List[AntiGoalTrigger] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "detection_method": self.detection_method,
            "detection_threshold": self.detection_threshold,
            "check_interval_seconds": self.check_interval_seconds,
            "prevention_actions": self.prevention_actions,
            "auto_prevent": self.auto_prevent,
            "max_auto_severity": self.max_auto_severity.name,
            "enabled": self.enabled,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "current_severity": self.current_severity.name,
            "trigger_count": len(self.trigger_history),
        }


class AntiGoalDetector:
    """Base class for anti-goal detection."""

    def __init__(self, anti_goal_type: AntiGoalType):
        self.anti_goal_type = anti_goal_type

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect if the anti-goal condition is emerging."""
        raise NotImplementedError

    def get_prevention_action(self, trigger: AntiGoalTrigger) -> str:
        """Get the recommended prevention action."""
        raise NotImplementedError


class TechDebtDetector(AntiGoalDetector):
    """Detects accumulating technical debt."""

    # Indicators of tech debt
    DEBT_INDICATORS = {
        "todo_count": {"threshold": 20, "weight": 1.0},
        "fixme_count": {"threshold": 5, "weight": 2.0},
        "hack_count": {"threshold": 3, "weight": 3.0},
        "complexity_violations": {"threshold": 10, "weight": 1.5},
        "duplicate_code_blocks": {"threshold": 5, "weight": 2.0},
    }

    def __init__(self):
        super().__init__(AntiGoalType.PREVENT_TECH_DEBT)

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect tech debt accumulation."""
        debt_score = 0.0
        evidence = {}

        for indicator, config in self.DEBT_INDICATORS.items():
            value = context.get(indicator, 0)
            if value > config["threshold"]:
                debt_score += (value / config["threshold"]) * config["weight"]
                evidence[indicator] = {
                    "value": value,
                    "threshold": config["threshold"],
                    "exceeded_by": value - config["threshold"],
                }

        if debt_score > 0:
            severity = TriggerSeverity.INFO
            if debt_score > 5:
                severity = TriggerSeverity.WARNING
            if debt_score > 10:
                severity = TriggerSeverity.CRITICAL

            return AntiGoalTrigger(
                anti_goal_type=self.anti_goal_type,
                severity=severity,
                description=f"Tech debt accumulating (score: {debt_score:.1f})",
                evidence=evidence,
                prevention_action="Address highest-weight debt indicators first",
                estimated_cost_if_ignored=debt_score * 2,  # Hours of future work
            )

        return None


class TestBrittlenessDetector(AntiGoalDetector):
    """Detects brittle or flaky tests."""

    def __init__(self):
        super().__init__(AntiGoalType.PREVENT_TEST_BRITTLENESS)

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect test brittleness."""
        evidence = {}

        flaky_tests = context.get("flaky_test_count", 0)
        mock_heavy_tests = context.get("mock_heavy_tests", 0)
        slow_tests = context.get("slow_tests", 0)
        no_assertion_tests = context.get("tests_without_assertions", 0)

        issues = []
        if flaky_tests > 0:
            issues.append(f"{flaky_tests} flaky tests")
            evidence["flaky_tests"] = flaky_tests

        if mock_heavy_tests > 5:
            issues.append(f"{mock_heavy_tests} mock-heavy tests")
            evidence["mock_heavy_tests"] = mock_heavy_tests

        if slow_tests > 10:
            issues.append(f"{slow_tests} slow tests (>1s)")
            evidence["slow_tests"] = slow_tests

        if no_assertion_tests > 0:
            issues.append(f"{no_assertion_tests} tests without assertions")
            evidence["no_assertion_tests"] = no_assertion_tests

        if issues:
            severity = TriggerSeverity.INFO
            if flaky_tests > 0 or no_assertion_tests > 0:
                severity = TriggerSeverity.WARNING
            if flaky_tests > 5:
                severity = TriggerSeverity.CRITICAL

            return AntiGoalTrigger(
                anti_goal_type=self.anti_goal_type,
                severity=severity,
                description=f"Test suite health issues: {', '.join(issues)}",
                evidence=evidence,
                prevention_action="Fix flaky tests, reduce mocking, add assertions",
                estimated_cost_if_ignored=flaky_tests * 0.5 + mock_heavy_tests * 0.25,
            )

        return None


class DocDriftDetector(AntiGoalDetector):
    """Detects documentation drift from code."""

    def __init__(self):
        super().__init__(AntiGoalType.PREVENT_DOC_DRIFT)

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect documentation drift."""
        evidence = {}

        outdated_docstrings = context.get("outdated_docstrings", 0)
        missing_changelog = context.get("missing_changelog_entries", 0)
        stale_readme_days = context.get("readme_age_days", 0)
        api_doc_coverage = context.get("api_doc_coverage", 100)

        issues = []
        if outdated_docstrings > 5:
            issues.append(f"{outdated_docstrings} potentially outdated docstrings")
            evidence["outdated_docstrings"] = outdated_docstrings

        if missing_changelog > 0:
            issues.append(f"{missing_changelog} undocumented changes")
            evidence["missing_changelog"] = missing_changelog

        if stale_readme_days > 90:
            issues.append(f"README not updated in {stale_readme_days} days")
            evidence["stale_readme_days"] = stale_readme_days

        if api_doc_coverage < 80:
            issues.append(f"API documentation at {api_doc_coverage}%")
            evidence["api_doc_coverage"] = api_doc_coverage

        if issues:
            severity = TriggerSeverity.INFO
            if api_doc_coverage < 50 or stale_readme_days > 180:
                severity = TriggerSeverity.WARNING

            return AntiGoalTrigger(
                anti_goal_type=self.anti_goal_type,
                severity=severity,
                description=f"Documentation drift detected: {', '.join(issues)}",
                evidence=evidence,
                prevention_action="Update documentation to match current implementation",
                estimated_cost_if_ignored=len(issues) * 1.0,
            )

        return None


class SecurityRegressionDetector(AntiGoalDetector):
    """Detects security regressions and vulnerabilities."""

    def __init__(self):
        super().__init__(AntiGoalType.PREVENT_SECURITY_REGRESSION)

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect security issues."""
        evidence = {}

        vulnerable_deps = context.get("vulnerable_dependencies", 0)
        hardcoded_secrets = context.get("hardcoded_secrets", 0)
        unsafe_patterns = context.get("unsafe_code_patterns", 0)
        missing_auth_checks = context.get("missing_auth_checks", 0)

        issues = []
        severity = TriggerSeverity.INFO

        if vulnerable_deps > 0:
            issues.append(f"{vulnerable_deps} vulnerable dependencies")
            evidence["vulnerable_deps"] = vulnerable_deps
            severity = TriggerSeverity.WARNING

        if hardcoded_secrets > 0:
            issues.append(f"{hardcoded_secrets} potential hardcoded secrets")
            evidence["hardcoded_secrets"] = hardcoded_secrets
            severity = TriggerSeverity.CRITICAL

        if unsafe_patterns > 0:
            issues.append(f"{unsafe_patterns} unsafe code patterns")
            evidence["unsafe_patterns"] = unsafe_patterns
            if severity.value < TriggerSeverity.WARNING.value:
                severity = TriggerSeverity.WARNING

        if missing_auth_checks > 0:
            issues.append(f"{missing_auth_checks} endpoints without auth")
            evidence["missing_auth_checks"] = missing_auth_checks
            severity = TriggerSeverity.CRITICAL

        if issues:
            return AntiGoalTrigger(
                anti_goal_type=self.anti_goal_type,
                severity=severity,
                description=f"Security concerns: {', '.join(issues)}",
                evidence=evidence,
                prevention_action="Address security issues immediately, prioritize secrets and auth",
                estimated_cost_if_ignored=100.0 if hardcoded_secrets > 0 else 10.0,  # High cost for security
            )

        return None


class PerformanceDegradationDetector(AntiGoalDetector):
    """Detects performance degradation."""

    def __init__(self):
        super().__init__(AntiGoalType.PREVENT_PERFORMANCE_DEGRADATION)

    def detect(self, context: Dict[str, Any]) -> Optional[AntiGoalTrigger]:
        """Detect performance issues."""
        evidence = {}

        slow_endpoints = context.get("slow_endpoints", 0)
        memory_leaks = context.get("potential_memory_leaks", 0)
        n_plus_one_queries = context.get("n_plus_one_queries", 0)
        unoptimized_loops = context.get("unoptimized_loops", 0)

        issues = []
        if slow_endpoints > 0:
            issues.append(f"{slow_endpoints} slow endpoints")
            evidence["slow_endpoints"] = slow_endpoints

        if memory_leaks > 0:
            issues.append(f"{memory_leaks} potential memory leaks")
            evidence["memory_leaks"] = memory_leaks

        if n_plus_one_queries > 0:
            issues.append(f"{n_plus_one_queries} N+1 query patterns")
            evidence["n_plus_one_queries"] = n_plus_one_queries

        if unoptimized_loops > 5:
            issues.append(f"{unoptimized_loops} unoptimized loops")
            evidence["unoptimized_loops"] = unoptimized_loops

        if issues:
            severity = TriggerSeverity.INFO
            if memory_leaks > 0 or slow_endpoints > 5:
                severity = TriggerSeverity.WARNING
            if memory_leaks > 3:
                severity = TriggerSeverity.CRITICAL

            return AntiGoalTrigger(
                anti_goal_type=self.anti_goal_type,
                severity=severity,
                description=f"Performance concerns: {', '.join(issues)}",
                evidence=evidence,
                prevention_action="Profile and optimize hot paths, fix memory leaks",
                estimated_cost_if_ignored=memory_leaks * 5 + slow_endpoints * 1,
            )

        return None


class AntiGoalGenerator:
    """
    Generates and manages anti-goals for prevention-focused operation.

    Anti-goals are continuously monitored conditions that should NOT occur.
    When a trigger is detected, prevention actions are recommended or
    automatically executed based on configuration.
    """

    def __init__(self):
        self._anti_goals: Dict[str, AntiGoal] = {}
        self._detectors: Dict[AntiGoalType, AntiGoalDetector] = {}
        self._lock = threading.RLock()

        # Statistics
        self._total_triggers = 0
        self._prevented_issues = 0
        self._auto_prevented = 0

        # Register default anti-goals and detectors
        self._register_defaults()

        logger.info("AntiGoalGenerator initialized with %d anti-goals", len(self._anti_goals))

    def _register_defaults(self):
        """Register default anti-goals and their detectors."""
        # Register detectors
        self._detectors = {
            AntiGoalType.PREVENT_TECH_DEBT: TechDebtDetector(),
            AntiGoalType.PREVENT_TEST_BRITTLENESS: TestBrittlenessDetector(),
            AntiGoalType.PREVENT_DOC_DRIFT: DocDriftDetector(),
            AntiGoalType.PREVENT_SECURITY_REGRESSION: SecurityRegressionDetector(),
            AntiGoalType.PREVENT_PERFORMANCE_DEGRADATION: PerformanceDegradationDetector(),
        }

        # Register anti-goals
        default_anti_goals = [
            AntiGoal(
                id="ag_tech_debt",
                type=AntiGoalType.PREVENT_TECH_DEBT,
                name="Prevent Tech Debt Accumulation",
                description="Stop technical debt from compounding before it becomes unmanageable",
                detection_method="Count TODOs, FIXMEs, complexity violations, duplicates",
                detection_threshold=0.3,
                check_interval_seconds=300,
                prevention_actions=[
                    "Address TODOs and FIXMEs",
                    "Refactor complex functions",
                    "Remove duplicate code",
                ],
                auto_prevent=True,
                max_auto_severity=TriggerSeverity.WARNING,
            ),
            AntiGoal(
                id="ag_test_brittleness",
                type=AntiGoalType.PREVENT_TEST_BRITTLENESS,
                name="Prevent Test Suite Brittleness",
                description="Keep test suite healthy and reliable",
                detection_method="Detect flaky tests, mock-heavy tests, slow tests",
                detection_threshold=0.4,
                check_interval_seconds=600,
                prevention_actions=[
                    "Fix flaky tests",
                    "Replace heavy mocks with test doubles",
                    "Optimize slow tests",
                    "Add assertions to empty tests",
                ],
                auto_prevent=True,
                max_auto_severity=TriggerSeverity.INFO,
            ),
            AntiGoal(
                id="ag_doc_drift",
                type=AntiGoalType.PREVENT_DOC_DRIFT,
                name="Prevent Documentation Drift",
                description="Keep documentation in sync with implementation",
                detection_method="Compare docstrings to function signatures, check staleness",
                detection_threshold=0.5,
                check_interval_seconds=900,
                prevention_actions=[
                    "Update docstrings",
                    "Sync README with current state",
                    "Add changelog entries",
                ],
                auto_prevent=True,
                max_auto_severity=TriggerSeverity.WARNING,
            ),
            AntiGoal(
                id="ag_security",
                type=AntiGoalType.PREVENT_SECURITY_REGRESSION,
                name="Prevent Security Regressions",
                description="Block security vulnerabilities before they reach production",
                detection_method="Scan for secrets, vulnerable deps, unsafe patterns",
                detection_threshold=0.1,  # Very sensitive
                check_interval_seconds=180,
                prevention_actions=[
                    "Remove hardcoded secrets",
                    "Update vulnerable dependencies",
                    "Add auth checks",
                    "Fix unsafe patterns",
                ],
                auto_prevent=False,  # Security requires human review
                max_auto_severity=TriggerSeverity.INFO,
            ),
            AntiGoal(
                id="ag_performance",
                type=AntiGoalType.PREVENT_PERFORMANCE_DEGRADATION,
                name="Prevent Performance Degradation",
                description="Catch performance issues before users notice",
                detection_method="Detect slow endpoints, memory leaks, N+1 queries",
                detection_threshold=0.4,
                check_interval_seconds=600,
                prevention_actions=[
                    "Optimize slow endpoints",
                    "Fix memory leaks",
                    "Add caching",
                    "Optimize database queries",
                ],
                auto_prevent=True,
                max_auto_severity=TriggerSeverity.INFO,
            ),
        ]

        for ag in default_anti_goals:
            self._anti_goals[ag.id] = ag

    def register_anti_goal(self, anti_goal: AntiGoal):
        """Register a custom anti-goal."""
        with self._lock:
            self._anti_goals[anti_goal.id] = anti_goal
            logger.info(f"Registered anti-goal: {anti_goal.name}")

    def register_detector(self, anti_goal_type: AntiGoalType, detector: AntiGoalDetector):
        """Register a custom detector for an anti-goal type."""
        with self._lock:
            self._detectors[anti_goal_type] = detector
            logger.info(f"Registered detector for: {anti_goal_type.value}")

    def check_anti_goals(self, context: Dict[str, Any]) -> List[AntiGoalTrigger]:
        """
        Check all anti-goals and return any triggers.

        Args:
            context: Dictionary containing metrics and indicators for detection

        Returns:
            List of triggered anti-goal conditions
        """
        triggers = []

        with self._lock:
            for anti_goal in self._anti_goals.values():
                if not anti_goal.enabled:
                    continue

                # Check if it's time to scan
                if anti_goal.last_checked:
                    elapsed = (datetime.now() - anti_goal.last_checked).total_seconds()
                    if elapsed < anti_goal.check_interval_seconds:
                        continue

                # Get detector
                detector = self._detectors.get(anti_goal.type)
                if not detector:
                    continue

                # Run detection
                try:
                    trigger = detector.detect(context)
                    anti_goal.last_checked = datetime.now()

                    if trigger:
                        anti_goal.current_severity = trigger.severity
                        anti_goal.trigger_history.append(trigger)

                        # Keep only last 100 triggers
                        anti_goal.trigger_history = anti_goal.trigger_history[-100:]

                        triggers.append(trigger)
                        self._total_triggers += 1

                        logger.info(
                            f"Anti-goal trigger: {anti_goal.name} "
                            f"(severity: {trigger.severity.name})"
                        )
                    else:
                        anti_goal.current_severity = TriggerSeverity.INFO

                except Exception as e:
                    logger.error(f"Error checking anti-goal {anti_goal.id}: {e}")

        return triggers

    def generate_prevention_goals(self, triggers: List[AntiGoalTrigger]) -> List[Goal]:
        """
        Generate prevention goals from anti-goal triggers.

        Converts detected threats into actionable goals.
        """
        goals = []

        for trigger in triggers:
            # Map severity to priority
            priority_map = {
                TriggerSeverity.INFO: GoalPriority.LOW,
                TriggerSeverity.WARNING: GoalPriority.MEDIUM,
                TriggerSeverity.CRITICAL: GoalPriority.HIGH,
                TriggerSeverity.EMERGENCY: GoalPriority.CRITICAL,
            }

            goal = Goal(
                id=f"prevention_{trigger.anti_goal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=GoalLevel.GOAL,
                title=f"Prevent: {trigger.description}",
                description=f"{trigger.description}\n\nRecommended action: {trigger.prevention_action}",
                priority=priority_map.get(trigger.severity, GoalPriority.MEDIUM),
                context={
                    "anti_goal_type": trigger.anti_goal_type.value,
                    "severity": trigger.severity.name,
                    "evidence": trigger.evidence,
                    "prevention_action": trigger.prevention_action,
                    "estimated_cost_if_ignored": trigger.estimated_cost_if_ignored,
                },
            )
            goal.metadata.source = GoalSource.AUTONOMOUS
            goal.metadata.confidence = 0.9
            goal.metadata.value_score = trigger.estimated_cost_if_ignored

            goals.append(goal)

        return goals

    def convert_to_opportunities(self, triggers: List[AntiGoalTrigger]) -> List[Opportunity]:
        """Convert anti-goal triggers to opportunities for the autonomous loop."""
        opportunities = []

        for trigger in triggers:
            # Map severity to priority
            priority_map = {
                TriggerSeverity.INFO: TaskPriority.LOW,
                TriggerSeverity.WARNING: TaskPriority.MEDIUM,
                TriggerSeverity.CRITICAL: TaskPriority.HIGH,
                TriggerSeverity.EMERGENCY: TaskPriority.CRITICAL,
            }

            opp = Opportunity(
                id=f"prevent_{trigger.anti_goal_type.value}_{hash(str(trigger.evidence))}",
                type=f"prevention_{trigger.anti_goal_type.value}",
                description=f"PREVENT: {trigger.description}",
                source="anti_goal_generator",
                priority=priority_map.get(trigger.severity, TaskPriority.MEDIUM),
                estimated_effort=trigger.estimated_cost_if_ignored * 0.5,  # Prevention is cheaper
                estimated_value=trigger.estimated_cost_if_ignored,  # Full value of prevention
                confidence=0.85,
                context={
                    "anti_goal_type": trigger.anti_goal_type.value,
                    "severity": trigger.severity.name,
                    "evidence": trigger.evidence,
                    "prevention_action": trigger.prevention_action,
                },
            )
            opportunities.append(opp)

        return opportunities

    def can_auto_prevent(self, trigger: AntiGoalTrigger) -> bool:
        """Check if a trigger can be automatically prevented."""
        with self._lock:
            for anti_goal in self._anti_goals.values():
                if anti_goal.type == trigger.anti_goal_type:
                    if not anti_goal.auto_prevent:
                        return False
                    if trigger.severity.value > anti_goal.max_auto_severity.value:
                        return False
                    return True
        return False

    def record_prevention(self, trigger: AntiGoalTrigger, auto: bool = False):
        """Record that a prevention action was taken."""
        with self._lock:
            self._prevented_issues += 1
            if auto:
                self._auto_prevented += 1
            logger.info(f"Prevention recorded: {trigger.anti_goal_type.value} ({'auto' if auto else 'manual'})")

    def get_anti_goals(self) -> List[Dict[str, Any]]:
        """Get all anti-goals."""
        with self._lock:
            return [ag.to_dict() for ag in self._anti_goals.values()]

    def get_anti_goal(self, anti_goal_id: str) -> Optional[AntiGoal]:
        """Get a specific anti-goal."""
        return self._anti_goals.get(anti_goal_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get anti-goal statistics."""
        with self._lock:
            by_severity = {}
            for ag in self._anti_goals.values():
                sev = ag.current_severity.name
                by_severity[sev] = by_severity.get(sev, 0) + 1

            return {
                "total_anti_goals": len(self._anti_goals),
                "enabled_anti_goals": sum(1 for ag in self._anti_goals.values() if ag.enabled),
                "total_triggers": self._total_triggers,
                "prevented_issues": self._prevented_issues,
                "auto_prevented": self._auto_prevented,
                "current_severity_distribution": by_severity,
            }

    def enable_anti_goal(self, anti_goal_id: str, enabled: bool = True):
        """Enable or disable an anti-goal."""
        with self._lock:
            if anti_goal_id in self._anti_goals:
                self._anti_goals[anti_goal_id].enabled = enabled
                logger.info(f"Anti-goal {anti_goal_id} {'enabled' if enabled else 'disabled'}")


# Global instance
_anti_goal_generator: Optional[AntiGoalGenerator] = None
_generator_lock = threading.Lock()


def get_anti_goal_generator() -> AntiGoalGenerator:
    """Get the global anti-goal generator instance."""
    global _anti_goal_generator
    with _generator_lock:
        if _anti_goal_generator is None:
            _anti_goal_generator = AntiGoalGenerator()
        return _anti_goal_generator


def init_anti_goal_generator() -> AntiGoalGenerator:
    """Initialize the global anti-goal generator."""
    global _anti_goal_generator
    with _generator_lock:
        _anti_goal_generator = AntiGoalGenerator()
        return _anti_goal_generator

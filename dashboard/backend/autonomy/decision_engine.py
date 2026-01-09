"""
Decision Engine - Multi-criteria decision making for PM-1000 autonomous operation.

Evaluates opportunities using multiple criteria, generates execution plans,
assesses risks, and selects optimal actions through intelligent ranking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categorization for decisions."""
    MINIMAL = 0      # No significant risk
    LOW = 1          # Minor potential issues
    MODERATE = 2     # Requires attention
    HIGH = 3         # Significant risk, needs approval
    CRITICAL = 4     # Major risk, should avoid


class DecisionConfidence(Enum):
    """Confidence level in the decision."""
    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class ExecutionStrategy(Enum):
    """Strategy for executing a decision."""
    IMMEDIATE = "immediate"           # Execute right away
    BATCHED = "batched"              # Group with similar tasks
    SCHEDULED = "scheduled"          # Execute at specific time
    CONDITIONAL = "conditional"      # Execute when conditions met
    ITERATIVE = "iterative"          # Execute in small steps
    PARALLEL = "parallel"            # Execute alongside others


@dataclass
class ExecutionStep:
    """Single step in an execution plan."""
    step_number: int
    action: str
    description: str
    estimated_duration_seconds: float
    dependencies: List[int] = field(default_factory=list)
    rollback_action: Optional[str] = None
    validation_check: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete plan for executing a decision."""
    steps: List[ExecutionStep]
    strategy: ExecutionStrategy
    total_estimated_duration: float
    parallelizable_steps: List[List[int]] = field(default_factory=list)
    checkpoint_steps: List[int] = field(default_factory=list)
    rollback_plan: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "description": s.description,
                    "estimated_duration_seconds": s.estimated_duration_seconds,
                    "dependencies": s.dependencies,
                    "rollback_action": s.rollback_action,
                    "validation_check": s.validation_check
                }
                for s in self.steps
            ],
            "strategy": self.strategy.value,
            "total_estimated_duration": self.total_estimated_duration,
            "parallelizable_steps": self.parallelizable_steps,
            "checkpoint_steps": self.checkpoint_steps,
            "rollback_plan": self.rollback_plan
        }


@dataclass
class RiskAssessment:
    """Risk assessment for a decision."""
    level: RiskLevel
    factors: List[str]
    mitigations: List[str]
    worst_case: str
    recovery_plan: Optional[str] = None
    requires_approval: bool = False
    approval_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "factors": self.factors,
            "mitigations": self.mitigations,
            "worst_case": self.worst_case,
            "recovery_plan": self.recovery_plan,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason
        }


@dataclass
class CriteriaScore:
    """Individual criterion score with explanation."""
    criterion: str
    score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float
    explanation: str


@dataclass
class Decision:
    """Complete decision with scoring, reasoning, and planning."""
    opportunity_id: str
    opportunity_type: str
    opportunity_data: Dict[str, Any]

    # Scoring
    total_score: float
    criteria_scores: List[CriteriaScore]
    confidence: DecisionConfidence

    # Reasoning
    reasoning: str
    key_factors: List[str]
    concerns: List[str]

    # Planning
    execution_plan: ExecutionPlan
    risk_assessment: RiskAssessment

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "opportunity_type": self.opportunity_type,
            "opportunity_data": self.opportunity_data,
            "total_score": self.total_score,
            "criteria_scores": [
                {
                    "criterion": cs.criterion,
                    "score": cs.score,
                    "weight": cs.weight,
                    "weighted_score": cs.weighted_score,
                    "explanation": cs.explanation
                }
                for cs in self.criteria_scores
            ],
            "confidence": self.confidence.name,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "concerns": self.concerns,
            "execution_plan": self.execution_plan.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags
        }


class CriteriaEvaluator:
    """Evaluates opportunities against specific criteria."""

    def __init__(self):
        self.criteria_weights = {
            "impact": 0.25,
            "urgency": 0.20,
            "effort": 0.15,
            "risk": 0.15,
            "alignment": 0.10,
            "dependencies": 0.10,
            "learning_value": 0.05
        }

    def evaluate_impact(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate the potential impact of addressing this opportunity."""
        impact_indicators = {
            "security": 1.0,
            "performance": 0.8,
            "reliability": 0.8,
            "maintainability": 0.6,
            "documentation": 0.4,
            "style": 0.2
        }

        opp_type = opportunity.get("type", "").lower()
        priority = opportunity.get("priority", "medium").lower()

        # Base score from type
        base_score = 0.5
        for indicator, score in impact_indicators.items():
            if indicator in opp_type:
                base_score = max(base_score, score)
                break

        # Adjust for priority
        priority_multipliers = {
            "critical": 1.0,
            "high": 0.85,
            "medium": 0.65,
            "low": 0.45
        }
        multiplier = priority_multipliers.get(priority, 0.5)
        final_score = base_score * multiplier

        explanation = f"Impact based on type '{opp_type}' (base: {base_score:.2f}) with {priority} priority"
        return (final_score, explanation)

    def evaluate_urgency(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate how urgent addressing this opportunity is."""
        urgency = opportunity.get("urgency", "medium").lower()
        age_hours = opportunity.get("age_hours", 0)

        urgency_scores = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.25
        }

        base_score = urgency_scores.get(urgency, 0.5)

        # Age decay - older items may need attention
        if age_hours > 168:  # > 1 week
            age_bonus = min(0.2, age_hours / 1000)
            base_score = min(1.0, base_score + age_bonus)

        explanation = f"Urgency '{urgency}' (score: {base_score:.2f})"
        if age_hours > 168:
            explanation += f", aged {age_hours:.0f} hours"

        return (base_score, explanation)

    def evaluate_effort(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate the effort required (higher score = less effort)."""
        complexity = opportunity.get("complexity", "moderate").lower()
        estimated_minutes = opportunity.get("estimated_minutes", 30)

        # Invert complexity (less complex = higher score)
        complexity_scores = {
            "trivial": 1.0,
            "simple": 0.85,
            "moderate": 0.6,
            "complex": 0.35,
            "expert": 0.15
        }

        base_score = complexity_scores.get(complexity, 0.5)

        # Time factor (shorter = better)
        if estimated_minutes <= 5:
            time_factor = 1.0
        elif estimated_minutes <= 15:
            time_factor = 0.9
        elif estimated_minutes <= 30:
            time_factor = 0.7
        elif estimated_minutes <= 60:
            time_factor = 0.5
        else:
            time_factor = max(0.1, 1.0 - (estimated_minutes / 200))

        final_score = (base_score + time_factor) / 2
        explanation = f"Complexity '{complexity}' ({base_score:.2f}), estimated {estimated_minutes} min ({time_factor:.2f})"

        return (final_score, explanation)

    def evaluate_risk(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate the risk of addressing this opportunity (higher = safer)."""
        risk_level = opportunity.get("risk_level", "low").lower()
        requires_approval = opportunity.get("requires_approval", False)
        is_reversible = opportunity.get("is_reversible", True)

        risk_scores = {
            "none": 1.0,
            "minimal": 0.9,
            "low": 0.75,
            "moderate": 0.5,
            "high": 0.25,
            "critical": 0.1
        }

        base_score = risk_scores.get(risk_level, 0.5)

        if requires_approval:
            base_score *= 0.8

        if not is_reversible:
            base_score *= 0.7

        explanation = f"Risk level '{risk_level}' (base: {base_score:.2f})"
        if requires_approval:
            explanation += ", needs approval"
        if not is_reversible:
            explanation += ", not reversible"

        return (base_score, explanation)

    def evaluate_alignment(self, opportunity: Dict[str, Any],
                          current_goals: List[str] = None) -> Tuple[float, str]:
        """Evaluate alignment with current goals and priorities."""
        tags = opportunity.get("tags", [])
        goal_related = opportunity.get("goal_related", False)

        base_score = 0.3  # Default alignment

        if goal_related:
            base_score = 0.8

        if current_goals:
            # Check tag overlap with goals
            overlap = len(set(tags) & set(current_goals))
            if overlap > 0:
                base_score = min(1.0, base_score + (overlap * 0.15))

        # Bonus for maintenance/improvement tasks
        maintenance_tags = {"cleanup", "refactor", "optimize", "fix", "improve"}
        if set(tags) & maintenance_tags:
            base_score = min(1.0, base_score + 0.1)

        explanation = f"Alignment score: {base_score:.2f}"
        if goal_related:
            explanation += " (goal-related)"

        return (base_score, explanation)

    def evaluate_dependencies(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate dependency complexity (fewer = better)."""
        dependencies = opportunity.get("dependencies", [])
        blockers = opportunity.get("blockers", [])

        dep_count = len(dependencies) + (len(blockers) * 2)  # Blockers count double

        if dep_count == 0:
            score = 1.0
        elif dep_count <= 2:
            score = 0.8
        elif dep_count <= 5:
            score = 0.5
        else:
            score = max(0.1, 1.0 - (dep_count / 10))

        explanation = f"{len(dependencies)} dependencies, {len(blockers)} blockers"
        return (score, explanation)

    def evaluate_learning_value(self, opportunity: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate the learning/improvement value of this opportunity."""
        is_novel = opportunity.get("is_novel", False)
        improvement_area = opportunity.get("improvement_area", None)

        base_score = 0.3

        if is_novel:
            base_score += 0.4

        if improvement_area:
            improvement_areas = {
                "architecture": 0.3,
                "testing": 0.25,
                "performance": 0.25,
                "security": 0.3,
                "documentation": 0.15
            }
            base_score += improvement_areas.get(improvement_area, 0.1)

        score = min(1.0, base_score)
        explanation = f"Learning value: {score:.2f}"
        if is_novel:
            explanation += " (novel task)"

        return (score, explanation)

    def evaluate_all(self, opportunity: Dict[str, Any],
                     current_goals: List[str] = None) -> List[CriteriaScore]:
        """Evaluate opportunity against all criteria."""
        evaluations = [
            ("impact", self.evaluate_impact(opportunity)),
            ("urgency", self.evaluate_urgency(opportunity)),
            ("effort", self.evaluate_effort(opportunity)),
            ("risk", self.evaluate_risk(opportunity)),
            ("alignment", self.evaluate_alignment(opportunity, current_goals)),
            ("dependencies", self.evaluate_dependencies(opportunity)),
            ("learning_value", self.evaluate_learning_value(opportunity))
        ]

        scores = []
        for criterion, (score, explanation) in evaluations:
            weight = self.criteria_weights.get(criterion, 0.1)
            scores.append(CriteriaScore(
                criterion=criterion,
                score=score,
                weight=weight,
                weighted_score=score * weight,
                explanation=explanation
            ))

        return scores


class ExecutionPlanner:
    """Generates execution plans for decisions."""

    def __init__(self):
        self.default_step_duration = 30.0  # seconds

    def create_plan(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Create an execution plan for an opportunity."""
        opp_type = opportunity.get("type", "unknown")

        # Route to specific planner based on type
        if "todo" in opp_type.lower():
            return self._plan_todo_fix(opportunity)
        elif "test" in opp_type.lower():
            return self._plan_test_task(opportunity)
        elif "doc" in opp_type.lower():
            return self._plan_documentation(opportunity)
        elif "security" in opp_type.lower():
            return self._plan_security_fix(opportunity)
        elif "refactor" in opp_type.lower():
            return self._plan_refactoring(opportunity)
        else:
            return self._plan_generic(opportunity)

    def _plan_todo_fix(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Plan for fixing a TODO comment."""
        file_path = opportunity.get("file_path", "unknown")
        line = opportunity.get("line", 0)

        steps = [
            ExecutionStep(
                step_number=1,
                action="read_file",
                description=f"Read context around TODO at {file_path}:{line}",
                estimated_duration_seconds=5.0,
                validation_check="file_exists"
            ),
            ExecutionStep(
                step_number=2,
                action="analyze_todo",
                description="Analyze what the TODO requires",
                estimated_duration_seconds=10.0,
                dependencies=[1]
            ),
            ExecutionStep(
                step_number=3,
                action="implement_fix",
                description="Implement the required functionality",
                estimated_duration_seconds=60.0,
                dependencies=[2],
                rollback_action="git_restore"
            ),
            ExecutionStep(
                step_number=4,
                action="remove_todo",
                description="Remove or update the TODO comment",
                estimated_duration_seconds=5.0,
                dependencies=[3]
            ),
            ExecutionStep(
                step_number=5,
                action="validate",
                description="Run tests to validate changes",
                estimated_duration_seconds=30.0,
                dependencies=[4],
                validation_check="tests_pass"
            )
        ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.ITERATIVE,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            checkpoint_steps=[3, 5],
            rollback_plan="git restore --staged . && git checkout -- ."
        )

    def _plan_test_task(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Plan for test-related tasks."""
        task_subtype = opportunity.get("subtype", "add_test")

        if task_subtype == "add_coverage":
            steps = [
                ExecutionStep(
                    step_number=1,
                    action="identify_uncovered",
                    description="Identify uncovered code paths",
                    estimated_duration_seconds=15.0
                ),
                ExecutionStep(
                    step_number=2,
                    action="write_test",
                    description="Write test for uncovered code",
                    estimated_duration_seconds=45.0,
                    dependencies=[1]
                ),
                ExecutionStep(
                    step_number=3,
                    action="run_tests",
                    description="Run test suite",
                    estimated_duration_seconds=30.0,
                    dependencies=[2],
                    validation_check="tests_pass"
                )
            ]
        else:
            steps = [
                ExecutionStep(
                    step_number=1,
                    action="analyze_test_need",
                    description="Analyze what needs testing",
                    estimated_duration_seconds=10.0
                ),
                ExecutionStep(
                    step_number=2,
                    action="write_test",
                    description="Write appropriate test",
                    estimated_duration_seconds=30.0,
                    dependencies=[1]
                ),
                ExecutionStep(
                    step_number=3,
                    action="validate",
                    description="Run and verify test",
                    estimated_duration_seconds=20.0,
                    dependencies=[2],
                    validation_check="tests_pass"
                )
            ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.IMMEDIATE,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            checkpoint_steps=[3],
            rollback_plan="git checkout -- tests/"
        )

    def _plan_documentation(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Plan for documentation tasks."""
        steps = [
            ExecutionStep(
                step_number=1,
                action="read_code",
                description="Read code to document",
                estimated_duration_seconds=15.0
            ),
            ExecutionStep(
                step_number=2,
                action="analyze_purpose",
                description="Analyze code purpose and behavior",
                estimated_duration_seconds=20.0,
                dependencies=[1]
            ),
            ExecutionStep(
                step_number=3,
                action="write_docs",
                description="Write documentation/docstrings",
                estimated_duration_seconds=30.0,
                dependencies=[2]
            ),
            ExecutionStep(
                step_number=4,
                action="validate_docs",
                description="Validate documentation accuracy",
                estimated_duration_seconds=10.0,
                dependencies=[3]
            )
        ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.BATCHED,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            parallelizable_steps=[[3, 4]],  # Can validate while writing
            rollback_plan="git checkout -- *.py *.md"
        )

    def _plan_security_fix(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Plan for security-related fixes."""
        steps = [
            ExecutionStep(
                step_number=1,
                action="assess_vulnerability",
                description="Assess the security vulnerability",
                estimated_duration_seconds=20.0
            ),
            ExecutionStep(
                step_number=2,
                action="research_fix",
                description="Research appropriate fix",
                estimated_duration_seconds=30.0,
                dependencies=[1]
            ),
            ExecutionStep(
                step_number=3,
                action="implement_fix",
                description="Implement security fix",
                estimated_duration_seconds=45.0,
                dependencies=[2],
                rollback_action="git_restore"
            ),
            ExecutionStep(
                step_number=4,
                action="security_test",
                description="Test fix for security",
                estimated_duration_seconds=30.0,
                dependencies=[3],
                validation_check="security_scan_pass"
            ),
            ExecutionStep(
                step_number=5,
                action="regression_test",
                description="Run regression tests",
                estimated_duration_seconds=45.0,
                dependencies=[4],
                validation_check="tests_pass"
            )
        ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.ITERATIVE,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            checkpoint_steps=[3, 4, 5],
            rollback_plan="git revert HEAD"
        )

    def _plan_refactoring(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Plan for refactoring tasks."""
        scope = opportunity.get("scope", "function")

        if scope == "file":
            duration_multiplier = 2.0
        elif scope == "module":
            duration_multiplier = 3.0
        else:
            duration_multiplier = 1.0

        steps = [
            ExecutionStep(
                step_number=1,
                action="analyze_code",
                description="Analyze code structure",
                estimated_duration_seconds=20.0 * duration_multiplier
            ),
            ExecutionStep(
                step_number=2,
                action="plan_refactor",
                description="Plan refactoring approach",
                estimated_duration_seconds=15.0
            ),
            ExecutionStep(
                step_number=3,
                action="refactor",
                description="Perform refactoring",
                estimated_duration_seconds=60.0 * duration_multiplier,
                dependencies=[1, 2],
                rollback_action="git_restore"
            ),
            ExecutionStep(
                step_number=4,
                action="test",
                description="Run tests",
                estimated_duration_seconds=30.0,
                dependencies=[3],
                validation_check="tests_pass"
            )
        ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.ITERATIVE,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            checkpoint_steps=[3, 4],
            rollback_plan="git checkout -- ."
        )

    def _plan_generic(self, opportunity: Dict[str, Any]) -> ExecutionPlan:
        """Generic execution plan."""
        steps = [
            ExecutionStep(
                step_number=1,
                action="analyze",
                description="Analyze the opportunity",
                estimated_duration_seconds=15.0
            ),
            ExecutionStep(
                step_number=2,
                action="implement",
                description="Implement solution",
                estimated_duration_seconds=45.0,
                dependencies=[1],
                rollback_action="git_restore"
            ),
            ExecutionStep(
                step_number=3,
                action="validate",
                description="Validate changes",
                estimated_duration_seconds=20.0,
                dependencies=[2]
            )
        ]

        return ExecutionPlan(
            steps=steps,
            strategy=ExecutionStrategy.IMMEDIATE,
            total_estimated_duration=sum(s.estimated_duration_seconds for s in steps),
            checkpoint_steps=[2, 3],
            rollback_plan="git checkout -- ."
        )


class RiskAnalyzer:
    """Analyzes risks associated with decisions."""

    def __init__(self):
        self.high_risk_patterns = {
            "delete": RiskLevel.HIGH,
            "remove": RiskLevel.MODERATE,
            "security": RiskLevel.HIGH,
            "database": RiskLevel.HIGH,
            "migration": RiskLevel.HIGH,
            "production": RiskLevel.CRITICAL,
            "credentials": RiskLevel.CRITICAL,
            "config": RiskLevel.MODERATE,
            "api": RiskLevel.MODERATE
        }

        self.low_risk_patterns = {
            "documentation": RiskLevel.MINIMAL,
            "comment": RiskLevel.MINIMAL,
            "test": RiskLevel.LOW,
            "style": RiskLevel.MINIMAL,
            "format": RiskLevel.MINIMAL
        }

    def assess(self, opportunity: Dict[str, Any],
               execution_plan: ExecutionPlan) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        factors = []
        mitigations = []

        # Determine base risk level
        opp_type = opportunity.get("type", "").lower()
        opp_desc = opportunity.get("description", "").lower()
        combined_text = f"{opp_type} {opp_desc}"

        max_risk = RiskLevel.MINIMAL

        # Check for high-risk patterns
        for pattern, risk in self.high_risk_patterns.items():
            if pattern in combined_text:
                if risk.value > max_risk.value:
                    max_risk = risk
                factors.append(f"Contains '{pattern}' pattern")

        # Check for low-risk patterns (can reduce risk)
        for pattern, risk in self.low_risk_patterns.items():
            if pattern in combined_text:
                factors.append(f"Low-risk '{pattern}' operation")

        # Assess execution plan risk
        if execution_plan.total_estimated_duration > 300:  # > 5 minutes
            factors.append("Long execution time increases risk")
            max_risk = max(max_risk, RiskLevel.MODERATE, key=lambda x: x.value)

        if not execution_plan.rollback_plan:
            factors.append("No rollback plan available")
            max_risk = max(max_risk, RiskLevel.MODERATE, key=lambda x: x.value)
        else:
            mitigations.append(f"Rollback available: {execution_plan.rollback_plan}")

        if execution_plan.checkpoint_steps:
            mitigations.append(f"Checkpoints at steps: {execution_plan.checkpoint_steps}")

        # Check reversibility
        is_reversible = opportunity.get("is_reversible", True)
        if not is_reversible:
            factors.append("Action is not reversible")
            max_risk = max(max_risk, RiskLevel.HIGH, key=lambda x: x.value)

        # File modification risk
        file_path = opportunity.get("file_path", "")
        if any(critical in file_path.lower() for critical in ["config", "settings", "env", ".yml", ".json"]):
            factors.append("Modifies configuration file")
            mitigations.append("Backup configuration before changes")

        # Generate worst case scenario
        worst_case = self._generate_worst_case(opportunity, max_risk)

        # Generate recovery plan
        recovery_plan = self._generate_recovery_plan(opportunity, execution_plan)

        # Determine if approval required
        requires_approval = max_risk.value >= RiskLevel.HIGH.value
        approval_reason = None
        if requires_approval:
            approval_reason = f"Risk level {max_risk.name} requires human approval"

        return RiskAssessment(
            level=max_risk,
            factors=factors if factors else ["No significant risk factors identified"],
            mitigations=mitigations if mitigations else ["Standard precautions apply"],
            worst_case=worst_case,
            recovery_plan=recovery_plan,
            requires_approval=requires_approval,
            approval_reason=approval_reason
        )

    def _generate_worst_case(self, opportunity: Dict[str, Any],
                            risk_level: RiskLevel) -> str:
        """Generate worst case scenario description."""
        opp_type = opportunity.get("type", "unknown")

        worst_cases = {
            RiskLevel.MINIMAL: "Minor formatting inconsistency",
            RiskLevel.LOW: "Test failure requiring quick fix",
            RiskLevel.MODERATE: "Feature regression requiring debugging",
            RiskLevel.HIGH: "Service disruption requiring rollback",
            RiskLevel.CRITICAL: "Data loss or security breach"
        }

        base_worst = worst_cases.get(risk_level, "Unknown issue")
        return f"{base_worst} from {opp_type} operation"

    def _generate_recovery_plan(self, opportunity: Dict[str, Any],
                               execution_plan: ExecutionPlan) -> str:
        """Generate recovery plan for worst case."""
        steps = []

        if execution_plan.rollback_plan:
            steps.append(f"1. Execute rollback: {execution_plan.rollback_plan}")

        steps.append("2. Run test suite to verify system state")
        steps.append("3. Review logs for any additional issues")
        steps.append("4. Document incident for future prevention")

        return " | ".join(steps)


class DecisionEngine:
    """
    Main decision engine for evaluating opportunities and making decisions.

    Uses multi-criteria evaluation, execution planning, and risk assessment
    to make intelligent decisions about which opportunities to pursue.
    """

    def __init__(self,
                 criteria_evaluator: Optional[CriteriaEvaluator] = None,
                 execution_planner: Optional[ExecutionPlanner] = None,
                 risk_analyzer: Optional[RiskAnalyzer] = None):
        self.criteria_evaluator = criteria_evaluator or CriteriaEvaluator()
        self.execution_planner = execution_planner or ExecutionPlanner()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()

        self.decision_history: List[Decision] = []
        self.current_goals: List[str] = []

        # Configuration
        self.min_score_threshold = 0.3
        self.max_concurrent_decisions = 5
        self.decision_expiry_hours = 4

    def set_current_goals(self, goals: List[str]):
        """Set current goals for alignment evaluation."""
        self.current_goals = goals

    def evaluate_opportunity(self, opportunity: Dict[str, Any]) -> Decision:
        """
        Evaluate a single opportunity and create a decision.

        Args:
            opportunity: Dictionary containing opportunity data

        Returns:
            Decision object with complete evaluation
        """
        # Extract opportunity metadata
        opp_id = opportunity.get("id", f"opp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        opp_type = opportunity.get("type", "unknown")

        # Evaluate against all criteria
        criteria_scores = self.criteria_evaluator.evaluate_all(
            opportunity,
            self.current_goals
        )

        # Calculate total score
        total_score = sum(cs.weighted_score for cs in criteria_scores)

        # Determine confidence based on score distribution
        score_variance = self._calculate_variance([cs.score for cs in criteria_scores])
        confidence = self._determine_confidence(total_score, score_variance)

        # Generate reasoning
        reasoning, key_factors, concerns = self._generate_reasoning(
            opportunity, criteria_scores, total_score
        )

        # Create execution plan
        execution_plan = self.execution_planner.create_plan(opportunity)

        # Assess risks
        risk_assessment = self.risk_analyzer.assess(opportunity, execution_plan)

        # Generate tags
        tags = self._generate_tags(opportunity, criteria_scores)

        # Create decision
        decision = Decision(
            opportunity_id=opp_id,
            opportunity_type=opp_type,
            opportunity_data=opportunity,
            total_score=total_score,
            criteria_scores=criteria_scores,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            concerns=concerns,
            execution_plan=execution_plan,
            risk_assessment=risk_assessment,
            expires_at=datetime.now() + timedelta(hours=self.decision_expiry_hours),
            tags=tags
        )

        # Track in history
        self.decision_history.append(decision)

        return decision

    def rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Decision]:
        """
        Evaluate and rank multiple opportunities.

        Args:
            opportunities: List of opportunity dictionaries

        Returns:
            List of Decision objects sorted by score (highest first)
        """
        decisions = []

        for opp in opportunities:
            try:
                decision = self.evaluate_opportunity(opp)
                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Failed to evaluate opportunity: {e}")
                continue

        # Sort by total score descending
        decisions.sort(key=lambda d: d.total_score, reverse=True)

        return decisions

    def select_best_action(self, decisions: List[Decision]) -> Optional[Decision]:
        """
        Select the best action from ranked decisions.

        Args:
            decisions: List of evaluated decisions

        Returns:
            Best decision to act on, or None if no suitable action
        """
        if not decisions:
            return None

        for decision in decisions:
            # Skip if below threshold
            if decision.total_score < self.min_score_threshold:
                logger.debug(f"Skipping {decision.opportunity_id}: score {decision.total_score:.2f} below threshold")
                continue

            # Skip if requires approval and not approved
            if decision.risk_assessment.requires_approval:
                logger.info(f"Skipping {decision.opportunity_id}: requires approval")
                continue

            # Skip if expired
            if decision.expires_at and datetime.now() > decision.expires_at:
                logger.debug(f"Skipping {decision.opportunity_id}: decision expired")
                continue

            # This is our best actionable decision
            return decision

        return None

    def select_batch(self, decisions: List[Decision],
                     max_batch_size: int = 3) -> List[Decision]:
        """
        Select a batch of compatible decisions for parallel execution.

        Args:
            decisions: List of evaluated decisions
            max_batch_size: Maximum number of decisions to batch

        Returns:
            List of compatible decisions for parallel execution
        """
        batch = []
        used_files = set()

        for decision in decisions:
            if len(batch) >= max_batch_size:
                break

            if decision.total_score < self.min_score_threshold:
                continue

            if decision.risk_assessment.requires_approval:
                continue

            # Check for file conflicts
            file_path = decision.opportunity_data.get("file_path", "")
            if file_path and file_path in used_files:
                continue

            # Can batch documentation with other tasks
            if decision.opportunity_type in ["documentation", "test"]:
                batch.append(decision)
                if file_path:
                    used_files.add(file_path)
            elif len(batch) == 0 or all(d.opportunity_type in ["documentation", "test"] for d in batch):
                batch.append(decision)
                if file_path:
                    used_files.add(file_path)

        return batch

    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    def _determine_confidence(self, total_score: float,
                            score_variance: float) -> DecisionConfidence:
        """Determine confidence level based on score and variance."""
        # High score with low variance = high confidence
        if total_score > 0.7 and score_variance < 0.05:
            return DecisionConfidence.VERY_HIGH
        elif total_score > 0.6 and score_variance < 0.1:
            return DecisionConfidence.HIGH
        elif total_score > 0.4 and score_variance < 0.15:
            return DecisionConfidence.MODERATE
        elif total_score > 0.3:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW

    def _generate_reasoning(self, opportunity: Dict[str, Any],
                           criteria_scores: List[CriteriaScore],
                           total_score: float) -> Tuple[str, List[str], List[str]]:
        """Generate reasoning explanation."""
        # Find top factors
        sorted_scores = sorted(criteria_scores, key=lambda cs: cs.weighted_score, reverse=True)
        key_factors = []
        concerns = []

        for cs in sorted_scores[:3]:
            if cs.score > 0.6:
                key_factors.append(f"{cs.criterion.capitalize()}: {cs.explanation}")

        for cs in sorted_scores:
            if cs.score < 0.4:
                concerns.append(f"{cs.criterion.capitalize()}: {cs.explanation}")

        # Build reasoning paragraph
        opp_type = opportunity.get("type", "unknown")
        reasoning_parts = [
            f"This {opp_type} opportunity scored {total_score:.2f} overall.",
        ]

        if key_factors:
            reasoning_parts.append(f"Strong factors: {', '.join(key_factors[:2])}.")

        if concerns:
            reasoning_parts.append(f"Concerns: {concerns[0]}.")

        reasoning = " ".join(reasoning_parts)

        return reasoning, key_factors, concerns

    def _generate_tags(self, opportunity: Dict[str, Any],
                      criteria_scores: List[CriteriaScore]) -> List[str]:
        """Generate tags for the decision."""
        tags = []

        # Type-based tags
        opp_type = opportunity.get("type", "").lower()
        if "security" in opp_type:
            tags.append("security")
        if "test" in opp_type:
            tags.append("testing")
        if "doc" in opp_type:
            tags.append("documentation")
        if "todo" in opp_type:
            tags.append("todo")
        if "refactor" in opp_type:
            tags.append("refactoring")

        # Score-based tags
        impact_score = next((cs.score for cs in criteria_scores if cs.criterion == "impact"), 0)
        if impact_score > 0.8:
            tags.append("high-impact")

        effort_score = next((cs.score for cs in criteria_scores if cs.criterion == "effort"), 0)
        if effort_score > 0.8:
            tags.append("quick-win")

        return tags

    def get_metrics(self) -> Dict[str, Any]:
        """Get decision engine metrics."""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "avg_score": 0.0,
                "by_confidence": {},
                "by_risk_level": {}
            }

        total = len(self.decision_history)
        avg_score = sum(d.total_score for d in self.decision_history) / total

        by_confidence = {}
        by_risk = {}

        for d in self.decision_history:
            conf = d.confidence.name
            by_confidence[conf] = by_confidence.get(conf, 0) + 1

            risk = d.risk_assessment.level.name
            by_risk[risk] = by_risk.get(risk, 0) + 1

        return {
            "total_decisions": total,
            "avg_score": avg_score,
            "by_confidence": by_confidence,
            "by_risk_level": by_risk,
            "current_goals": self.current_goals
        }


# Convenience function
def create_decision_engine(**kwargs) -> DecisionEngine:
    """Create a configured decision engine."""
    return DecisionEngine(**kwargs)

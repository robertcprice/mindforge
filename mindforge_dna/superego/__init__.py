"""
Superego Layer - IMMUTABLE ethical constraints and safety systems.

The Superego layer provides three core protective systems:
1. Values: Immutable ethical constraints (benevolence, honesty, humility, safety)
2. Safety: Runtime protection against dangerous operations
3. KVRM: Knowledge verification and fact grounding

These systems work together to ensure the agent operates within safe,
ethical boundaries that cannot be modified at runtime.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .values import (
    ValueType,
    ValueViolation,
    ValuesChecker,
    get_values_checker,
)
from .safety import (
    SafetyCheckResult,
    SafetyChecker,
    get_safety_checker,
)
from .kvrm import (
    ClaimType,
    GroundingResult,
    KVRMRouter,
    get_kvrm_router,
)

logger = logging.getLogger(__name__)


@dataclass
class SuperegoCheckResult:
    """
    Combined result from all Superego checks.

    Attributes:
        is_approved: Overall approval status (all checks passed)
        values_passed: Whether values check passed
        safety_passed: Whether safety check passed
        grounding_results: List of fact grounding results
        value_violations: Any value violations found
        safety_reason: Reason if safety check failed
        recommendation: Suggested action if blocked
    """
    is_approved: bool
    values_passed: bool
    safety_passed: bool
    grounding_results: List[GroundingResult]
    value_violations: List[ValueViolation]
    safety_reason: Optional[str] = None
    recommendation: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result directly in conditionals."""
        return self.is_approved

    def get_summary(self) -> str:
        """Generate human-readable summary of check results."""
        lines = ["Superego Check Summary:"]

        if self.is_approved:
            lines.append("  STATUS: APPROVED ")
        else:
            lines.append("  STATUS: BLOCKED ")

        lines.append(f"  Values: {'PASS' if self.values_passed else 'FAIL'}")
        lines.append(f"  Safety: {'PASS' if self.safety_passed else 'FAIL'}")

        if self.value_violations:
            lines.append(f"  Value Violations: {len(self.value_violations)}")
            for v in self.value_violations[:3]:  # Show first 3
                lines.append(f"    - [{v.severity}] {v.description}")

        if self.safety_reason:
            lines.append(f"  Safety Issue: {self.safety_reason}")

        if self.grounding_results:
            verified = sum(1 for r in self.grounding_results if r.is_verified)
            lines.append(f"  Grounding: {verified}/{len(self.grounding_results)} verified")

        if self.recommendation:
            lines.append(f"  Recommendation: {self.recommendation}")

        return "\n".join(lines)


class SuperegoLayer:
    """
    Unified Superego layer combining values, safety, and knowledge grounding.

    This layer provides comprehensive ethical and safety checks before
    actions are executed. It's designed to be fast, conservative, and
    immutable.
    """

    def __init__(self):
        """Initialize all Superego subsystems."""
        self.values_checker = get_values_checker()
        self.safety_checker = get_safety_checker()
        self.kvrm_router = get_kvrm_router()

        logger.info("SuperegoLayer initialized with all subsystems")

    def check_action(
        self,
        action_description: str,
        tool_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SuperegoCheckResult:
        """
        Comprehensive check before executing an action.

        Args:
            action_description: Text description of the action
            tool_name: Optional tool being invoked
            parameters: Optional tool parameters

        Returns:
            SuperegoCheckResult with comprehensive approval status
        """
        # 1. Values check
        values_passed, value_violations = self.values_checker.check_all(
            action_description
        )

        # 2. Safety check
        safety_result = SafetyCheckResult(is_safe=True)

        if tool_name:
            safety_result = self.safety_checker.check_tool_call(tool_name, parameters)

        if parameters:
            # Check specific parameter types
            if "command" in parameters:
                safety_result = self.safety_checker.check_command(parameters["command"])
            elif "file_path" in parameters:
                operation = "write" if tool_name in ("Write", "Edit") else "read"
                safety_result = self.safety_checker.check_path(
                    parameters["file_path"],
                    operation
                )

        # 3. Ground claims in action description (informational only)
        grounding_results = self.kvrm_router.ground_thought(action_description)

        # Determine overall approval
        is_approved = values_passed and safety_result.is_safe

        # Compile recommendation
        recommendation = None
        if not is_approved:
            if not values_passed:
                recommendation = "Action violates core values - please revise"
            elif not safety_result.is_safe:
                recommendation = safety_result.recommendation

        result = SuperegoCheckResult(
            is_approved=is_approved,
            values_passed=values_passed,
            safety_passed=safety_result.is_safe,
            grounding_results=grounding_results,
            value_violations=value_violations,
            safety_reason=safety_result.reason,
            recommendation=recommendation
        )

        if not is_approved:
            logger.warning(f"Action BLOCKED by Superego: {action_description[:100]}")
            logger.warning(result.get_summary())
        else:
            logger.debug(f"Action APPROVED by Superego: {action_description[:100]}")

        return result

    def check_thought(self, thought: str) -> SuperegoCheckResult:
        """
        Check and ground a thought/reasoning step.

        Args:
            thought: Thought content to validate and ground

        Returns:
            SuperegoCheckResult with grounding information
        """
        # 1. Values check (thoughts should be honest and humble)
        values_passed, value_violations = self.values_checker.check_all(thought)

        # 2. Ground all claims
        grounding_results = self.kvrm_router.ground_thought(thought)

        # 3. Safety is less critical for thoughts (not executing)
        safety_result = SafetyCheckResult(is_safe=True)

        # Thoughts are approved unless they have critical value violations
        critical_violations = [
            v for v in value_violations
            if v.severity == "critical"
        ]
        is_approved = len(critical_violations) == 0

        result = SuperegoCheckResult(
            is_approved=is_approved,
            values_passed=values_passed,
            safety_passed=True,  # Not applicable to thoughts
            grounding_results=grounding_results,
            value_violations=value_violations,
            recommendation="Revise thought to align with core values" if not is_approved else None
        )

        logger.debug(f"Thought checked: {len(grounding_results)} claims grounded")
        return result

    def verify_fact(
        self,
        claim: str,
        domain: str,
        source: str,
        confidence: float,
        evidence: Optional[str] = None
    ) -> str:
        """
        Store a verified fact in KVRM.

        Args:
            claim: Factual claim
            domain: Knowledge domain
            source: Verification source
            confidence: Confidence score 0.0-1.0
            evidence: Supporting evidence

        Returns:
            KVRM key for the stored fact
        """
        return self.kvrm_router.store_fact(
            claim=claim,
            domain=domain,
            source=source,
            confidence=confidence,
            evidence=evidence
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics from all subsystems.

        Returns:
            Dictionary of statistics
        """
        return {
            "safety": self.safety_checker.get_usage_stats(),
            "values": {
                "checker_initialized": self.values_checker is not None,
                "core_values": ["benevolence", "honesty", "humility", "safety"]
            },
            "kvrm": {
                "database": str(self.kvrm_router.db_path),
                "initialized": self.kvrm_router is not None
            }
        }


# Module-level singleton
_superego_layer_instance: SuperegoLayer | None = None


def get_superego_layer() -> SuperegoLayer:
    """
    Get singleton instance of SuperegoLayer.

    Returns:
        Shared SuperegoLayer instance
    """
    global _superego_layer_instance
    if _superego_layer_instance is None:
        _superego_layer_instance = SuperegoLayer()
    return _superego_layer_instance


# Convenience exports
__all__ = [
    # Main layer
    "SuperegoLayer",
    "SuperegoCheckResult",
    "get_superego_layer",

    # Values subsystem
    "ValueType",
    "ValueViolation",
    "ValuesChecker",
    "get_values_checker",

    # Safety subsystem
    "SafetyCheckResult",
    "SafetyChecker",
    "get_safety_checker",

    # KVRM subsystem
    "ClaimType",
    "GroundingResult",
    "KVRMRouter",
    "get_kvrm_router",
]

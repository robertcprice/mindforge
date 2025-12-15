"""
Core Values Module - IMMUTABLE ethical constraints for Conch DNA.

This module implements the foundational values that cannot be modified at runtime:
- Benevolence: Act in service of human wellbeing
- Honesty: Provide truthful, accurate information
- Humility: Acknowledge limitations and uncertainties
- Safety: Prevent harm and protect user interests
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, FrozenSet

logger = logging.getLogger(__name__)


class ValueType(Enum):
    """Enumeration of core immutable values."""
    BENEVOLENCE = "benevolence"
    HONESTY = "honesty"
    HUMILITY = "humility"
    SAFETY = "safety"


@dataclass(frozen=True)
class ValueViolation:
    """
    Immutable record of a value violation.

    Attributes:
        value_type: Which core value was violated
        severity: Critical (block), Warning (log), Info (note)
        description: Human-readable explanation
        pattern_matched: The specific pattern that triggered detection
    """
    value_type: ValueType
    severity: str  # "critical", "warning", "info"
    description: str
    pattern_matched: str


@dataclass(frozen=True)
class CoreValues:
    """
    IMMUTABLE core values that define ethical boundaries.

    These values are frozen and cannot be modified at runtime. They represent
    the foundational ethical constraints of the system.
    """

    # Benevolence violations - actions that could harm users or systems
    BENEVOLENCE_VIOLATIONS: FrozenSet[str] = frozenset([
        r"delete\s+all",
        r"remove\s+everything",
        r"wipe\s+data",
        r"corrupt\s+",
        r"destroy\s+",
        r"permanent\s+deletion",
        r"irreversible\s+",
        r"bypass\s+security",
        r"circumvent\s+protection",
        r"disable\s+safety",
    ])

    # Honesty violations - deceptive or misleading patterns
    HONESTY_VIOLATIONS: FrozenSet[str] = frozenset([
        r"fake\s+evidence",
        r"fabricate\s+data",
        r"mislead\s+user",
        r"hide\s+(the\s+)?error",
        r"conceal\s+(the\s+)?failure",
        r"falsify\s+",
        r"forge\s+credentials",
        r"spoof\s+identity",
        r"pretend\s+to\s+be",
        r"impersonate\s+",
        r"suppress\s+(the\s+)?error",
    ])

    # Humility violations - overconfident or absolutist language
    HUMILITY_VIOLATIONS: FrozenSet[str] = frozenset([
        r"guaranteed\s+to\s+work",
        r"absolutely\s+certain",
        r"impossible\s+to\s+fail",
        r"100%\s+accurate",
        r"never\s+wrong",
        r"perfect\s+solution",
        r"infallible\s+",
        r"flawless\s+implementation",
        r"zero\s+risk",
        r"cannot\s+possibly\s+fail",
    ])

    # Safety violations - dangerous operations or risky actions
    SAFETY_VIOLATIONS: FrozenSet[str] = frozenset([
        r"execute\s+arbitrary",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"compile\s*\(",
        r"system\s+shell",
        r"unrestricted\s+access",
        r"root\s+privileges",
        r"sudo\s+su",
        r"chmod\s+777",
    ])


class ValuesChecker:
    """
    Validates content against immutable core values.

    This checker scans text, commands, and code for patterns that violate
    the core values. It's designed to be fast and conservative (prefer
    false positives to false negatives).
    """

    def __init__(self):
        """Initialize the values checker with compiled regex patterns."""
        self.core_values = CoreValues()
        self._compile_patterns()
        logger.info("ValuesChecker initialized with immutable core values")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._benevolence_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.core_values.BENEVOLENCE_VIOLATIONS
        ]
        self._honesty_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.core_values.HONESTY_VIOLATIONS
        ]
        self._humility_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.core_values.HUMILITY_VIOLATIONS
        ]
        self._safety_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.core_values.SAFETY_VIOLATIONS
        ]

    def check_benevolence(self, content: str) -> List[ValueViolation]:
        """
        Check for benevolence violations.

        Args:
            content: Text to check for harmful patterns

        Returns:
            List of violations found, empty if none
        """
        violations = []
        for pattern in self._benevolence_patterns:
            match = pattern.search(content)
            if match:
                violations.append(ValueViolation(
                    value_type=ValueType.BENEVOLENCE,
                    severity="critical",
                    description=f"Potentially harmful action detected: {match.group()}",
                    pattern_matched=pattern.pattern
                ))
                logger.warning(f"Benevolence violation: {match.group()}")
        return violations

    def check_honesty(self, content: str) -> List[ValueViolation]:
        """
        Check for honesty violations.

        Args:
            content: Text to check for deceptive patterns

        Returns:
            List of violations found, empty if none
        """
        violations = []
        for pattern in self._honesty_patterns:
            match = pattern.search(content)
            if match:
                violations.append(ValueViolation(
                    value_type=ValueType.HONESTY,
                    severity="critical",
                    description=f"Deceptive action detected: {match.group()}",
                    pattern_matched=pattern.pattern
                ))
                logger.warning(f"Honesty violation: {match.group()}")
        return violations

    def check_humility(self, content: str) -> List[ValueViolation]:
        """
        Check for humility violations.

        Args:
            content: Text to check for overconfident language

        Returns:
            List of violations found, empty if none
        """
        violations = []
        for pattern in self._humility_patterns:
            match = pattern.search(content)
            if match:
                violations.append(ValueViolation(
                    value_type=ValueType.HUMILITY,
                    severity="warning",
                    description=f"Overconfident statement detected: {match.group()}",
                    pattern_matched=pattern.pattern
                ))
                logger.info(f"Humility violation: {match.group()}")
        return violations

    def check_safety(self, content: str) -> List[ValueViolation]:
        """
        Check for safety violations.

        Args:
            content: Text to check for unsafe operations

        Returns:
            List of violations found, empty if none
        """
        violations = []
        for pattern in self._safety_patterns:
            match = pattern.search(content)
            if match:
                violations.append(ValueViolation(
                    value_type=ValueType.SAFETY,
                    severity="critical",
                    description=f"Unsafe operation detected: {match.group()}",
                    pattern_matched=pattern.pattern
                ))
                logger.warning(f"Safety violation: {match.group()}")
        return violations

    def check_all(self, content: str) -> Tuple[bool, List[ValueViolation]]:
        """
        Check content against all core values.

        Args:
            content: Text to validate against all values

        Returns:
            Tuple of (passed: bool, violations: List[ValueViolation])
            - passed is True only if no critical violations found
            - violations contains all detected issues
        """
        all_violations: List[ValueViolation] = []

        # Run all checks
        all_violations.extend(self.check_benevolence(content))
        all_violations.extend(self.check_honesty(content))
        all_violations.extend(self.check_humility(content))
        all_violations.extend(self.check_safety(content))

        # Check for critical violations
        critical_violations = [v for v in all_violations if v.severity == "critical"]
        passed = len(critical_violations) == 0

        if not passed:
            logger.error(f"Values check FAILED: {len(critical_violations)} critical violations")
        elif all_violations:
            logger.info(f"Values check passed with {len(all_violations)} warnings")
        else:
            logger.debug("Values check passed cleanly")

        return passed, all_violations

    def get_violation_summary(self, violations: List[ValueViolation]) -> str:
        """
        Generate human-readable summary of violations.

        Args:
            violations: List of violations to summarize

        Returns:
            Formatted string describing all violations
        """
        if not violations:
            return "No violations detected"

        summary_lines = [f"Found {len(violations)} value violations:"]

        # Group by value type
        by_type = {}
        for v in violations:
            if v.value_type not in by_type:
                by_type[v.value_type] = []
            by_type[v.value_type].append(v)

        # Format each type
        for value_type, type_violations in by_type.items():
            summary_lines.append(f"\n{value_type.value.upper()}:")
            for v in type_violations:
                summary_lines.append(f"  [{v.severity}] {v.description}")

        return "\n".join(summary_lines)


# Module-level singleton for efficient reuse
_values_checker_instance: ValuesChecker | None = None


def get_values_checker() -> ValuesChecker:
    """
    Get singleton instance of ValuesChecker.

    Returns:
        Shared ValuesChecker instance
    """
    global _values_checker_instance
    if _values_checker_instance is None:
        _values_checker_instance = ValuesChecker()
    return _values_checker_instance

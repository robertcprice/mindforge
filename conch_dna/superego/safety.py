"""
Safety Checker Module - Runtime protection against dangerous operations.

This module implements safety checks for commands, file paths, and tool calls
to prevent accidental or malicious harm to the system.
"""

import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, FrozenSet, Optional

logger = logging.getLogger(__name__)


# IMMUTABLE blocked commands - operations that should never be executed
BLOCKED_COMMANDS: FrozenSet[str] = frozenset([
    # Destructive file operations
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $HOME",
    "sudo rm -rf",

    # Filesystem destruction
    "mkfs",
    "mkfs.ext4",
    "mkfs.xfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",

    # System modification
    "sudo su",
    "sudo -i",
    "chmod 777 /",
    "chown root:root /",

    # Fork bombs and resource exhaustion
    ":(){ :|:& };:",
    "while true; do fork",
    "/dev/null &",

    # Package manager risks
    "pip install --break-system-packages",
    "npm install -g --unsafe-perm",

    # Network attacks
    "nmap -A",
    "hping3",
    "metasploit",

    # Kernel/boot modification
    "grub-install",
    "update-grub",
    "kernel panic",
])


# IMMUTABLE blocked path patterns - sensitive locations
BLOCKED_PATH_PATTERNS: FrozenSet[str] = frozenset([
    # System directories
    r"^/etc/",
    r"^/boot/",
    r"^/sys/",
    r"^/proc/",
    r"^/dev/",
    r"^/root/",

    # Security credentials
    r"\.ssh/",
    r"\.gnupg/",
    r"\.aws/credentials",
    r"\.kube/config",

    # Secrets and keys
    r"\.env$",
    r"\.env\..*",
    r"\.pem$",
    r"\.key$",
    r"\.crt$",
    r"\.p12$",
    r"\.pfx$",
    r"credentials\.json",
    r"secrets\..*",
    r"private.*key",

    # Database files (write protection)
    r"\.db$",
    r"\.sqlite$",
    r"\.sqlite3$",
])


# Rate limiting configuration
MAX_ACTIONS_PER_MINUTE = 30
RATE_LIMIT_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class SafetyCheckResult:
    """
    Immutable result of a safety check.

    Attributes:
        is_safe: Whether the operation is safe to proceed
        reason: Explanation if unsafe, None if safe
        severity: "blocked", "warning", or "info"
        recommendation: Suggested alternative action if available
    """
    is_safe: bool
    reason: Optional[str] = None
    severity: str = "info"
    recommendation: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result directly in conditionals."""
        return self.is_safe


class SafetyChecker:
    """
    Runtime safety verification for system operations.

    Provides protection against:
    - Destructive commands
    - Unauthorized file access
    - Rate limit violations
    - Resource exhaustion
    """

    def __init__(self):
        """Initialize safety checker with rate limiting."""
        self._compile_patterns()
        self._action_timestamps: Deque[float] = deque(maxlen=MAX_ACTIONS_PER_MINUTE)
        self._tool_call_counts: Dict[str, int] = {}
        logger.info("SafetyChecker initialized with immutable rules")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient path matching."""
        self._blocked_path_regexes = [
            re.compile(pattern) for pattern in BLOCKED_PATH_PATTERNS
        ]

    def check_command(self, command: str) -> SafetyCheckResult:
        """
        Verify command safety before execution.

        Args:
            command: Shell command to validate

        Returns:
            SafetyCheckResult indicating safety status
        """
        # Normalize command for comparison
        normalized = command.strip().lower()

        # Check against blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in normalized:
                logger.error(f"BLOCKED COMMAND: {command}")
                return SafetyCheckResult(
                    is_safe=False,
                    reason=f"Command contains blocked pattern: {blocked}",
                    severity="blocked",
                    recommendation="This command is permanently blocked for safety"
                )

        # Check for dangerous patterns
        dangerous_patterns = [
            (r"rm\s+-rf\s+/", "Recursive delete from root directory"),
            (r"chmod\s+777", "Overly permissive file permissions"),
            (r">\s*/dev/sd[a-z]", "Direct disk write operation"),
            (r"curl.*\|\s*bash", "Piping web content to shell"),
            (r"wget.*\|\s*sh", "Piping downloaded content to shell"),
            (r"eval\s*\(", "Dynamic code evaluation"),
            (r"exec\s*\(", "Dynamic code execution"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, normalized):
                logger.warning(f"DANGEROUS COMMAND PATTERN: {description}")
                return SafetyCheckResult(
                    is_safe=False,
                    reason=description,
                    severity="blocked",
                    recommendation="Please review and use a safer alternative"
                )

        # Check rate limiting
        rate_check = self._check_rate_limit("command")
        if not rate_check.is_safe:
            return rate_check

        logger.debug(f"Command safety check passed: {command[:50]}")
        return SafetyCheckResult(is_safe=True)

    def check_path(self, path: str, operation: str = "read") -> SafetyCheckResult:
        """
        Verify file path access safety.

        Args:
            path: File path to validate
            operation: Type of operation ("read", "write", "delete")

        Returns:
            SafetyCheckResult indicating safety status
        """
        # Normalize path
        normalized = path.strip()

        # Check against blocked patterns
        for regex in self._blocked_path_regexes:
            if regex.search(normalized):
                logger.error(f"BLOCKED PATH ACCESS: {path} ({operation})")
                return SafetyCheckResult(
                    is_safe=False,
                    reason=f"Path matches blocked pattern: {regex.pattern}",
                    severity="blocked",
                    recommendation="This path is protected for security"
                )

        # Additional write/delete restrictions
        if operation in ("write", "delete"):
            write_protected = [
                (r"^/usr/", "System binaries directory"),
                (r"^/bin/", "Essential binaries directory"),
                (r"^/sbin/", "System binaries directory"),
                (r"^/lib/", "System libraries directory"),
                (r"\.git/config$", "Git configuration file"),
                (r"\.git/HEAD$", "Git HEAD reference"),
            ]

            for pattern, description in write_protected:
                if re.search(pattern, normalized):
                    logger.warning(f"WRITE-PROTECTED PATH: {path}")
                    return SafetyCheckResult(
                        is_safe=False,
                        reason=f"Write protected: {description}",
                        severity="blocked",
                        recommendation="Use a different location for this operation"
                    )

        logger.debug(f"Path safety check passed: {path} ({operation})")
        return SafetyCheckResult(is_safe=True)

    def check_tool_call(
        self,
        tool_name: str,
        parameters: Optional[Dict] = None
    ) -> SafetyCheckResult:
        """
        Verify tool invocation safety.

        Args:
            tool_name: Name of the tool being called
            parameters: Optional parameters passed to the tool

        Returns:
            SafetyCheckResult indicating safety status
        """
        # Check rate limiting
        rate_check = self._check_rate_limit(f"tool:{tool_name}")
        if not rate_check.is_safe:
            return rate_check

        # Track tool usage
        self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1

        # Check for specific dangerous tool patterns
        if parameters:
            # Check Bash tool for dangerous commands
            if tool_name == "Bash" and "command" in parameters:
                return self.check_command(parameters["command"])

            # Check Write/Edit tools for dangerous paths
            if tool_name in ("Write", "Edit") and "file_path" in parameters:
                return self.check_path(parameters["file_path"], operation="write")

            # Check for eval/exec in Python execution
            if tool_name == "PythonExec" and "code" in parameters:
                code = parameters["code"]
                if re.search(r"\beval\s*\(", code) or re.search(r"\bexec\s*\(", code):
                    logger.warning(f"DANGEROUS CODE PATTERN in {tool_name}")
                    return SafetyCheckResult(
                        is_safe=False,
                        reason="Code contains eval() or exec()",
                        severity="blocked",
                        recommendation="Avoid dynamic code evaluation"
                    )

        logger.debug(f"Tool call safety check passed: {tool_name}")
        return SafetyCheckResult(is_safe=True)

    def _check_rate_limit(self, action_key: str) -> SafetyCheckResult:
        """
        Check if action exceeds rate limits.

        Args:
            action_key: Identifier for the type of action

        Returns:
            SafetyCheckResult indicating rate limit status
        """
        current_time = time.time()

        # Remove timestamps outside the window
        cutoff_time = current_time - RATE_LIMIT_WINDOW_SECONDS
        while self._action_timestamps and self._action_timestamps[0] < cutoff_time:
            self._action_timestamps.popleft()

        # Check if at limit
        if len(self._action_timestamps) >= MAX_ACTIONS_PER_MINUTE:
            logger.warning(f"RATE LIMIT EXCEEDED: {action_key}")
            return SafetyCheckResult(
                is_safe=False,
                reason=f"Rate limit exceeded: {MAX_ACTIONS_PER_MINUTE} actions per minute",
                severity="warning",
                recommendation="Please slow down and wait before retrying"
            )

        # Record this action
        self._action_timestamps.append(current_time)
        return SafetyCheckResult(is_safe=True)

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current usage statistics.

        Returns:
            Dictionary of tool names to call counts
        """
        return {
            "actions_in_window": len(self._action_timestamps),
            "max_actions_per_minute": MAX_ACTIONS_PER_MINUTE,
            **self._tool_call_counts
        }

    def reset_rate_limits(self) -> None:
        """Reset rate limiting counters (for testing or manual override)."""
        self._action_timestamps.clear()
        logger.info("Rate limits reset")


# Module-level singleton
_safety_checker_instance: SafetyChecker | None = None


def get_safety_checker() -> SafetyChecker:
    """
    Get singleton instance of SafetyChecker.

    Returns:
        Shared SafetyChecker instance
    """
    global _safety_checker_instance
    if _safety_checker_instance is None:
        _safety_checker_instance = SafetyChecker()
    return _safety_checker_instance

"""
Conch Shell Tool

Provides safe command execution capabilities.
"""

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from conch.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class ShellTool(Tool):
    """Tool for executing shell commands safely.

    Features:
    - Command sanitization
    - Timeout handling
    - Working directory management
    - Output capture
    - Safety guardrails
    """

    # Additional dangerous commands to block
    DANGEROUS_COMMANDS = [
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init 0",
        "init 6",
        "killall",
        "pkill -9",
        "format",
        "fdisk",
        "parted",
        "mkswap",
        "swapon",
        "swapoff",
    ]

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        timeout: int = 60,
        max_output_size: int = 100000,
    ):
        """Initialize shell tool.

        Args:
            working_dir: Default working directory
            timeout: Default timeout in seconds
            max_output_size: Maximum output size in bytes
        """
        super().__init__(
            name="shell",
            description="Execute shell commands safely",
            requires_confirmation=False,  # Safety checks happen internally
        )

        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout
        self.max_output_size = max_output_size

    def execute(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        timeout: Optional[int] = None,
        env: Optional[dict] = None,
        capture_stderr: bool = True,
    ) -> ToolResult:
        """Execute a shell command.

        Args:
            command: Command to execute
            working_dir: Working directory (overrides default)
            timeout: Timeout in seconds (overrides default)
            env: Additional environment variables
            capture_stderr: Whether to capture stderr

        Returns:
            ToolResult with command output
        """
        start_time = time.time()

        # Safety check
        is_safe, reason = self._check_safety(command)
        if not is_safe:
            logger.warning(f"Blocked unsafe command: {command}")
            result = ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=f"Command blocked: {reason}",
                execution_time=time.time() - start_time,
                metadata={"command": command},
            )
            self._record_result(result)
            return result

        # Prepare execution
        cwd = working_dir or self.working_dir
        timeout_val = timeout or self.timeout

        # Merge environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        try:
            # Execute command
            process = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                env=run_env,
                capture_output=True,
                timeout=timeout_val,
                text=True,
            )

            # Combine output
            stdout = process.stdout or ""
            stderr = process.stderr or "" if capture_stderr else ""

            # Note: No truncation - show full output

            output = stdout
            if stderr:
                output += f"\n[stderr]\n{stderr}"

            result = ToolResult(
                status=ToolStatus.SUCCESS if process.returncode == 0 else ToolStatus.ERROR,
                output=output,
                error=stderr if process.returncode != 0 else None,
                execution_time=time.time() - start_time,
                metadata={
                    "command": command,
                    "return_code": process.returncode,
                    "working_dir": str(cwd),
                },
            )

        except subprocess.TimeoutExpired:
            result = ToolResult(
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"Command timed out after {timeout_val} seconds",
                execution_time=time.time() - start_time,
                metadata={"command": command},
            )

        except Exception as e:
            result = ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                metadata={"command": command},
            )

        self._record_result(result)
        logger.debug(f"Shell: {command} -> {result.status.value}")
        return result

    def _check_safety(self, command: str) -> tuple[bool, str]:
        """Extended safety check for shell commands."""
        # Base class check
        is_safe, reason = self.is_safe(command)
        if not is_safe:
            return False, reason

        command_lower = command.lower()

        # Check dangerous commands
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous.lower() in command_lower:
                return False, f"Dangerous command: {dangerous}"

        # Check for sudo with dangerous operations
        if "sudo" in command_lower:
            dangerous_sudo = ["rm", "chmod 777", "chown", "dd", "mkfs"]
            for d in dangerous_sudo:
                if d in command_lower:
                    return False, f"Dangerous sudo operation: {d}"

        return True, "OK"

    def run(self, command: str, **kwargs) -> str:
        """Convenience method - run command and return output.

        Args:
            command: Command to run
            **kwargs: Additional arguments

        Returns:
            Command output string

        Raises:
            RuntimeError: If command fails
        """
        result = self.execute(command, **kwargs)
        if result.success:
            return result.output
        raise RuntimeError(f"Command failed: {result.error}")


# Convenience functions
_shell_tool: Optional[ShellTool] = None


def get_shell() -> ShellTool:
    """Get the global shell tool instance."""
    global _shell_tool
    if _shell_tool is None:
        _shell_tool = ShellTool()
    return _shell_tool


def run_command(command: str, **kwargs) -> ToolResult:
    """Run a shell command using the global shell tool.

    Args:
        command: Command to run
        **kwargs: Additional arguments

    Returns:
        ToolResult
    """
    return get_shell().execute(command, **kwargs)


def sh(command: str, **kwargs) -> str:
    """Run a command and return output (raises on error).

    Args:
        command: Command to run
        **kwargs: Additional arguments

    Returns:
        Command output

    Raises:
        RuntimeError: If command fails
    """
    return get_shell().run(command, **kwargs)

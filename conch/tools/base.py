"""
Conch Tool Base Classes

Defines the interface and registry for all Conch tools.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Blocked by safety guardrails


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolStatus
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class Tool(ABC):
    """Base class for all Conch tools.

    All tools must:
    - Have a unique name
    - Provide a description
    - Implement execute()
    - Respect safety guardrails
    """

    # Safety: Commands that are always blocked
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
        "chmod -R 777 /",
        "sudo rm",
        "> /dev/sda",
    ]

    def __init__(
        self,
        name: str,
        description: str,
        requires_confirmation: bool = False,
    ):
        """Initialize tool.

        Args:
            name: Unique tool name
            description: What the tool does
            requires_confirmation: Whether to ask before executing
        """
        self.name = name
        self.description = description
        self.requires_confirmation = requires_confirmation

        # Execution history
        self.history: list[ToolResult] = []

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with output or error
        """
        pass

    def is_safe(self, command: str) -> tuple[bool, str]:
        """Check if a command is safe to execute.

        Args:
            command: Command to check

        Returns:
            (is_safe, reason) tuple
        """
        command_lower = command.lower()

        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in command_lower:
                return False, f"Blocked pattern detected: {pattern}"

        return True, "OK"

    def _record_result(self, result: ToolResult) -> None:
        """Record result in history."""
        self.history.append(result)
        # Keep only last 100 results
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_usage_stats(self) -> dict:
        """Get tool usage statistics."""
        if not self.history:
            return {"executions": 0}

        successes = sum(1 for r in self.history if r.success)
        total_time = sum(r.execution_time for r in self.history)

        return {
            "executions": len(self.history),
            "successes": successes,
            "failures": len(self.history) - successes,
            "success_rate": successes / len(self.history) if self.history else 0,
            "total_time": total_time,
            "avg_time": total_time / len(self.history) if self.history else 0,
        }


class ToolRegistry:
    """Registry for managing all available tools."""

    def __init__(self):
        """Initialize registry."""
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: Tool, category: str = "general") -> None:
        """Register a tool.

        Args:
            tool: Tool to register
            category: Tool category
        """
        # Skip if tool is already registered (avoid spam on repeated calls)
        if tool.name in self._tools:
            return

        self._tools[tool.name] = tool

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name} ({category})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            for cat_tools in self._categories.values():
                if name in cat_tools:
                    cat_tools.remove(name)
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Tool '{name}' not found",
            )

        try:
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

    def list_tools(self) -> dict[str, list[dict]]:
        """List all tools by category."""
        result = {}
        for category, tool_names in self._categories.items():
            result[category] = [
                {
                    "name": name,
                    "description": self._tools[name].description,
                    "requires_confirmation": self._tools[name].requires_confirmation,
                }
                for name in tool_names
                if name in self._tools
            ]
        return result

    def get_tool_help(self, name: str) -> str:
        """Get help text for a tool."""
        tool = self._tools.get(name)
        if not tool:
            return f"Tool '{name}' not found"

        return f"""Tool: {tool.name}
Description: {tool.description}
Requires Confirmation: {tool.requires_confirmation}

Usage: registry.execute("{tool.name}", ...)
"""


# Global registry
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry

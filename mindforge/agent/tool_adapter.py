"""
MindForge Tool Adapter

Bridges MindForge tools to LangGraph's tool interface.
"""

import json
import logging
from typing import Any, Callable, Optional

from mindforge.tools import (
    Tool,
    ToolResult,
    ToolStatus,
    ToolRegistry,
    get_registry,
    setup_default_registry,
)
from mindforge.integrations.n8n import N8NTool, get_n8n
from mindforge.integrations.ollama import OllamaTool, get_ollama

logger = logging.getLogger(__name__)


class ToolAdapter:
    """Adapts MindForge tools for use with LangGraph.

    Provides a unified interface that can be used as LangGraph tools
    while maintaining the MindForge tool semantics.
    """

    def __init__(self, tool: Tool):
        """Initialize adapter with a MindForge tool.

        Args:
            tool: MindForge tool to adapt
        """
        self.tool = tool
        self.name = tool.name
        self.description = tool.description

    def __call__(self, **kwargs) -> str:
        """Execute the tool and return string result.

        LangGraph tools need to return strings, so we format
        the ToolResult appropriately.
        """
        result = self.tool.execute(**kwargs)

        if result.success:
            return result.output
        else:
            return f"Error: {result.error}"

    def to_dict(self) -> dict:
        """Convert to dictionary format for tool schemas."""
        return {
            "name": self.name,
            "description": self.description,
        }


def create_tool_function(
    tool: Tool,
    operation: Optional[str] = None,
) -> Callable:
    """Create a callable function from a MindForge tool.

    Args:
        tool: MindForge tool
        operation: Specific operation to bind (for multi-operation tools)

    Returns:
        Callable function for LangGraph
    """

    def tool_func(**kwargs) -> str:
        """Execute the tool."""
        if operation:
            kwargs["operation"] = operation

        result = tool.execute(**kwargs)

        if result.success:
            output = result.output
            # Include relevant metadata
            if result.metadata:
                meta_str = json.dumps(result.metadata, indent=2, default=str)
                if len(meta_str) < 500:  # Don't include huge metadata
                    output += f"\n\nMetadata: {meta_str}"
            return output
        else:
            return f"Error: {result.error}"

    # Set function name and docstring for LangGraph
    tool_func.__name__ = f"{tool.name}_{operation}" if operation else tool.name
    tool_func.__doc__ = tool.description

    return tool_func


def get_all_tools(
    n8n_url: str = "http://localhost:5678",
    n8n_api_key: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "echo_assistant",
) -> dict[str, Tool]:
    """Get all available MindForge tools.

    Args:
        n8n_url: n8n server URL
        n8n_api_key: n8n API key
        ollama_host: Ollama server URL
        ollama_model: Default Ollama model

    Returns:
        Dictionary of tool name to tool instance
    """
    # Setup default registry (shell, filesystem, web, code, git)
    registry = setup_default_registry()

    # Add integration tools
    n8n = get_n8n(base_url=n8n_url, api_key=n8n_api_key)
    ollama = get_ollama(host=ollama_host, default_model=ollama_model)

    registry.register(n8n, category="integrations")
    registry.register(ollama, category="integrations")

    # Collect all tools
    tools = {}
    for category_tools in registry.list_tools().values():
        for tool_info in category_tools:
            tool = registry.get(tool_info["name"])
            if tool:
                tools[tool.name] = tool

    return tools


def create_langgraph_tools(
    n8n_url: str = "http://localhost:5678",
    n8n_api_key: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "echo_assistant",
) -> list[dict]:
    """Create tool definitions for LangGraph.

    This creates a list of tool schemas that can be used with
    LangGraph's tool calling interface.

    Returns:
        List of tool definitions with name, description, and function
    """
    tools = get_all_tools(
        n8n_url=n8n_url,
        n8n_api_key=n8n_api_key,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )

    langgraph_tools = []

    for name, tool in tools.items():
        # Create base tool entry
        tool_def = {
            "name": name,
            "description": tool.description,
            "func": create_tool_function(tool),
        }

        # For multi-operation tools, also create operation-specific tools
        if name == "n8n":
            operations = ["health", "list", "run", "webhook", "history"]
            for op in operations:
                op_tool = {
                    "name": f"n8n_{op}",
                    "description": f"n8n: {op} operation",
                    "func": create_tool_function(tool, operation=op),
                }
                langgraph_tools.append(op_tool)

        elif name == "ollama":
            operations = ["health", "list", "generate", "chat"]
            for op in operations:
                op_tool = {
                    "name": f"ollama_{op}",
                    "description": f"Ollama: {op} operation",
                    "func": create_tool_function(tool, operation=op),
                }
                langgraph_tools.append(op_tool)

        elif name == "filesystem":
            operations = ["read", "write", "list", "glob", "grep", "mkdir"]
            for op in operations:
                op_tool = {
                    "name": f"fs_{op}",
                    "description": f"Filesystem: {op} operation",
                    "func": create_tool_function(tool, operation=op),
                }
                langgraph_tools.append(op_tool)

        elif name == "code":
            operations = ["analyze", "validate", "symbols", "diff"]
            for op in operations:
                op_tool = {
                    "name": f"code_{op}",
                    "description": f"Code analysis: {op} operation",
                    "func": create_tool_function(tool, operation=op),
                }
                langgraph_tools.append(op_tool)

        elif name == "git":
            operations = ["status", "diff", "log", "commit", "branch"]
            for op in operations:
                op_tool = {
                    "name": f"git_{op}",
                    "description": f"Git: {op} operation",
                    "func": create_tool_function(tool, operation=op),
                }
                langgraph_tools.append(op_tool)

        else:
            # Simple tools get added directly
            langgraph_tools.append(tool_def)

    return langgraph_tools


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools.

    Useful for including in system prompts so the agent
    knows what tools are available.
    """
    tools = get_all_tools()

    descriptions = ["Available tools:\n"]

    for name, tool in sorted(tools.items()):
        descriptions.append(f"- **{name}**: {tool.description}")

    return "\n".join(descriptions)


class DoNothingTool(Tool):
    """Special tool for doing nothing.

    This is an explicit action the consciousness can take
    to rest, observe, or wait.
    """

    def __init__(self):
        super().__init__(
            name="do_nothing",
            description="Explicitly choose inaction. Use this to rest, observe, or when no action is needed.",
            requires_confirmation=False,
        )

    def execute(self, reason: str = "No action needed") -> ToolResult:
        """Execute doing nothing.

        Args:
            reason: Why choosing inaction

        Returns:
            ToolResult indicating successful inaction
        """
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Chose inaction: {reason}",
            metadata={"action": "do_nothing", "reason": reason},
        )


def get_do_nothing_tool() -> DoNothingTool:
    """Get the do-nothing tool for conscious inaction."""
    return DoNothingTool()

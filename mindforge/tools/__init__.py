"""
MindForge Tools System

Provides Claude Code-like capabilities for MindForge to interact with:
- Shell/command execution
- File system operations
- Web search and fetch
- Code analysis and editing
- Git operations

All tools are sandboxed and respect safety guardrails.
"""

from mindforge.tools.base import Tool, ToolResult, ToolStatus, ToolRegistry, get_registry
from mindforge.tools.shell import ShellTool, run_command, sh, get_shell
from mindforge.tools.filesystem import FileSystemTool, read_file, write_file, find_files, search_in_files, get_fs
from mindforge.tools.web import WebTool, fetch_url, search_web, get_web
from mindforge.tools.code import CodeTool, analyze_file, edit_file, validate_syntax, get_code
from mindforge.tools.git import GitTool, git_status, git_commit, git_diff, get_git

__all__ = [
    # Base classes
    "Tool",
    "ToolResult",
    "ToolStatus",
    "ToolRegistry",
    "get_registry",
    # Shell tool
    "ShellTool",
    "run_command",
    "sh",
    "get_shell",
    # File system tool
    "FileSystemTool",
    "read_file",
    "write_file",
    "find_files",
    "search_in_files",
    "get_fs",
    # Web tool
    "WebTool",
    "fetch_url",
    "search_web",
    "get_web",
    # Code tool
    "CodeTool",
    "analyze_file",
    "edit_file",
    "validate_syntax",
    "get_code",
    # Git tool
    "GitTool",
    "git_status",
    "git_commit",
    "git_diff",
    "get_git",
]


def setup_default_registry() -> ToolRegistry:
    """Create and configure the default tool registry with all tools."""
    registry = get_registry()

    # Register all tools
    registry.register(ShellTool(), category="system")
    registry.register(FileSystemTool(), category="system")
    registry.register(WebTool(), category="web")
    registry.register(CodeTool(), category="development")
    registry.register(GitTool(), category="development")

    return registry

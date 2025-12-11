"""
MindForge DNA - Tools Layer

Tool system for safe execution of operations with security validation.
Provides filesystem, shell, git, and KVRM integration capabilities.

Usage:
    from mindforge_dna.tools import create_default_registry

    registry = create_default_registry()
    result = registry.execute("filesystem", operation="read", path="/path/to/file")
"""

import logging
from typing import Any, Dict, List, Optional, Set

from .base import Tool, ToolRegistry, ToolResult, ToolStatus
from .filesystem import FileSystemTool
from .git import GitTool
from .kvrm_tool import KVRMTool
from .shell import ShellTool

logger = logging.getLogger(__name__)

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolStatus",
    "ShellTool",
    "FileSystemTool",
    "GitTool",
    "KVRMTool",
    "create_default_registry",
    "create_restricted_registry",
]


def create_default_registry(
    kvrm_instance: Optional[Any] = None,
    shell_timeout: float = 30.0,
    max_read_size: int = 10 * 1024 * 1024,
) -> ToolRegistry:
    """
    Create a registry with all standard tools configured.

    Args:
        kvrm_instance: Optional KVRM instance for symbolic reasoning
        shell_timeout: Default timeout for shell commands
        max_read_size: Maximum file size for reads (bytes)

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    # Register shell tool
    shell = ShellTool(
        name="shell",
        description="Execute shell commands safely",
        timeout=shell_timeout,
    )
    registry.register(shell)

    # Register filesystem tool
    filesystem = FileSystemTool(
        name="filesystem",
        description="Perform filesystem operations",
        max_read_size=max_read_size,
    )
    registry.register(filesystem)

    # Register git tool
    git = GitTool(
        name="git",
        description="Perform safe git operations",
    )
    registry.register(git)

    # Register KVRM tool
    kvrm = KVRMTool(
        name="kvrm",
        description="Access KVRM for symbolic reasoning",
        kvrm_instance=kvrm_instance,
    )
    registry.register(kvrm)

    logger.info(
        f"Default registry created with {len(registry)} tools: "
        f"{', '.join(t.name for t in registry)}"
    )

    return registry


def create_restricted_registry(
    allowed_operations: Dict[str, List[str]],
    allowed_paths: Optional[Set[str]] = None,
    allowed_commands: Optional[Set[str]] = None,
) -> ToolRegistry:
    """
    Create a restricted registry with limited capabilities.

    Args:
        allowed_operations: Map of tool -> allowed operations
            Example: {"filesystem": ["read", "list"], "git": ["status"]}
        allowed_paths: Allowed filesystem paths
        allowed_commands: Allowed shell commands

    Returns:
        Restricted ToolRegistry
    """
    registry = ToolRegistry()

    # Configure shell if allowed
    if "shell" in allowed_operations:
        shell = ShellTool(
            name="shell",
            description="Execute restricted shell commands",
            allowed_commands=allowed_commands,
        )
        # Disable if no operations allowed
        if not allowed_operations["shell"]:
            shell.enabled = False
        registry.register(shell)

    # Configure filesystem if allowed
    if "filesystem" in allowed_operations:
        filesystem = FileSystemTool(
            name="filesystem",
            description="Perform restricted filesystem operations",
            allowed_paths=allowed_paths,
        )
        # Disable if no operations allowed
        if not allowed_operations["filesystem"]:
            filesystem.enabled = False
        registry.register(filesystem)

    # Configure git if allowed
    if "git" in allowed_operations:
        git = GitTool(
            name="git",
            description="Perform restricted git operations",
        )
        # Disable if no operations allowed
        if not allowed_operations["git"]:
            git.enabled = False
        registry.register(git)

    # KVRM is always safe (read-only by design)
    if "kvrm" in allowed_operations:
        kvrm = KVRMTool(
            name="kvrm",
            description="Access KVRM for symbolic reasoning",
        )
        registry.register(kvrm)

    logger.info(
        f"Restricted registry created with {len(registry)} tools: "
        f"{', '.join(t.name for t in registry)}"
    )

    return registry


def get_tool_documentation() -> Dict[str, str]:
    """
    Get documentation for all available tools.

    Returns:
        Map of tool name to documentation string
    """
    return {
        "shell": """
        Execute shell commands safely with security validation.

        Operations:
            execute(command, timeout, cwd, env, shell)

        Security:
            - Blocks dangerous patterns (rm -rf /, dd, mkfs, etc.)
            - Configurable timeout (default 30s)
            - Optional command whitelist

        Example:
            result = registry.execute(
                "shell",
                command="ls -la /tmp",
                timeout=10.0
            )
        """,

        "filesystem": """
        Perform filesystem operations with path validation.

        Operations:
            read(path, encoding)
            write(path, content, encoding, create_dirs)
            list(path, pattern, recursive)
            exists(path)
            info(path)
            mkdir(path, parents)
            delete(path, recursive)

        Security:
            - Blocks system paths (/etc/passwd, /boot, etc.)
            - Size limits on reads (default 10MB)
            - Optional path whitelist

        Example:
            result = registry.execute(
                "filesystem",
                operation="read",
                path="/path/to/file.txt"
            )
        """,

        "git": """
        Perform safe git operations (no destructive commands).

        Operations:
            status(cwd)
            log(cwd, max_count, oneline)
            diff(cwd, cached, file_path)
            branch(cwd, list_all)
            add(cwd, paths, all_files)
            commit(cwd, message)

        Security:
            - Blocks force push, hard reset, etc.
            - Repository validation
            - Timeout support

        Example:
            result = registry.execute(
                "git",
                operation="status",
                cwd="/path/to/repo"
            )
        """,

        "kvrm": """
        Access KVRM system for symbolic reasoning.

        Operations:
            resolve(query, context)
            search(query, domain, max_results)
            store(key, value, domain)
            ground(symbol, grounding)
            list(domain, pattern)

        Features:
            - Knowledge retrieval and resolution
            - Value alignment checking
            - Rule evaluation
            - Memory storage and grounding

        Example:
            result = registry.execute(
                "kvrm",
                operation="search",
                query="safety rules",
                domain="rules"
            )
        """,
    }

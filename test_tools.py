#!/usr/bin/env python3
"""
Quick test script for Conch DNA Tools layer.
Verifies basic functionality of all tool implementations.
"""

import sys
import tempfile
from pathlib import Path

# Add conch_dna to path
sys.path.insert(0, str(Path(__file__).parent))

from conch_dna.tools import (
    create_default_registry,
    create_restricted_registry,
    ToolStatus,
)


def test_registry_creation():
    """Test registry creation and basic operations."""
    print("=" * 60)
    print("Testing Registry Creation")
    print("=" * 60)

    registry = create_default_registry()
    print(f"✓ Created registry with {len(registry)} tools")

    tools = registry.list_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    print()


def test_filesystem_tool():
    """Test filesystem operations."""
    print("=" * 60)
    print("Testing FileSystem Tool")
    print("=" * 60)

    registry = create_default_registry()

    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"

        # Test write
        result = registry.execute(
            "filesystem",
            operation="write",
            path=str(test_file),
            content="Hello, Conch!",
        )
        print(f"Write: {result.status.value} - {result.output}")

        # Test read
        result = registry.execute(
            "filesystem",
            operation="read",
            path=str(test_file),
        )
        print(f"Read: {result.status.value}")
        print(f"Content: {result.output}")

        # Test exists
        result = registry.execute(
            "filesystem",
            operation="exists",
            path=str(test_file),
        )
        print(f"Exists: {result.output}")

        # Test list
        result = registry.execute(
            "filesystem",
            operation="list",
            path=tmpdir,
        )
        print(f"List directory:\n{result.output}")

    print()


def test_shell_tool():
    """Test shell command execution."""
    print("=" * 60)
    print("Testing Shell Tool")
    print("=" * 60)

    registry = create_default_registry()

    # Test safe command
    result = registry.execute(
        "shell",
        command="echo 'Hello from shell'",
    )
    print(f"Safe command: {result.status.value}")
    print(f"Output: {result.output.strip()}")

    # Test blocked command
    result = registry.execute(
        "shell",
        command="rm -rf /",
    )
    print(f"Blocked command: {result.status.value}")
    print(f"Error: {result.error}")

    print()


def test_git_tool():
    """Test git operations."""
    print("=" * 60)
    print("Testing Git Tool")
    print("=" * 60)

    registry = create_default_registry()

    # Test in current directory (may or may not be a git repo)
    result = registry.execute(
        "git",
        operation="status",
        cwd=".",
    )
    print(f"Git status: {result.status.value}")
    if result.success:
        print(f"Output:\n{result.output}")
    else:
        print(f"Error: {result.error}")

    print()


def test_kvrm_tool():
    """Test KVRM integration."""
    print("=" * 60)
    print("Testing KVRM Tool")
    print("=" * 60)

    registry = create_default_registry()

    # Test search (placeholder implementation)
    result = registry.execute(
        "kvrm",
        operation="search",
        query="test query",
    )
    print(f"KVRM search: {result.status.value}")
    print(f"Output:\n{result.output}")

    # Test list
    result = registry.execute(
        "kvrm",
        operation="list",
        domain="knowledge",
    )
    print(f"\nKVRM list: {result.status.value}")
    print(f"Output:\n{result.output}")

    print()


def test_restricted_registry():
    """Test restricted registry with limited capabilities."""
    print("=" * 60)
    print("Testing Restricted Registry")
    print("=" * 60)

    # Create registry with only read-only filesystem access
    registry = create_restricted_registry(
        allowed_operations={
            "filesystem": ["read", "list", "exists"],
            "git": ["status", "log"],
        },
        allowed_paths={"/tmp"},
    )

    print(f"✓ Created restricted registry with {len(registry)} tools")

    tools = registry.list_tools()
    for tool in tools:
        status = "enabled" if tool["enabled"] else "disabled"
        print(f"  - {tool['name']}: {status}")

    print()


def test_security_features():
    """Test security validation."""
    print("=" * 60)
    print("Testing Security Features")
    print("=" * 60)

    registry = create_default_registry()

    # Test blocked filesystem paths
    blocked_paths = ["/etc/passwd", "/etc/shadow", "/boot/vmlinuz"]
    for path in blocked_paths:
        result = registry.execute(
            "filesystem",
            operation="read",
            path=path,
        )
        assert result.status == ToolStatus.BLOCKED, f"Failed to block {path}"
        print(f"✓ Blocked access to {path}")

    # Test blocked shell commands
    blocked_cmds = [
        "rm -rf /",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|: & };:",
    ]
    for cmd in blocked_cmds:
        result = registry.execute("shell", command=cmd)
        assert result.status == ToolStatus.BLOCKED, f"Failed to block {cmd}"
        print(f"✓ Blocked command: {cmd}")

    print()


def test_statistics():
    """Test execution statistics."""
    print("=" * 60)
    print("Testing Statistics")
    print("=" * 60)

    registry = create_default_registry()

    # Execute some operations
    registry.execute("shell", command="echo 'test'")
    registry.execute("shell", command="rm -rf /")  # Blocked
    registry.execute("filesystem", operation="exists", path="/tmp")

    # Get statistics
    stats = registry.get_stats()
    print(f"Total tools: {stats['total_tools']}")
    print(f"Enabled tools: {stats['enabled_tools']}")
    print("\nTool statistics:")
    for name, tool_stats in stats['tools'].items():
        print(f"  {name}:")
        print(f"    Executions: {tool_stats['executions']}")
        print(f"    Errors: {tool_stats['errors']}")
        print(f"    Success rate: {tool_stats['success_rate']:.1%}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Conch DNA - Tools Layer Test Suite")
    print("=" * 60 + "\n")

    try:
        test_registry_creation()
        test_filesystem_tool()
        test_shell_tool()
        test_git_tool()
        test_kvrm_tool()
        test_restricted_registry()
        test_security_features()
        test_statistics()

        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

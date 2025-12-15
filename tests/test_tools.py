"""
Conch Tools Test Suite

Comprehensive tests for all tool capabilities.
"""

import pytest
import tempfile
import os
from pathlib import Path

from conch.tools import (
    Tool, ToolResult, ToolStatus, ToolRegistry, get_registry,
    ShellTool, FileSystemTool, WebTool, CodeTool, GitTool,
    setup_default_registry,
)


class TestToolBase:
    """Test base tool classes."""

    def test_tool_result_success(self):
        """Test successful tool result."""
        result = ToolResult(
            status=ToolStatus.SUCCESS,
            output="test output",
        )
        assert result.success is True
        assert result.output == "test output"
        assert result.error is None

    def test_tool_result_error(self):
        """Test error tool result."""
        result = ToolResult(
            status=ToolStatus.ERROR,
            output="",
            error="something went wrong",
        )
        assert result.success is False
        assert result.error == "something went wrong"

    def test_tool_result_to_dict(self):
        """Test result serialization."""
        result = ToolResult(
            status=ToolStatus.SUCCESS,
            output="test",
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["output"] == "test"
        assert d["metadata"]["key"] == "value"

    def test_registry_register(self):
        """Test tool registration."""
        registry = ToolRegistry()
        shell = ShellTool()
        registry.register(shell, category="test")

        assert registry.get("shell") is shell
        tools = registry.list_tools()
        assert "test" in tools
        assert any(t["name"] == "shell" for t in tools["test"])

    def test_registry_execute(self):
        """Test executing via registry."""
        registry = ToolRegistry()
        registry.register(ShellTool(), category="test")

        result = registry.execute("shell", command="echo hello")
        assert result.success
        assert "hello" in result.output

    def test_registry_not_found(self):
        """Test executing non-existent tool."""
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert not result.success
        assert "not found" in result.error


class TestShellTool:
    """Test shell tool."""

    def test_echo_command(self):
        """Test basic echo command."""
        shell = ShellTool()
        result = shell.execute("echo 'test output'")
        assert result.success
        assert "test output" in result.output

    def test_blocked_command(self):
        """Test dangerous command blocking."""
        shell = ShellTool()
        result = shell.execute("rm -rf /")
        assert result.status == ToolStatus.BLOCKED
        assert "blocked" in result.error.lower()

    def test_sudo_rm_blocked(self):
        """Test sudo rm is blocked."""
        shell = ShellTool()
        result = shell.execute("sudo rm -rf /home")
        assert result.status == ToolStatus.BLOCKED

    def test_fork_bomb_blocked(self):
        """Test fork bomb is blocked."""
        shell = ShellTool()
        result = shell.execute(":(){:|:&};:")
        assert result.status == ToolStatus.BLOCKED

    def test_command_timeout(self):
        """Test command timeout handling."""
        shell = ShellTool(timeout=1)
        result = shell.execute("sleep 10")
        assert result.status == ToolStatus.TIMEOUT

    def test_run_convenience_method(self):
        """Test run() convenience method."""
        shell = ShellTool()
        output = shell.run("echo success")
        assert "success" in output

    def test_run_raises_on_failure(self):
        """Test run() raises on failure."""
        shell = ShellTool()
        with pytest.raises(RuntimeError):
            shell.run("exit 1")


class TestFileSystemTool:
    """Test file system tool."""

    def test_read_file(self):
        """Test reading a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            fs = FileSystemTool()
            result = fs.execute("read", path=temp_path)
            assert result.success
            assert "test content" in result.output
        finally:
            os.unlink(temp_path)

    def test_write_file(self):
        """Test writing a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystemTool()
            path = os.path.join(tmpdir, "test.txt")

            result = fs.execute("write", path=path, content="hello world")
            assert result.success

            # Verify content
            with open(path) as f:
                assert f.read() == "hello world"

    def test_list_directory(self):
        """Test listing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()
            Path(tmpdir, "subdir").mkdir()

            fs = FileSystemTool()
            result = fs.execute("list", path=tmpdir)
            assert result.success
            assert result.metadata["count"] == 3

    def test_glob_pattern(self):
        """Test glob file search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").touch()
            Path(tmpdir, "file2.py").touch()
            Path(tmpdir, "file3.txt").touch()

            fs = FileSystemTool()
            result = fs.execute("glob", pattern="*.py", path=tmpdir)
            assert result.success
            assert result.metadata["count"] == 2

    def test_grep_search(self):
        """Test grep in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "test.py")
            path.write_text("def hello():\n    pass\n\ndef world():\n    pass\n")

            fs = FileSystemTool()
            result = fs.execute("grep", pattern="def", path=tmpdir, file_pattern="*.py")
            assert result.success
            assert result.metadata["matches"] == 2

    def test_file_not_found(self):
        """Test reading non-existent file."""
        fs = FileSystemTool()
        result = fs.execute("read", path="/nonexistent/file.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_mkdir(self):
        """Test creating directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystemTool()
            path = os.path.join(tmpdir, "new", "nested", "dir")

            result = fs.execute("mkdir", path=path)
            assert result.success
            assert Path(path).is_dir()


class TestWebTool:
    """Test web tool."""

    def test_validate_safe_url(self):
        """Test validating safe URL."""
        web = WebTool()
        result = web.execute("validate", url="https://example.com")
        assert result.success
        assert result.metadata["valid"] is True

    def test_validate_localhost_blocked(self):
        """Test localhost is blocked."""
        web = WebTool()
        result = web.execute("validate", url="http://localhost:8080")
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_validate_private_ip_blocked(self):
        """Test private IP is blocked."""
        web = WebTool()

        # Test various private ranges
        for ip in ["192.168.1.1", "10.0.0.1", "127.0.0.1"]:
            result = web.execute("validate", url=f"http://{ip}")
            assert not result.success

    def test_validate_invalid_scheme(self):
        """Test invalid URL scheme."""
        web = WebTool()
        result = web.execute("validate", url="ftp://example.com")
        assert not result.success

    def test_search_placeholder(self):
        """Test search returns placeholder message."""
        web = WebTool()
        result = web.execute("search", query="test query")
        assert result.success
        assert "API" in result.output  # Mentions needing API


class TestCodeTool:
    """Test code tool."""

    def test_analyze_python_file(self):
        """Test analyzing Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
from pathlib import Path

class MyClass:
    def method(self):
        pass

def my_function(arg1, arg2):
    return arg1 + arg2
""")
            temp_path = f.name

        try:
            code = CodeTool()
            result = code.execute("analyze", path=temp_path)
            assert result.success

            meta = result.metadata
            assert meta["language"] == "python"
            assert meta["syntax_valid"] is True
            assert len(meta["functions"]) >= 1
            assert len(meta["classes"]) >= 1
            assert len(meta["imports"]) >= 2
        finally:
            os.unlink(temp_path)

    def test_validate_valid_python(self):
        """Test validating valid Python syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            temp_path = f.name

        try:
            code = CodeTool()
            result = code.execute("validate", path=temp_path)
            assert result.success
        finally:
            os.unlink(temp_path)

    def test_validate_invalid_python(self):
        """Test validating invalid Python syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello(\n    return 'broken'\n")
            temp_path = f.name

        try:
            code = CodeTool()
            result = code.execute("validate", path=temp_path)
            assert not result.success
            assert "syntax" in result.error.lower()
        finally:
            os.unlink(temp_path)

    def test_extract_symbols(self):
        """Test extracting symbols from code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class Person:
    """A person class."""
    def __init__(self, name):
        self.name = name

def greet(person):
    """Greet a person."""
    return f"Hello, {person.name}"

NAME = "test"
''')
            temp_path = f.name

        try:
            code = CodeTool()
            result = code.execute("symbols", path=temp_path)
            assert result.success

            symbols = result.metadata["symbols"]
            names = [s["name"] for s in symbols]
            assert "Person" in names
            assert "greet" in names
            assert "NAME" in names
        finally:
            os.unlink(temp_path)

    def test_generate_diff(self):
        """Test generating diff."""
        code = CodeTool()
        result = code.execute(
            "diff",
            old_content="line 1\nline 2\nline 3\n",
            new_content="line 1\nline 2 modified\nline 3\n",
            filename="test.py"
        )
        assert result.success
        assert "-line 2" in result.output
        assert "+line 2 modified" in result.output


class TestGitTool:
    """Test git tool."""

    def test_not_a_repo(self):
        """Test operations on non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git = GitTool(repo_path=Path(tmpdir))
            result = git.execute("info")
            assert not result.success
            assert "not a git repository" in result.error.lower()

    def test_dangerous_operation_blocked(self):
        """Test dangerous operations are blocked."""
        git = GitTool()

        dangerous_commands = [
            ["push", "--force"],
            ["reset", "--hard"],
            ["branch", "-D", "main"],
            ["clean", "-fd"],
        ]

        for args in dangerous_commands:
            with pytest.raises(ValueError) as exc:
                git._run_git(args)
            assert "dangerous" in str(exc.value).lower()

    def test_status_on_non_repo(self):
        """Test status on non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git = GitTool(repo_path=Path(tmpdir))
            result = git.execute("status")
            assert not result.success


class TestSetupRegistry:
    """Test default registry setup."""

    def test_setup_default_registry(self):
        """Test setting up default registry."""
        registry = setup_default_registry()

        # Check all tools registered
        assert registry.get("shell") is not None
        assert registry.get("filesystem") is not None
        assert registry.get("web") is not None
        assert registry.get("code") is not None
        assert registry.get("git") is not None

    def test_registry_categories(self):
        """Test tools are in correct categories."""
        registry = setup_default_registry()
        tools = registry.list_tools()

        assert "system" in tools
        assert "web" in tools
        assert "development" in tools

        system_names = [t["name"] for t in tools["system"]]
        assert "shell" in system_names
        assert "filesystem" in system_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

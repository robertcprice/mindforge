# Conch DNA - Tools Layer

Production-ready tool system for safe execution of operations with comprehensive security validation, error handling, and logging.

## Overview

The Tools layer provides a secure, extensible framework for executing operations across different domains:

- **Shell**: Safe command execution with pattern blocking
- **FileSystem**: File and directory operations with path validation
- **Git**: Version control operations (read-only and safe writes)
- **KVRM**: Symbolic reasoning integration (Superego layer)

## Architecture

```
ToolRegistry
├── ShellTool
│   ├── Command execution
│   ├── Timeout support
│   └── Security validation
├── FileSystemTool
│   ├── Read/Write operations
│   ├── Path validation
│   └── Size limits
├── GitTool
│   ├── Safe git operations
│   └── Destructive command blocking
└── KVRMTool
    ├── Knowledge resolution
    ├── Symbolic search
    └── Memory grounding
```

## Quick Start

```python
from conch_dna.tools import create_default_registry

# Create registry with all tools
registry = create_default_registry()

# Execute filesystem operation
result = registry.execute(
    "filesystem",
    operation="read",
    path="/path/to/file.txt"
)

if result.success:
    print(result.output)
else:
    print(f"Error: {result.error}")
```

## Tool Reference

### ShellTool

Execute shell commands with security validation.

**Operations:**
- `execute(command, timeout, cwd, env, shell)`

**Security Features:**
- Blocks dangerous patterns (rm -rf /, dd, mkfs, fork bombs)
- Configurable timeout (default 30s)
- Optional command whitelist
- Stdout/stderr capture

**Example:**
```python
result = registry.execute(
    "shell",
    command="ls -la /tmp",
    timeout=10.0,
    cwd="/home/user"
)
```

**Blocked Patterns:**
- `rm -rf /` - Recursive delete from root
- `dd if=/dev/... of=/` - Direct disk writes
- `mkfs.*` - Format filesystem
- `:(){ :|: & };:` - Fork bomb
- `shutdown|reboot|halt` - System control
- `chmod 777` - Dangerous permissions
- `curl ... | sh` - Pipe to shell

### FileSystemTool

Perform filesystem operations with path validation.

**Operations:**
- `read(path, encoding="utf-8")` - Read file contents
- `write(path, content, encoding="utf-8", create_dirs=True)` - Write file
- `list(path, pattern="*", recursive=False)` - List directory
- `exists(path)` - Check if path exists
- `info(path)` - Get file/directory information
- `mkdir(path, parents=True)` - Create directory
- `delete(path, recursive=False)` - Delete file/directory

**Security Features:**
- Blocks system paths (/etc/passwd, /boot, /dev, /proc, /sys)
- Size limits on reads (default 10MB)
- Optional path whitelist
- Automatic parent directory creation

**Example:**
```python
# Write file
result = registry.execute(
    "filesystem",
    operation="write",
    path="/tmp/output.txt",
    content="Hello, World!"
)

# Read file
result = registry.execute(
    "filesystem",
    operation="read",
    path="/tmp/output.txt"
)

# List directory
result = registry.execute(
    "filesystem",
    operation="list",
    path="/tmp",
    pattern="*.txt",
    recursive=True
)
```

### GitTool

Safe git operations with protection against destructive commands.

**Operations:**
- `status(cwd)` - Get repository status
- `log(cwd, max_count=10, oneline=False)` - View commit history
- `diff(cwd, cached=False, file_path=None)` - Show changes
- `branch(cwd, list_all=False)` - List branches
- `add(cwd, paths=None, all_files=False)` - Stage files
- `commit(cwd, message)` - Create commit

**Security Features:**
- Blocks force push, hard reset, branch deletion
- Repository validation before operations
- Timeout support for long operations
- Safe operations only

**Example:**
```python
# Get status
result = registry.execute(
    "git",
    operation="status",
    cwd="/path/to/repo"
)

# View log
result = registry.execute(
    "git",
    operation="log",
    cwd="/path/to/repo",
    max_count=5,
    oneline=True
)

# Stage and commit
result = registry.execute(
    "git",
    operation="add",
    cwd="/path/to/repo",
    paths=["file1.py", "file2.py"]
)

result = registry.execute(
    "git",
    operation="commit",
    cwd="/path/to/repo",
    message="Update files"
)
```

**Blocked Operations:**
- `push --force` / `push -f`
- `reset --hard`
- `clean -fd` / `clean -fdx`
- `branch -D`
- `rebase --force`
- `filter-branch`

### KVRMTool

Integration with Superego KVRM for symbolic reasoning.

**Operations:**
- `resolve(query, context)` - Resolve knowledge query
- `search(query, domain, max_results=10)` - Search knowledge base
- `store(key, value, domain="memory")` - Store information
- `ground(symbol, grounding)` - Ground symbolic concept
- `list(domain, pattern)` - List KVRM contents

**Domains:**
- `knowledge` - Facts and information
- `values` - Value alignment data
- `rules` - Behavioral rules
- `memory` - Session memory

**Example:**
```python
# Search knowledge base
result = registry.execute(
    "kvrm",
    operation="search",
    query="safety protocols",
    domain="rules",
    max_results=5
)

# Store memory
result = registry.execute(
    "kvrm",
    operation="store",
    key="user_preference",
    value={"style": "verbose"},
    domain="memory"
)

# Resolve query
result = registry.execute(
    "kvrm",
    operation="resolve",
    query="what are safety rules?",
    context={"user_level": "admin"}
)
```

## Advanced Usage

### Creating Restricted Registry

For security-sensitive contexts, create a restricted registry:

```python
from conch_dna.tools import create_restricted_registry

registry = create_restricted_registry(
    allowed_operations={
        "filesystem": ["read", "list", "exists"],  # Read-only
        "git": ["status", "log", "diff"],          # Read-only
    },
    allowed_paths={"/home/user/projects"},
    allowed_commands={"ls", "echo", "cat"}
)
```

### Custom Tool Implementation

Create custom tools by extending the `Tool` base class:

```python
from conch_dna.tools.base import Tool, ToolResult, ToolStatus

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom tool implementation"
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            # Your implementation here
            output = "Success!"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={"custom_key": "value"}
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e)
            )

# Register custom tool
registry = create_default_registry()
registry.register(CustomTool())
```

### Execution Statistics

Track tool usage and performance:

```python
# Get overall statistics
stats = registry.get_stats()
print(f"Total tools: {stats['total_tools']}")
print(f"Enabled: {stats['enabled_tools']}")

# Tool-specific stats
for name, tool_stats in stats['tools'].items():
    print(f"{name}:")
    print(f"  Executions: {tool_stats['executions']}")
    print(f"  Errors: {tool_stats['errors']}")
    print(f"  Success rate: {tool_stats['success_rate']:.1%}")

# Individual tool stats
shell_tool = registry.get("shell")
print(shell_tool.get_stats())
```

### Dynamic Tool Management

Enable/disable tools at runtime:

```python
# Disable tool
registry.disable("shell")

# Enable tool
registry.enable("shell")

# Check if tool exists
if "filesystem" in registry:
    print("Filesystem tool available")

# Iterate over tools
for tool in registry:
    print(f"{tool.name}: {tool.description}")
```

## Security Model

### Defense in Depth

1. **Input Validation**: All inputs validated before execution
2. **Pattern Blocking**: Dangerous patterns blocked at tool level
3. **Path Validation**: Filesystem paths checked against blocked patterns
4. **Command Whitelisting**: Optional command restriction
5. **Timeout Protection**: All operations have configurable timeouts
6. **Error Isolation**: Exceptions caught and returned as ToolResult

### Blocked Resources

**Filesystem Paths:**
- `/etc/passwd`, `/etc/shadow` - System credentials
- `/boot/` - Boot files
- `/.ssh/` - SSH keys
- `/proc/`, `/sys/`, `/dev/` - System interfaces

**Shell Commands:**
- Recursive deletes from root
- Disk formatting operations
- Fork bombs
- System shutdown/reboot
- Pipe to shell execution

**Git Operations:**
- Force push
- Hard reset
- Destructive branch operations
- Rebase force
- Filter branch

## Error Handling

All operations return `ToolResult` with status codes:

```python
result = registry.execute("filesystem", operation="read", path="/nonexistent")

# Check status
if result.status == ToolStatus.SUCCESS:
    print(result.output)
elif result.status == ToolStatus.ERROR:
    print(f"Error: {result.error}")
elif result.status == ToolStatus.BLOCKED:
    print(f"Blocked: {result.error}")
elif result.status == ToolStatus.TIMEOUT:
    print(f"Timeout: {result.error}")

# Simpler check
if result.success:
    print(result.output)
else:
    print(result.error)
```

## Logging

Tools use Python's logging framework:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Tool operations will log:
# - Tool initialization
# - Command execution
# - Security blocks
# - Errors and exceptions
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_tools.py
```

Tests cover:
- Registry creation and management
- All tool operations
- Security validation
- Error handling
- Statistics tracking
- Restricted registries

## Integration with Conch DNA

Tools integrate with other layers:

```python
from conch_dna.id import create_regulator
from conch_dna.tools import create_default_registry

# ID layer drives tool selection
regulator = create_regulator("balanced")
registry = create_default_registry()

# Use dominant need to guide tool usage
guidance = regulator.process_event("task_started")
if guidance["dominant_need"] == "reliability":
    # Use git for version control
    result = registry.execute("git", operation="status")
elif guidance["dominant_need"] == "curiosity":
    # Explore filesystem
    result = registry.execute("filesystem", operation="list", path=".")
```

## Performance Considerations

- **Timeout Settings**: Adjust based on expected operation duration
- **File Size Limits**: Configure max_read_size for large files
- **Parallel Execution**: Tools are stateless and thread-safe
- **Resource Cleanup**: Temporary files cleaned automatically

## Best Practices

1. **Always Check Results**: Verify `result.success` before using output
2. **Use Timeouts**: Set appropriate timeouts for long operations
3. **Restrict Access**: Use restricted registry in sensitive contexts
4. **Log Operations**: Enable logging for audit trails
5. **Handle Errors**: Implement proper error handling
6. **Test Security**: Verify blocked patterns match your requirements

## Future Enhancements

- Full KVRM integration when Superego layer complete
- Additional tools (HTTP, Database, etc.)
- Tool chaining and composition
- Async operation support
- Resource usage monitoring
- Audit logging to database

## License

Part of the Conch DNA consciousness system.

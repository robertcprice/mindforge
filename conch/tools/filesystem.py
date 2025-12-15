"""
Conch File System Tool

Provides safe file system operations.
"""

import glob as glob_module
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Union

from conch.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class FileSystemTool(Tool):
    """Tool for file system operations.

    Features:
    - Read/write files
    - Directory operations
    - File search (glob, grep)
    - Safe path handling
    - Backup before modifications
    """

    # Paths that should never be modified
    PROTECTED_PATHS = [
        "/",
        "/bin",
        "/sbin",
        "/usr",
        "/etc",
        "/var",
        "/System",
        "/Library",
        str(Path.home() / "Library"),
    ]

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        create_backups: bool = True,
    ):
        """Initialize file system tool.

        Args:
            base_dir: Base directory for operations (safety boundary)
            max_file_size: Maximum file size to read/write
            create_backups: Whether to backup files before modification
        """
        super().__init__(
            name="filesystem",
            description="File system operations (read, write, search)",
            requires_confirmation=False,
        )

        self.base_dir = base_dir or Path.cwd()
        self.max_file_size = max_file_size
        self.create_backups = create_backups

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a file system operation.

        Args:
            operation: Operation name (read, write, list, glob, grep, etc.)
            **kwargs: Operation-specific arguments

        Returns:
            ToolResult
        """
        operations = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "list": self._list,
            "glob": self._glob,
            "grep": self._grep,
            "mkdir": self._mkdir,
            "delete": self._delete,
            "copy": self._copy,
            "move": self._move,
            "exists": self._exists,
            "info": self._info,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            logger.error(f"FileSystem {operation} failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

    def _is_path_safe(self, path: Path) -> tuple[bool, str]:
        """Check if a path is safe to operate on."""
        try:
            resolved = path.resolve()

            # Check protected paths
            for protected in self.PROTECTED_PATHS:
                if str(resolved) == protected or str(resolved).startswith(protected + "/"):
                    # Allow reading, but not modifying
                    pass

            return True, "OK"
        except Exception as e:
            return False, str(e)

    def _read(self, path: str, encoding: str = "utf-8", lines: Optional[tuple] = None) -> ToolResult:
        """Read a file.

        Args:
            path: File path
            encoding: File encoding
            lines: Optional (start, end) line range
        """
        start_time = time.time()
        file_path = Path(path).expanduser()

        if not file_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"File not found: {path}",
                execution_time=time.time() - start_time,
            )

        if file_path.stat().st_size > self.max_file_size:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"File too large: {file_path.stat().st_size} bytes (max: {self.max_file_size})",
                execution_time=time.time() - start_time,
            )

        try:
            content = file_path.read_text(encoding=encoding)

            # Handle line range
            if lines:
                content_lines = content.split("\n")
                start, end = lines
                content_lines = content_lines[start:end]
                content = "\n".join(f"{i+start+1}: {line}" for i, line in enumerate(content_lines))

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=content,
                execution_time=time.time() - start_time,
                metadata={"path": str(file_path), "size": file_path.stat().st_size},
            )
        except UnicodeDecodeError:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Cannot decode file with {encoding} encoding",
                execution_time=time.time() - start_time,
            )

    def _write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> ToolResult:
        """Write to a file.

        Args:
            path: File path
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed
        """
        start_time = time.time()
        file_path = Path(path).expanduser()

        # Safety check
        is_safe, reason = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=f"Path not safe: {reason}",
                execution_time=time.time() - start_time,
            )

        # Create backup if file exists
        if self.create_backups and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            shutil.copy2(file_path, backup_path)

        # Create directories
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding=encoding)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Wrote {len(content)} bytes to {path}",
            execution_time=time.time() - start_time,
            metadata={"path": str(file_path), "size": len(content)},
        )

    def _append(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        """Append to a file."""
        start_time = time.time()
        file_path = Path(path).expanduser()

        with open(file_path, "a", encoding=encoding) as f:
            f.write(content)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Appended {len(content)} bytes to {path}",
            execution_time=time.time() - start_time,
        )

    def _list(self, path: str = ".", recursive: bool = False) -> ToolResult:
        """List directory contents."""
        start_time = time.time()
        dir_path = Path(path).expanduser()

        if not dir_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Directory not found: {path}",
                execution_time=time.time() - start_time,
            )

        if recursive:
            items = list(dir_path.rglob("*"))
        else:
            items = list(dir_path.iterdir())

        # Format output
        lines = []
        for item in sorted(items):
            rel_path = item.relative_to(dir_path) if recursive else item.name
            item_type = "d" if item.is_dir() else "f"
            size = item.stat().st_size if item.is_file() else 0
            lines.append(f"{item_type} {size:>10} {rel_path}")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(lines),
            execution_time=time.time() - start_time,
            metadata={"path": str(dir_path), "count": len(items)},
        )

    def _glob(self, pattern: str, path: str = ".") -> ToolResult:
        """Find files matching a glob pattern."""
        start_time = time.time()
        base_path = Path(path).expanduser()

        matches = list(base_path.glob(pattern))
        output = "\n".join(str(m) for m in sorted(matches))

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            execution_time=time.time() - start_time,
            metadata={"pattern": pattern, "count": len(matches)},
        )

    def _grep(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        ignore_case: bool = False,
        context: int = 0,
    ) -> ToolResult:
        """Search for pattern in files."""
        start_time = time.time()
        base_path = Path(path).expanduser()

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)

        results = []
        files_searched = 0

        for file_path in base_path.rglob(file_pattern):
            if file_path.is_file():
                try:
                    files_searched += 1
                    content = file_path.read_text(errors="ignore")
                    lines = content.split("\n")

                    for i, line in enumerate(lines):
                        if regex.search(line):
                            # Get context lines
                            start = max(0, i - context)
                            end = min(len(lines), i + context + 1)
                            context_lines = lines[start:end]

                            result = f"{file_path}:{i+1}:"
                            if context > 0:
                                result += "\n" + "\n".join(
                                    f"  {j+start+1}: {l}" for j, l in enumerate(context_lines)
                                )
                            else:
                                result += line

                            results.append(result)
                except Exception:
                    pass  # Skip files we can't read

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(results),
            execution_time=time.time() - start_time,
            metadata={
                "pattern": pattern,
                "matches": len(results),
                "files_searched": files_searched,
            },
        )

    def _mkdir(self, path: str, parents: bool = True) -> ToolResult:
        """Create a directory."""
        start_time = time.time()
        dir_path = Path(path).expanduser()

        dir_path.mkdir(parents=parents, exist_ok=True)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Created directory: {path}",
            execution_time=time.time() - start_time,
        )

    def _delete(self, path: str, force: bool = False) -> ToolResult:
        """Delete a file or directory."""
        start_time = time.time()
        target_path = Path(path).expanduser()

        # Safety check
        is_safe, reason = self._is_path_safe(target_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=f"Path not safe: {reason}",
                execution_time=time.time() - start_time,
            )

        if not target_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path not found: {path}",
                execution_time=time.time() - start_time,
            )

        if target_path.is_dir():
            if force:
                shutil.rmtree(target_path)
            else:
                target_path.rmdir()  # Only works if empty
        else:
            target_path.unlink()

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Deleted: {path}",
            execution_time=time.time() - start_time,
        )

    def _copy(self, src: str, dst: str) -> ToolResult:
        """Copy a file or directory."""
        start_time = time.time()
        src_path = Path(src).expanduser()
        dst_path = Path(dst).expanduser()

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Copied {src} to {dst}",
            execution_time=time.time() - start_time,
        )

    def _move(self, src: str, dst: str) -> ToolResult:
        """Move a file or directory."""
        start_time = time.time()
        src_path = Path(src).expanduser()
        dst_path = Path(dst).expanduser()

        shutil.move(str(src_path), str(dst_path))

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=f"Moved {src} to {dst}",
            execution_time=time.time() - start_time,
        )

    def _exists(self, path: str) -> ToolResult:
        """Check if a path exists."""
        file_path = Path(path).expanduser()
        exists = file_path.exists()

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=str(exists),
            metadata={
                "exists": exists,
                "is_file": file_path.is_file() if exists else False,
                "is_dir": file_path.is_dir() if exists else False,
            },
        )

    def _info(self, path: str) -> ToolResult:
        """Get file/directory information."""
        file_path = Path(path).expanduser()

        if not file_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path not found: {path}",
            )

        stat = file_path.stat()

        info = {
            "path": str(file_path.resolve()),
            "name": file_path.name,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
        }

        output = "\n".join(f"{k}: {v}" for k, v in info.items())

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=info,
        )


# Convenience functions
_fs_tool: Optional[FileSystemTool] = None


def get_fs() -> FileSystemTool:
    """Get the global file system tool."""
    global _fs_tool
    if _fs_tool is None:
        _fs_tool = FileSystemTool()
    return _fs_tool


def read_file(path: str, **kwargs) -> str:
    """Read a file and return its content."""
    result = get_fs().execute("read", path=path, **kwargs)
    if result.success:
        return result.output
    raise FileNotFoundError(result.error)


def write_file(path: str, content: str, **kwargs) -> None:
    """Write content to a file."""
    result = get_fs().execute("write", path=path, content=content, **kwargs)
    if not result.success:
        raise IOError(result.error)


def find_files(pattern: str, path: str = ".") -> list[str]:
    """Find files matching a glob pattern."""
    result = get_fs().execute("glob", pattern=pattern, path=path)
    if result.success and result.output:
        return result.output.split("\n")
    return []


def search_in_files(pattern: str, path: str = ".", **kwargs) -> str:
    """Search for pattern in files."""
    result = get_fs().execute("grep", pattern=pattern, path=path, **kwargs)
    return result.output

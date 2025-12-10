"""
MindForge Git Tool

Provides Git operations with safety guardrails.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from mindforge.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class GitTool(Tool):
    """Tool for Git operations.

    Features:
    - Repository status and info
    - Safe commit operations
    - Branch management
    - Diff viewing
    - Log browsing
    - Safety guardrails for destructive operations
    """

    # Dangerous git operations that require confirmation
    DANGEROUS_OPERATIONS = [
        "push --force",
        "push -f",
        "reset --hard",
        "clean -fd",
        "branch -D",
        "rebase",
        "merge --abort",
        "checkout --",
        "stash drop",
        "reflog expire",
        "gc --prune",
    ]

    def __init__(
        self,
        repo_path: Optional[Path] = None,
        timeout: int = 60,
    ):
        """Initialize git tool.

        Args:
            repo_path: Path to git repository (defaults to cwd)
            timeout: Command timeout in seconds
        """
        super().__init__(
            name="git",
            description="Git operations (status, commit, branch, diff)",
            requires_confirmation=False,  # Per-operation confirmation
        )

        self.repo_path = repo_path or Path.cwd()
        self.timeout = timeout

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a git operation.

        Args:
            operation: Operation name
            **kwargs: Operation-specific arguments

        Returns:
            ToolResult
        """
        operations = {
            "status": self._status,
            "diff": self._diff,
            "log": self._log,
            "branch": self._branch,
            "commit": self._commit,
            "add": self._add,
            "checkout": self._checkout,
            "pull": self._pull,
            "push": self._push,
            "stash": self._stash,
            "show": self._show,
            "blame": self._blame,
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
            logger.error(f"Git {operation} failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

    def _run_git(self, args: list[str], check: bool = True) -> tuple[int, str, str]:
        """Run a git command.

        Args:
            args: Git command arguments
            check: Whether to raise on non-zero exit

        Returns:
            (return_code, stdout, stderr)
        """
        cmd = ["git"] + args

        # Safety check
        cmd_str = " ".join(args)
        for dangerous in self.DANGEROUS_OPERATIONS:
            if dangerous in cmd_str:
                raise ValueError(f"Dangerous operation blocked: {dangerous}")

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        return result.returncode, result.stdout, result.stderr

    def _is_git_repo(self) -> bool:
        """Check if current path is a git repository."""
        try:
            code, _, _ = self._run_git(["rev-parse", "--git-dir"])
            return code == 0
        except Exception:
            return False

    def _status(self, short: bool = False) -> ToolResult:
        """Get repository status.

        Args:
            short: Use short format
        """
        start_time = time.time()

        if not self._is_git_repo():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Not a git repository",
                execution_time=time.time() - start_time,
            )

        args = ["status"]
        if short:
            args.append("-s")

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
            metadata={"repo": str(self.repo_path)},
        )

    def _diff(
        self,
        staged: bool = False,
        path: Optional[str] = None,
        commit: Optional[str] = None,
    ) -> ToolResult:
        """Show differences.

        Args:
            staged: Show staged changes
            path: Specific file path
            commit: Specific commit to diff against
        """
        start_time = time.time()

        args = ["diff"]
        if staged:
            args.append("--staged")
        if commit:
            args.append(commit)
        if path:
            args.extend(["--", path])

        code, stdout, stderr = self._run_git(args)

        if not stdout:
            stdout = "No differences found."

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _log(
        self,
        count: int = 10,
        oneline: bool = True,
        path: Optional[str] = None,
    ) -> ToolResult:
        """Show commit log.

        Args:
            count: Number of commits to show
            oneline: Use one-line format
            path: Show log for specific path
        """
        start_time = time.time()

        args = ["log", f"-{count}"]
        if oneline:
            args.append("--oneline")
        else:
            args.append("--pretty=format:%h %ad | %s [%an]")
            args.append("--date=short")
        if path:
            args.extend(["--", path])

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
            metadata={"count": count},
        )

    def _branch(
        self,
        list_all: bool = True,
        create: Optional[str] = None,
        delete: Optional[str] = None,
    ) -> ToolResult:
        """Manage branches.

        Args:
            list_all: List all branches
            create: Create a new branch
            delete: Delete a branch (safe delete only)
        """
        start_time = time.time()

        if create:
            args = ["branch", create]
        elif delete:
            args = ["branch", "-d", delete]  # Safe delete only
        else:
            args = ["branch"]
            if list_all:
                args.append("-a")

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _commit(self, message: str, add_all: bool = False) -> ToolResult:
        """Create a commit.

        Args:
            message: Commit message
            add_all: Add all changes before committing
        """
        start_time = time.time()

        if add_all:
            code, stdout, stderr = self._run_git(["add", "-A"])
            if code != 0:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"Failed to add files: {stderr}",
                    execution_time=time.time() - start_time,
                )

        # Format commit message
        formatted_message = message.strip()
        if not formatted_message:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Commit message cannot be empty",
                execution_time=time.time() - start_time,
            )

        code, stdout, stderr = self._run_git(["commit", "-m", formatted_message])

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
            metadata={"message": formatted_message},
        )

    def _add(self, path: str = ".", all_changes: bool = False) -> ToolResult:
        """Stage files for commit.

        Args:
            path: Path to add
            all_changes: Add all changes including deletions
        """
        start_time = time.time()

        if all_changes:
            args = ["add", "-A"]
        else:
            args = ["add", path]

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=f"Added: {path}" if code == 0 else "",
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _checkout(self, branch: str, create: bool = False) -> ToolResult:
        """Switch branches.

        Args:
            branch: Branch name
            create: Create branch if it doesn't exist
        """
        start_time = time.time()

        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(branch)

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout or f"Switched to branch '{branch}'",
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _pull(self, remote: str = "origin", branch: Optional[str] = None) -> ToolResult:
        """Pull from remote.

        Args:
            remote: Remote name
            branch: Branch name
        """
        start_time = time.time()

        args = ["pull", remote]
        if branch:
            args.append(branch)

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _push(self, remote: str = "origin", branch: Optional[str] = None) -> ToolResult:
        """Push to remote (safe push only).

        Args:
            remote: Remote name
            branch: Branch name
        """
        start_time = time.time()

        args = ["push", remote]
        if branch:
            args.append(branch)

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _stash(
        self,
        action: str = "list",
        message: Optional[str] = None,
    ) -> ToolResult:
        """Manage stash.

        Args:
            action: Stash action (list, push, pop, apply)
            message: Message for stash push
        """
        start_time = time.time()

        if action == "list":
            args = ["stash", "list"]
        elif action == "push":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
        elif action == "pop":
            args = ["stash", "pop"]
        elif action == "apply":
            args = ["stash", "apply"]
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unknown stash action: {action}. Use: list, push, pop, apply",
                execution_time=time.time() - start_time,
            )

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout or f"Stash {action} completed",
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _show(self, commit: str = "HEAD", stat: bool = False) -> ToolResult:
        """Show commit details.

        Args:
            commit: Commit reference
            stat: Show stat instead of full diff
        """
        start_time = time.time()

        args = ["show", commit]
        if stat:
            args.append("--stat")

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _blame(self, path: str, lines: Optional[tuple] = None) -> ToolResult:
        """Show file blame.

        Args:
            path: File path
            lines: Optional (start, end) line range
        """
        start_time = time.time()

        args = ["blame", path]
        if lines:
            args.extend(["-L", f"{lines[0]},{lines[1]}"])

        code, stdout, stderr = self._run_git(args)

        return ToolResult(
            status=ToolStatus.SUCCESS if code == 0 else ToolStatus.ERROR,
            output=stdout,
            error=stderr if code != 0 else None,
            execution_time=time.time() - start_time,
        )

    def _info(self) -> ToolResult:
        """Get repository information."""
        start_time = time.time()

        if not self._is_git_repo():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Not a git repository",
                execution_time=time.time() - start_time,
            )

        info = {"repo_path": str(self.repo_path)}

        # Get current branch
        code, stdout, _ = self._run_git(["branch", "--show-current"])
        if code == 0:
            info["current_branch"] = stdout.strip()

        # Get remote URL
        code, stdout, _ = self._run_git(["remote", "get-url", "origin"])
        if code == 0:
            info["remote_url"] = stdout.strip()

        # Get latest commit
        code, stdout, _ = self._run_git(["log", "-1", "--pretty=format:%h %s"])
        if code == 0:
            info["latest_commit"] = stdout.strip()

        # Get uncommitted changes count
        code, stdout, _ = self._run_git(["status", "--porcelain"])
        if code == 0:
            changes = [l for l in stdout.strip().split("\n") if l]
            info["uncommitted_changes"] = len(changes)

        # Format output
        output_lines = [
            f"Repository: {info['repo_path']}",
            f"Branch: {info.get('current_branch', 'N/A')}",
            f"Remote: {info.get('remote_url', 'N/A')}",
            f"Latest Commit: {info.get('latest_commit', 'N/A')}",
            f"Uncommitted Changes: {info.get('uncommitted_changes', 0)}",
        ]

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_lines),
            execution_time=time.time() - start_time,
            metadata=info,
        )


# Convenience functions
_git_tool: Optional[GitTool] = None


def get_git(repo_path: Optional[Path] = None) -> GitTool:
    """Get the global git tool instance."""
    global _git_tool
    if _git_tool is None or repo_path:
        _git_tool = GitTool(repo_path=repo_path)
    return _git_tool


def git_status() -> str:
    """Get git status."""
    result = get_git().execute("status")
    return result.output


def git_commit(message: str, add_all: bool = False) -> bool:
    """Create a git commit."""
    result = get_git().execute("commit", message=message, add_all=add_all)
    return result.success


def git_diff(staged: bool = False) -> str:
    """Get git diff."""
    result = get_git().execute("diff", staged=staged)
    return result.output

"""
Conch Code Tool

Provides code analysis and editing capabilities.
"""

import ast
import difflib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from conch.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


@dataclass
class CodeLocation:
    """Location in source code."""
    file: str
    line: int
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class CodeSymbol:
    """A symbol in source code."""
    name: str
    kind: str  # function, class, variable, import
    location: CodeLocation
    docstring: Optional[str] = None
    signature: Optional[str] = None


class CodeTool(Tool):
    """Tool for code analysis and editing.

    Features:
    - Code parsing and analysis
    - Symbol extraction
    - Diff generation
    - Safe code editing
    - Syntax validation
    """

    # Supported languages for analysis
    SUPPORTED_LANGUAGES = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
    }

    def __init__(self, max_file_size: int = 5 * 1024 * 1024):  # 5MB
        """Initialize code tool.

        Args:
            max_file_size: Maximum file size to process
        """
        super().__init__(
            name="code",
            description="Code analysis and editing operations",
            requires_confirmation=False,
        )

        self.max_file_size = max_file_size

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a code operation.

        Args:
            operation: Operation name
            **kwargs: Operation-specific arguments

        Returns:
            ToolResult
        """
        operations = {
            "analyze": self._analyze,
            "symbols": self._extract_symbols,
            "diff": self._generate_diff,
            "edit": self._edit,
            "validate": self._validate,
            "format": self._format_code,
            "search": self._search_code,
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
            logger.error(f"Code {operation} failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

    def _get_language(self, path: str) -> str:
        """Get language from file extension."""
        ext = Path(path).suffix.lower()
        return self.SUPPORTED_LANGUAGES.get(ext, "unknown")

    def _analyze(self, path: str) -> ToolResult:
        """Analyze a source code file.

        Args:
            path: Path to the file
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
                error=f"File too large: {file_path.stat().st_size} bytes",
                execution_time=time.time() - start_time,
            )

        try:
            content = file_path.read_text()
            language = self._get_language(path)

            analysis = {
                "file": str(file_path),
                "language": language,
                "lines": len(content.split("\n")),
                "characters": len(content),
                "size_bytes": file_path.stat().st_size,
            }

            # Python-specific analysis
            if language == "python":
                try:
                    tree = ast.parse(content)
                    analysis["imports"] = []
                    analysis["functions"] = []
                    analysis["classes"] = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                analysis["imports"].append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for alias in node.names:
                                analysis["imports"].append(f"{module}.{alias.name}")
                        elif isinstance(node, ast.FunctionDef):
                            analysis["functions"].append({
                                "name": node.name,
                                "line": node.lineno,
                                "args": len(node.args.args),
                            })
                        elif isinstance(node, ast.ClassDef):
                            analysis["classes"].append({
                                "name": node.name,
                                "line": node.lineno,
                                "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                            })

                    analysis["syntax_valid"] = True
                except SyntaxError as e:
                    analysis["syntax_valid"] = False
                    analysis["syntax_error"] = str(e)

            # Format output
            output_lines = [
                f"File: {analysis['file']}",
                f"Language: {analysis['language']}",
                f"Lines: {analysis['lines']}",
                f"Characters: {analysis['characters']}",
            ]

            if language == "python":
                output_lines.append(f"Syntax Valid: {analysis.get('syntax_valid', 'N/A')}")
                if analysis.get("imports"):
                    output_lines.append(f"Imports ({len(analysis['imports'])}): {', '.join(analysis['imports'][:10])}")
                if analysis.get("functions"):
                    output_lines.append(f"Functions ({len(analysis['functions'])}): {', '.join(f['name'] for f in analysis['functions'][:10])}")
                if analysis.get("classes"):
                    output_lines.append(f"Classes ({len(analysis['classes'])}): {', '.join(c['name'] for c in analysis['classes'][:10])}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                execution_time=time.time() - start_time,
                metadata=analysis,
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _extract_symbols(self, path: str) -> ToolResult:
        """Extract symbols from a source file.

        Args:
            path: Path to the file
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

        try:
            content = file_path.read_text()
            language = self._get_language(path)
            symbols = []

            if language == "python":
                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            docstring = ast.get_docstring(node)
                            args = [arg.arg for arg in node.args.args]
                            symbols.append(CodeSymbol(
                                name=node.name,
                                kind="function",
                                location=CodeLocation(
                                    file=str(file_path),
                                    line=node.lineno,
                                    end_line=node.end_lineno,
                                ),
                                docstring=docstring,
                                signature=f"def {node.name}({', '.join(args)})",
                            ))
                        elif isinstance(node, ast.ClassDef):
                            docstring = ast.get_docstring(node)
                            symbols.append(CodeSymbol(
                                name=node.name,
                                kind="class",
                                location=CodeLocation(
                                    file=str(file_path),
                                    line=node.lineno,
                                    end_line=node.end_lineno,
                                ),
                                docstring=docstring,
                                signature=f"class {node.name}",
                            ))
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    symbols.append(CodeSymbol(
                                        name=target.id,
                                        kind="variable",
                                        location=CodeLocation(
                                            file=str(file_path),
                                            line=node.lineno,
                                        ),
                                    ))

                except SyntaxError:
                    pass

            # Format output
            output_lines = [f"Symbols in {path}:", ""]
            for sym in symbols:
                output_lines.append(f"  {sym.kind}: {sym.name} (line {sym.location.line})")
                if sym.signature:
                    output_lines.append(f"    {sym.signature}")
                if sym.docstring:
                    output_lines.append(f"    \"{sym.docstring[:50]}...\"")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                execution_time=time.time() - start_time,
                metadata={
                    "file": str(file_path),
                    "symbol_count": len(symbols),
                    "symbols": [{"name": s.name, "kind": s.kind, "line": s.location.line} for s in symbols],
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _generate_diff(
        self,
        old_content: str,
        new_content: str,
        filename: str = "file",
    ) -> ToolResult:
        """Generate a unified diff between two strings.

        Args:
            old_content: Original content
            new_content: Modified content
            filename: Filename for diff header
        """
        start_time = time.time()

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )

        diff_text = "".join(diff)

        if not diff_text:
            diff_text = "No differences found."

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=diff_text,
            execution_time=time.time() - start_time,
            metadata={
                "old_lines": len(old_lines),
                "new_lines": len(new_lines),
                "has_changes": bool(diff_text and diff_text != "No differences found."),
            },
        )

    def _edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        create_backup: bool = True,
    ) -> ToolResult:
        """Edit a file by replacing a string.

        Args:
            path: Path to the file
            old_string: String to find and replace
            new_string: Replacement string
            create_backup: Whether to create a backup
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

        try:
            content = file_path.read_text()

            # Check if old_string exists
            if old_string not in content:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error="old_string not found in file",
                    execution_time=time.time() - start_time,
                )

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"old_string found {count} times - must be unique. Provide more context.",
                    execution_time=time.time() - start_time,
                )

            # Create backup
            if create_backup:
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                backup_path.write_text(content)

            # Perform replacement
            new_content = content.replace(old_string, new_string)
            file_path.write_text(new_content)

            # Generate diff for output
            diff_result = self._generate_diff(content, new_content, file_path.name)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Successfully edited {path}\n\n{diff_result.output}",
                execution_time=time.time() - start_time,
                metadata={
                    "file": str(file_path),
                    "backup_created": create_backup,
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _validate(self, path: str) -> ToolResult:
        """Validate source code syntax.

        Args:
            path: Path to the file
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

        try:
            content = file_path.read_text()
            language = self._get_language(path)

            if language == "python":
                try:
                    ast.parse(content)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Syntax valid: {path}",
                        execution_time=time.time() - start_time,
                        metadata={"valid": True, "language": language},
                    )
                except SyntaxError as e:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=f"Syntax error at line {e.lineno}: {e.msg}",
                        execution_time=time.time() - start_time,
                        metadata={
                            "valid": False,
                            "language": language,
                            "line": e.lineno,
                            "message": e.msg,
                        },
                    )

            elif language == "json":
                import json as json_module
                try:
                    json_module.loads(content)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"JSON valid: {path}",
                        execution_time=time.time() - start_time,
                        metadata={"valid": True, "language": language},
                    )
                except json_module.JSONDecodeError as e:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=f"JSON error at line {e.lineno}: {e.msg}",
                        execution_time=time.time() - start_time,
                        metadata={"valid": False, "language": language},
                    )

            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"No validator available for {language}",
                    execution_time=time.time() - start_time,
                    metadata={"valid": None, "language": language},
                )

        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _format_code(self, content: str, language: str = "python") -> ToolResult:
        """Format code (basic implementation).

        Args:
            content: Code content to format
            language: Programming language
        """
        start_time = time.time()

        if language == "python":
            # Basic Python formatting
            lines = content.split("\n")
            formatted_lines = []

            for line in lines:
                # Remove trailing whitespace
                line = line.rstrip()
                formatted_lines.append(line)

            formatted = "\n".join(formatted_lines)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=formatted,
                execution_time=time.time() - start_time,
                metadata={
                    "language": language,
                    "original_length": len(content),
                    "formatted_length": len(formatted),
                },
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=content,  # No formatting available
            execution_time=time.time() - start_time,
            metadata={"language": language, "formatted": False},
        )

    def _search_code(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*.py",
        context: int = 2,
    ) -> ToolResult:
        """Search for pattern in code files.

        Args:
            pattern: Regex pattern to search
            path: Directory to search in
            file_pattern: File glob pattern
            context: Lines of context around matches
        """
        start_time = time.time()
        base_path = Path(path).expanduser()

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Invalid regex pattern: {e}",
                execution_time=time.time() - start_time,
            )

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
                            start = max(0, i - context)
                            end = min(len(lines), i + context + 1)
                            context_lines = lines[start:end]

                            result = {
                                "file": str(file_path),
                                "line": i + 1,
                                "match": line.strip(),
                                "context": "\n".join(
                                    f"{j+start+1}: {l}" for j, l in enumerate(context_lines)
                                ),
                            }
                            results.append(result)

                except Exception:
                    pass

        # Format output
        output_lines = [f"Search: {pattern}", f"Files searched: {files_searched}", ""]

        for r in results[:50]:  # Limit output
            output_lines.append(f"{r['file']}:{r['line']}")
            output_lines.append(r["context"])
            output_lines.append("")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_lines),
            execution_time=time.time() - start_time,
            metadata={
                "pattern": pattern,
                "matches": len(results),
                "files_searched": files_searched,
            },
        )


# Convenience functions
_code_tool: Optional[CodeTool] = None


def get_code() -> CodeTool:
    """Get the global code tool instance."""
    global _code_tool
    if _code_tool is None:
        _code_tool = CodeTool()
    return _code_tool


def analyze_file(path: str) -> dict:
    """Analyze a source file."""
    result = get_code().execute("analyze", path=path)
    return result.metadata if result.success else {}


def edit_file(path: str, old_string: str, new_string: str) -> bool:
    """Edit a file by replacing a string."""
    result = get_code().execute("edit", path=path, old_string=old_string, new_string=new_string)
    return result.success


def validate_syntax(path: str) -> bool:
    """Validate syntax of a source file."""
    result = get_code().execute("validate", path=path)
    return result.success

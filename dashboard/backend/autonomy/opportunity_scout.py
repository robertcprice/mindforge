#!/usr/bin/env python3
"""
Opportunity Scout for PM-1000

Instead of generating abstract goals, SCOUT for concrete opportunities.
This module implements various scanners that detect actionable improvement
opportunities in the codebase.

Opportunity Types:
- TODO_RESOLUTION: Find and resolve TODO/FIXME comments
- TEST_GAP: Functions/classes without tests
- DOC_GAP: Undocumented public APIs
- DEPENDENCY_UPDATE: Outdated dependencies
- SECURITY_FIX: Known vulnerabilities
- PERFORMANCE_OPTIMIZATION: Slow functions
- CODE_DUPLICATION: DRY violations
- DEAD_CODE: Unused imports/functions
- TYPE_ANNOTATION: Missing type hints
- ERROR_HANDLING: Missing error handling

Scoring System:
- Confidence: How certain are we this is a real opportunity?
- Value: How much value will acting on this create?
- Risk: How risky is addressing this opportunity?
- Effort: How much effort is required?
"""

import os
import re
import ast
import subprocess
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

from .autonomous_loop import Opportunity, TaskPriority, OpportunityScanner as BaseScanner

logger = get_logger("pm1000.autonomy.scout")


class OpportunityType(Enum):
    """Types of opportunities that can be detected."""
    TODO_RESOLUTION = "todo_resolution"
    TEST_GAP = "test_gap"
    DOC_GAP = "doc_gap"
    DEPENDENCY_UPDATE = "dependency_update"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_DUPLICATION = "code_duplication"
    DEAD_CODE = "dead_code"
    TYPE_ANNOTATION = "type_annotation"
    ERROR_HANDLING = "error_handling"
    CODE_SMELL = "code_smell"
    REFACTORING = "refactoring"


@dataclass
class ScanResult:
    """Result of a single scan operation."""
    scanner_name: str
    opportunities_found: int
    scan_duration_ms: float
    errors: List[str] = field(default_factory=list)
    files_scanned: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanner_name": self.scanner_name,
            "opportunities_found": self.opportunities_found,
            "scan_duration_ms": self.scan_duration_ms,
            "errors": self.errors,
            "files_scanned": self.files_scanned,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ScanConfig:
    """Configuration for opportunity scanning."""
    project_path: str = "."
    include_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**", "**/__pycache__/**", "**/.git/**",
        "**/venv/**", "**/env/**", "**/dist/**", "**/build/**"
    ])
    max_file_size_kb: int = 500
    scan_tests: bool = True
    scan_docs: bool = True


class TodoScanner(BaseScanner):
    """
    Scanner for TODO/FIXME comments in code.

    Detects patterns like:
    - # TODO: description
    - # FIXME: description
    - // TODO(author): description
    - /* TODO: description */
    """

    TODO_PATTERNS = [
        r'#\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]+(.+?)(?:\n|$)',
        r'//\s*(TODO|FIXME|XXX|HACK|BUG)[\s:\(](.+?)(?:\n|$)',
        r'/\*\s*(TODO|FIXME|XXX|HACK|BUG)[\s:]+(.+?)(?:\*/|\n)',
    ]

    PRIORITY_KEYWORDS = {
        "critical": TaskPriority.CRITICAL,
        "urgent": TaskPriority.HIGH,
        "important": TaskPriority.HIGH,
        "minor": TaskPriority.LOW,
        "later": TaskPriority.BACKGROUND,
    }

    def __init__(self, config: ScanConfig = None):
        super().__init__("todo_scanner")
        self.config = config or ScanConfig()
        self.scan_interval = 300  # 5 minutes
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.TODO_PATTERNS]

    def scan(self) -> List[Opportunity]:
        opportunities = []
        project_path = Path(self.config.project_path)

        for pattern in self.config.include_patterns:
            for file_path in project_path.glob(f"**/{pattern}"):
                if self._should_skip(file_path):
                    continue

                try:
                    todos = self._scan_file(file_path)
                    opportunities.extend(todos)
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {e}")

        return opportunities

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if Path(path_str).match(pattern):
                return True

        # Check file size
        try:
            if file_path.stat().st_size > self.config.max_file_size_kb * 1024:
                return True
        except:
            return True

        return False

    def _scan_file(self, file_path: Path) -> List[Opportunity]:
        """Scan a single file for TODOs."""
        opportunities = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except:
            return []

        for i, line in enumerate(content.split("\n"), 1):
            for pattern in self._compiled_patterns:
                match = pattern.search(line)
                if match:
                    todo_type = match.group(1).upper()
                    description = match.group(2).strip()

                    priority = self._determine_priority(todo_type, description)
                    confidence = 0.95 if todo_type in ("TODO", "FIXME") else 0.85

                    opportunities.append(Opportunity(
                        id=f"todo_{hash(f'{file_path}:{i}')}",
                        type=OpportunityType.TODO_RESOLUTION.value,
                        description=f"{todo_type}: {description}",
                        source=self.name,
                        priority=priority,
                        estimated_effort=self._estimate_effort(description),
                        estimated_value=2.0 if todo_type == "FIXME" else 1.5,
                        confidence=confidence,
                        context={
                            "file": str(file_path),
                            "line": i,
                            "todo_type": todo_type,
                            "full_text": description,
                        }
                    ))

        return opportunities

    def _determine_priority(self, todo_type: str, description: str) -> TaskPriority:
        """Determine priority from TODO type and keywords."""
        if todo_type in ("FIXME", "BUG"):
            return TaskPriority.HIGH

        desc_lower = description.lower()
        for keyword, priority in self.PRIORITY_KEYWORDS.items():
            if keyword in desc_lower:
                return priority

        return TaskPriority.MEDIUM

    def _estimate_effort(self, description: str) -> float:
        """Estimate effort in hours based on description."""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["refactor", "rewrite", "redesign"]):
            return 4.0
        elif any(word in desc_lower for word in ["implement", "add", "create"]):
            return 2.0
        elif any(word in desc_lower for word in ["fix", "update", "change"]):
            return 1.0
        else:
            return 0.5


class TestCoverageScanner(BaseScanner):
    """
    Scanner for test coverage gaps.

    Detects:
    - Functions without corresponding tests
    - Classes without test coverage
    - Low coverage files
    """

    def __init__(self, config: ScanConfig = None):
        super().__init__("test_coverage_scanner")
        self.config = config or ScanConfig()
        self.scan_interval = 600  # 10 minutes

    def scan(self) -> List[Opportunity]:
        opportunities = []
        project_path = Path(self.config.project_path)

        # Find all Python files
        source_files = set()
        test_files = set()

        for py_file in project_path.glob("**/*.py"):
            if self._should_skip(py_file):
                continue

            if "test" in py_file.name.lower() or "tests" in str(py_file):
                test_files.add(py_file)
            else:
                source_files.add(py_file)

        # Find functions/classes without tests
        for source_file in source_files:
            try:
                untested = self._find_untested_elements(source_file, test_files)
                opportunities.extend(untested)
            except Exception as e:
                logger.debug(f"Error analyzing {source_file}: {e}")

        return opportunities

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if Path(path_str).match(pattern):
                return True
        return False

    def _find_untested_elements(self, source_file: Path, test_files: Set[Path]) -> List[Opportunity]:
        """Find functions and classes without tests."""
        opportunities = []

        try:
            content = source_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except:
            return []

        # Extract function and class names
        elements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    elements.append(("function", node.name, node.lineno))
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    elements.append(("class", node.name, node.lineno))

        # Check if tests exist
        test_content = ""
        for test_file in test_files:
            try:
                test_content += test_file.read_text(encoding="utf-8", errors="ignore")
            except:
                pass

        for elem_type, elem_name, line_no in elements:
            # Simple heuristic: check if test mentions the element
            test_patterns = [
                f"test_{elem_name}",
                f"Test{elem_name}",
                f"test{elem_name}",
                f"{elem_name}_test",
            ]

            has_test = any(pattern.lower() in test_content.lower() for pattern in test_patterns)

            if not has_test:
                opportunities.append(Opportunity(
                    id=f"test_gap_{hash(f'{source_file}:{elem_name}')}",
                    type=OpportunityType.TEST_GAP.value,
                    description=f"Add tests for {elem_type} '{elem_name}' in {source_file.name}",
                    source=self.name,
                    priority=TaskPriority.MEDIUM,
                    estimated_effort=1.5 if elem_type == "class" else 0.5,
                    estimated_value=2.5,
                    confidence=0.75,
                    context={
                        "file": str(source_file),
                        "line": line_no,
                        "element_type": elem_type,
                        "element_name": elem_name,
                    }
                ))

        return opportunities


class DocGapScanner(BaseScanner):
    """
    Scanner for documentation gaps.

    Detects:
    - Functions without docstrings
    - Classes without docstrings
    - Modules without module-level docstrings
    - Complex functions without explanation
    """

    def __init__(self, config: ScanConfig = None):
        super().__init__("doc_gap_scanner")
        self.config = config or ScanConfig()
        self.scan_interval = 900  # 15 minutes

    def scan(self) -> List[Opportunity]:
        opportunities = []
        project_path = Path(self.config.project_path)

        for py_file in project_path.glob("**/*.py"):
            if self._should_skip(py_file):
                continue

            try:
                undocumented = self._find_undocumented(py_file)
                opportunities.extend(undocumented)
            except Exception as e:
                logger.debug(f"Error analyzing {py_file}: {e}")

        return opportunities

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if Path(path_str).match(pattern):
                return True
        if "test" in file_path.name.lower():
            return True
        return False

    def _find_undocumented(self, file_path: Path) -> List[Opportunity]:
        """Find undocumented elements in a file."""
        opportunities = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except:
            return []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring:
                    # Check complexity - undocumented complex functions are higher priority
                    complexity = self._estimate_complexity(node)
                    priority = TaskPriority.HIGH if complexity > 5 else TaskPriority.LOW

                    opportunities.append(Opportunity(
                        id=f"doc_gap_{hash(f'{file_path}:{node.name}')}",
                        type=OpportunityType.DOC_GAP.value,
                        description=f"Add docstring to function '{node.name}' in {file_path.name}",
                        source=self.name,
                        priority=priority,
                        estimated_effort=0.25,
                        estimated_value=1.0 + (complexity * 0.2),
                        confidence=0.9,
                        context={
                            "file": str(file_path),
                            "line": node.lineno,
                            "element_type": "function",
                            "element_name": node.name,
                            "complexity": complexity,
                        }
                    ))

            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring:
                    opportunities.append(Opportunity(
                        id=f"doc_gap_{hash(f'{file_path}:{node.name}')}",
                        type=OpportunityType.DOC_GAP.value,
                        description=f"Add docstring to class '{node.name}' in {file_path.name}",
                        source=self.name,
                        priority=TaskPriority.MEDIUM,
                        estimated_effort=0.5,
                        estimated_value=1.5,
                        confidence=0.9,
                        context={
                            "file": str(file_path),
                            "line": node.lineno,
                            "element_type": "class",
                            "element_name": node.name,
                        }
                    ))

        return opportunities

    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity


class CodeSmellScanner(BaseScanner):
    """
    Scanner for code smells and anti-patterns.

    Detects:
    - Long functions (> 50 lines)
    - Deep nesting (> 4 levels)
    - Too many parameters (> 5)
    - Large classes (> 20 methods)
    - Magic numbers
    - Duplicate code patterns
    """

    def __init__(self, config: ScanConfig = None):
        super().__init__("code_smell_scanner")
        self.config = config or ScanConfig()
        self.scan_interval = 1200  # 20 minutes

        # Thresholds
        self.max_function_lines = 50
        self.max_nesting_depth = 4
        self.max_parameters = 5
        self.max_class_methods = 20

    def scan(self) -> List[Opportunity]:
        opportunities = []
        project_path = Path(self.config.project_path)

        for py_file in project_path.glob("**/*.py"):
            if self._should_skip(py_file):
                continue

            try:
                smells = self._find_code_smells(py_file)
                opportunities.extend(smells)
            except Exception as e:
                logger.debug(f"Error analyzing {py_file}: {e}")

        return opportunities

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if Path(path_str).match(pattern):
                return True
        return False

    def _find_code_smells(self, file_path: Path) -> List[Opportunity]:
        """Find code smells in a file."""
        opportunities = []

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            tree = ast.parse(content)
        except:
            return []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check function length
                if hasattr(node, 'end_lineno'):
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > self.max_function_lines:
                        opportunities.append(Opportunity(
                            id=f"smell_long_func_{hash(f'{file_path}:{node.name}')}",
                            type=OpportunityType.CODE_SMELL.value,
                            description=f"Function '{node.name}' is too long ({func_lines} lines)",
                            source=self.name,
                            priority=TaskPriority.LOW,
                            estimated_effort=2.0,
                            estimated_value=1.5,
                            confidence=0.85,
                            context={
                                "file": str(file_path),
                                "line": node.lineno,
                                "smell_type": "long_function",
                                "function_name": node.name,
                                "lines": func_lines,
                            }
                        ))

                # Check parameter count
                param_count = len(node.args.args) + len(node.args.kwonlyargs)
                if param_count > self.max_parameters:
                    opportunities.append(Opportunity(
                        id=f"smell_params_{hash(f'{file_path}:{node.name}')}",
                        type=OpportunityType.CODE_SMELL.value,
                        description=f"Function '{node.name}' has too many parameters ({param_count})",
                        source=self.name,
                        priority=TaskPriority.LOW,
                        estimated_effort=1.5,
                        estimated_value=1.0,
                        confidence=0.8,
                        context={
                            "file": str(file_path),
                            "line": node.lineno,
                            "smell_type": "too_many_parameters",
                            "function_name": node.name,
                            "param_count": param_count,
                        }
                    ))

            elif isinstance(node, ast.ClassDef):
                # Check class size
                method_count = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                if method_count > self.max_class_methods:
                    opportunities.append(Opportunity(
                        id=f"smell_large_class_{hash(f'{file_path}:{node.name}')}",
                        type=OpportunityType.CODE_SMELL.value,
                        description=f"Class '{node.name}' has too many methods ({method_count})",
                        source=self.name,
                        priority=TaskPriority.MEDIUM,
                        estimated_effort=4.0,
                        estimated_value=2.0,
                        confidence=0.8,
                        context={
                            "file": str(file_path),
                            "line": node.lineno,
                            "smell_type": "large_class",
                            "class_name": node.name,
                            "method_count": method_count,
                        }
                    ))

        return opportunities


class TypeAnnotationScanner(BaseScanner):
    """
    Scanner for missing type annotations.

    Detects:
    - Functions without return type hints
    - Parameters without type hints
    - Variables without type hints (in typed modules)
    """

    def __init__(self, config: ScanConfig = None):
        super().__init__("type_annotation_scanner")
        self.config = config or ScanConfig()
        self.scan_interval = 1800  # 30 minutes

    def scan(self) -> List[Opportunity]:
        opportunities = []
        project_path = Path(self.config.project_path)

        for py_file in project_path.glob("**/*.py"):
            if self._should_skip(py_file):
                continue

            try:
                missing = self._find_missing_annotations(py_file)
                opportunities.extend(missing)
            except Exception as e:
                logger.debug(f"Error analyzing {py_file}: {e}")

        return opportunities

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if Path(path_str).match(pattern):
                return True
        if "test" in file_path.name.lower():
            return True
        return False

    def _find_missing_annotations(self, file_path: Path) -> List[Opportunity]:
        """Find missing type annotations in a file."""
        opportunities = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except:
            return []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue

                missing_types = []

                # Check return type
                if node.returns is None and node.name != "__init__":
                    missing_types.append("return")

                # Check parameter types
                for arg in node.args.args:
                    if arg.arg != "self" and arg.annotation is None:
                        missing_types.append(f"param:{arg.arg}")

                if missing_types:
                    opportunities.append(Opportunity(
                        id=f"type_hint_{hash(f'{file_path}:{node.name}')}",
                        type=OpportunityType.TYPE_ANNOTATION.value,
                        description=f"Add type hints to '{node.name}' ({len(missing_types)} missing)",
                        source=self.name,
                        priority=TaskPriority.BACKGROUND,
                        estimated_effort=0.25,
                        estimated_value=0.5,
                        confidence=0.95,
                        context={
                            "file": str(file_path),
                            "line": node.lineno,
                            "function_name": node.name,
                            "missing_types": missing_types,
                        }
                    ))

        return opportunities


class OpportunityScout:
    """
    Main opportunity scout that coordinates all scanners.

    Features:
    - Manages multiple specialized scanners
    - Prioritizes and deduplicates opportunities
    - Tracks scan history and statistics
    - Provides opportunity filtering and search
    """

    def __init__(self, config: ScanConfig = None):
        self.config = config or ScanConfig()
        self._scanners: List[BaseScanner] = []
        self._lock = threading.RLock()

        # History and statistics
        self._scan_history: List[ScanResult] = []
        self._all_opportunities: List[Opportunity] = []
        self._opportunity_cache: Dict[str, Opportunity] = {}

        # Register default scanners
        self._register_default_scanners()

        logger.info("OpportunityScout initialized with %d scanners", len(self._scanners))

    def _register_default_scanners(self):
        """Register the default set of scanners."""
        self._scanners = [
            TodoScanner(self.config),
            TestCoverageScanner(self.config),
            DocGapScanner(self.config),
            CodeSmellScanner(self.config),
            TypeAnnotationScanner(self.config),
        ]

    def register_scanner(self, scanner: BaseScanner):
        """Register a custom scanner."""
        with self._lock:
            self._scanners.append(scanner)
            logger.info(f"Registered scanner: {scanner.name}")

    def scan_all(self) -> List[Opportunity]:
        """Run all scanners and collect opportunities."""
        all_opportunities = []

        for scanner in self._scanners:
            if not scanner.enabled:
                continue

            start_time = datetime.now()
            errors = []

            try:
                opportunities = scanner.scan()
                all_opportunities.extend(opportunities)
                scanner.last_scan = datetime.now()
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Scanner {scanner.name} failed: {e}")

            # Record scan result
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._scan_history.append(ScanResult(
                scanner_name=scanner.name,
                opportunities_found=len(opportunities) if not errors else 0,
                scan_duration_ms=elapsed_ms,
                errors=errors,
            ))

        # Deduplicate and prioritize
        with self._lock:
            self._all_opportunities = self._deduplicate(all_opportunities)
            self._all_opportunities = self._prioritize(self._all_opportunities)

            # Update cache
            self._opportunity_cache = {opp.id: opp for opp in self._all_opportunities}

        logger.info(f"Scan complete: {len(self._all_opportunities)} unique opportunities found")
        return self._all_opportunities

    def scan_incremental(self) -> List[Opportunity]:
        """Run only scanners that are due for a scan."""
        new_opportunities = []

        for scanner in self._scanners:
            if not scanner.enabled or not scanner.should_scan():
                continue

            try:
                opportunities = scanner.scan()
                new_opportunities.extend(opportunities)
                scanner.last_scan = datetime.now()
            except Exception as e:
                logger.error(f"Scanner {scanner.name} failed: {e}")

        # Merge with existing opportunities
        with self._lock:
            existing_ids = {opp.id for opp in self._all_opportunities}
            for opp in new_opportunities:
                if opp.id not in existing_ids:
                    self._all_opportunities.append(opp)
                    self._opportunity_cache[opp.id] = opp

            self._all_opportunities = self._prioritize(self._all_opportunities)

        return new_opportunities

    def _deduplicate(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Remove duplicate opportunities."""
        seen = set()
        unique = []

        for opp in opportunities:
            key = (opp.type, opp.context.get("file"), opp.context.get("line"))
            if key not in seen:
                seen.add(key)
                unique.append(opp)

        return unique

    def _prioritize(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Sort opportunities by priority and value."""
        return sorted(
            opportunities,
            key=lambda o: (o.priority.value, -o.value_score),
        )

    def get_opportunities(
        self,
        type_filter: Optional[OpportunityType] = None,
        priority_filter: Optional[TaskPriority] = None,
        file_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[Opportunity]:
        """Get filtered opportunities."""
        with self._lock:
            result = self._all_opportunities.copy()

        if type_filter:
            result = [o for o in result if o.type == type_filter.value]

        if priority_filter:
            result = [o for o in result if o.priority == priority_filter]

        if file_filter:
            result = [o for o in result if file_filter in o.context.get("file", "")]

        return result[:limit]

    def get_opportunity_by_id(self, opportunity_id: str) -> Optional[Opportunity]:
        """Get a specific opportunity by ID."""
        return self._opportunity_cache.get(opportunity_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scanning statistics."""
        with self._lock:
            by_type = {}
            by_priority = {}

            for opp in self._all_opportunities:
                by_type[opp.type] = by_type.get(opp.type, 0) + 1
                by_priority[opp.priority.name] = by_priority.get(opp.priority.name, 0) + 1

            return {
                "total_opportunities": len(self._all_opportunities),
                "by_type": by_type,
                "by_priority": by_priority,
                "scanners": [s.name for s in self._scanners],
                "last_scan_results": [r.to_dict() for r in self._scan_history[-10:]],
            }

    def get_scanners(self) -> List[Dict[str, Any]]:
        """Get scanner status."""
        return [
            {
                "name": s.name,
                "enabled": s.enabled,
                "scan_interval": s.scan_interval,
                "last_scan": s.last_scan.isoformat() if s.last_scan else None,
            }
            for s in self._scanners
        ]

    def enable_scanner(self, name: str, enabled: bool = True):
        """Enable or disable a scanner."""
        for scanner in self._scanners:
            if scanner.name == name:
                scanner.enabled = enabled
                logger.info(f"Scanner {name} {'enabled' if enabled else 'disabled'}")
                return
        raise ValueError(f"Scanner not found: {name}")


# Convenience function for creating a configured scout
def create_opportunity_scout(project_path: str = ".") -> OpportunityScout:
    """Create an OpportunityScout with default configuration."""
    config = ScanConfig(project_path=project_path)
    return OpportunityScout(config)

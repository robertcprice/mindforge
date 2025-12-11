"""
Tool Response Format Specification for MindForge

This module defines the exact response formats the AI should produce
when deciding to use tools. These patterns are used for:
1. Training data generation
2. Response validation
3. Fine-tuning the model to follow structured output

Enhanced with JSON schema validation and robust parsing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import json
import re
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Valid action types the AI can choose."""
    TOOL = "TOOL"
    DO_NOTHING = "DO_NOTHING"
    REFLECT = "REFLECT"


# JSON Schema definitions for each tool's arguments
TOOL_SCHEMAS: Dict[str, dict] = {
    "shell": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"}
        },
        "required": ["command"],
        "additionalProperties": False
    },
    "filesystem": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list", "exists", "info", "delete", "mkdir", "grep", "glob"]
            },
            "path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["operation", "path"],
        "additionalProperties": False
    },
    "git": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "log", "diff", "branch", "add", "commit", "checkout", "pull", "push"]
            },
            "args": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    },
    "web": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["fetch", "search", "extract", "validate"]
            },
            "url": {"type": "string"},
            "query": {"type": "string"},
            "extract_links": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    },
    "n8n": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["health", "list", "get", "run", "webhook", "history", "start", "stop", "create"]
            },
            "workflow_id": {"type": "string"},
            "webhook_path": {"type": "string"},
            "data": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    },
    "ollama": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["health", "list", "generate", "show", "chat"]
            },
            "model": {"type": "string"},
            "prompt": {"type": "string"},
            "system": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    },
    "code": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["analyze", "symbols", "validate", "edit", "diff", "search"]
            },
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "old_content": {"type": "string"},
            "new_content": {"type": "string"},
            "content": {"type": "string"},
            "language": {"type": "string"},
            "pattern": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    },
    "kvrm": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["resolve", "search", "store", "ground", "list"]
            },
            "key": {"type": "string"},
            "query": {"type": "string"},
            "claim": {"type": "string"},
            "content": {"type": "string"},
            "source": {"type": "string"}
        },
        "required": ["operation"],
        "additionalProperties": False
    }
}


def validate_against_schema(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Validate arguments against the tool's JSON schema.

    Returns None if valid, or an error message if invalid.
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return None  # No schema, allow through

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in args:
            return f"Missing required field '{field}' for {tool_name}"

    # Check field types and enums
    properties = schema.get("properties", {})
    for key, value in args.items():
        if key not in properties:
            # Unknown field - warn but don't fail
            logger.debug(f"Unknown field '{key}' for {tool_name}, will be filtered")
            continue

        prop_schema = properties[key]

        # Check enum values
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            valid_ops = ", ".join(prop_schema["enum"])
            return f"Invalid {key}='{value}' for {tool_name}. Valid values: {valid_ops}"

    return None  # Valid


@dataclass
class ToolSpec:
    """Specification for a tool's expected format."""
    name: str
    description: str
    required_args: List[str]
    optional_args: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    reward_on_success: float = 1.0
    reward_on_failure: float = -0.5


# Define all tool specifications with exact formats
TOOL_SPECS: Dict[str, ToolSpec] = {
    "shell": ToolSpec(
        name="shell",
        description="Execute safe shell commands",
        required_args=["command"],
        optional_args=[],
        examples=[
            'TOOL: shell(command="ls")',
            'TOOL: shell(command="pwd")',
            'TOOL: shell(command="date")',
            'TOOL: shell(command="whoami")',
            'TOOL: shell(command="echo hello world")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "filesystem": ToolSpec(
        name="filesystem",
        description="Read files and list directories",
        required_args=["operation", "path"],
        optional_args=["content"],
        examples=[
            'TOOL: filesystem(operation="read", path="./README.md")',
            'TOOL: filesystem(operation="list", path=".")',
            'TOOL: filesystem(operation="write", path="./notes.txt", content="My notes")',
            'TOOL: filesystem(operation="exists", path="./file.txt")',
            'TOOL: filesystem(operation="info", path=".")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "git": ToolSpec(
        name="git",
        description="Git repository operations",
        required_args=["operation"],
        optional_args=["args"],
        examples=[
            'TOOL: git(operation="status")',
            'TOOL: git(operation="log", args="--oneline -5")',
            'TOOL: git(operation="diff")',
            'TOOL: git(operation="branch")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "web": ToolSpec(
        name="web",
        description="Fetch web content and search the web",
        required_args=["operation"],
        optional_args=["url", "query", "extract_links"],
        examples=[
            'TOOL: web(operation="fetch", url="https://api.github.com/zen")',
            'TOOL: web(operation="fetch", url="https://httpbin.org/get")',
            'TOOL: web(operation="search", query="python tutorial")',
            'TOOL: web(operation="extract", url="https://example.com", extract_links="true")',
            'TOOL: web(operation="validate", url="https://example.com")',
        ],
        reward_on_success=1.2,  # Higher reward for web exploration
        reward_on_failure=-0.2,
    ),
    "n8n": ToolSpec(
        name="n8n",
        description="Workflow automation via n8n",
        required_args=["operation"],
        optional_args=["workflow_id", "webhook_path", "data"],
        examples=[
            'TOOL: n8n(operation="health")',
            'TOOL: n8n(operation="list")',
            'TOOL: n8n(operation="get", workflow_id="123")',
            'TOOL: n8n(operation="run", workflow_id="my_workflow")',
            'TOOL: n8n(operation="webhook", webhook_path="my-webhook")',
            'TOOL: n8n(operation="history")',
            'TOOL: n8n(operation="start")',
            'TOOL: n8n(operation="stop")',
        ],
        reward_on_success=1.5,  # High reward for automation
        reward_on_failure=-0.4,
    ),
    "ollama": ToolSpec(
        name="ollama",
        description="Query local LLM models",
        required_args=["operation"],
        optional_args=["model", "prompt", "system"],
        examples=[
            'TOOL: ollama(operation="health")',
            'TOOL: ollama(operation="list")',
            'TOOL: ollama(operation="generate", model="qwen3:8b", prompt="Hello")',
            'TOOL: ollama(operation="show", model="qwen3:8b")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "code": ToolSpec(
        name="code",
        description="Code analysis and editing operations",
        required_args=["operation"],
        optional_args=["path", "old_string", "new_string", "old_content", "new_content", "content", "language"],
        examples=[
            'TOOL: code(operation="analyze", path="./main.py")',
            'TOOL: code(operation="symbols", path="./main.py")',
            'TOOL: code(operation="validate", path="./main.py")',
            'TOOL: code(operation="edit", path="./main.py", old_string="foo", new_string="bar")',
            'TOOL: code(operation="diff", old_content="x=1", new_content="x=2")',
            'TOOL: code(operation="search", pattern="TODO", path=".")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "kvrm": ToolSpec(
        name="kvrm",
        description="Key-Value Response Mapping for fact verification",
        required_args=["operation"],
        optional_args=["key", "query", "claim", "content", "source"],
        examples=[
            'TOOL: kvrm(operation="resolve", key="user.name")',
            'TOOL: kvrm(operation="search", query="project status")',
            'TOOL: kvrm(operation="store", key="fact.today", content="Tuesday", source="system")',
            'TOOL: kvrm(operation="ground", claim="The sky is blue")',
        ],
        reward_on_success=1.3,  # Encourage fact verification
        reward_on_failure=-0.2,
    ),
}

# Non-tool action formats
DO_NOTHING_FORMAT = "DO_NOTHING: {reason}"
REFLECT_FORMAT = "REFLECT: {topic}"


@dataclass
class ParsedAction:
    """Result of parsing an AI response."""
    action_type: ActionType
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    is_valid: bool = False
    validation_error: Optional[str] = None


def parse_action(response: str) -> ParsedAction:
    """
    Parse an AI response into a structured action.

    Returns ParsedAction with validation status.
    """
    response = response.strip()

    # Try TOOL format
    if response.upper().startswith("TOOL:"):
        return _parse_tool_action(response[5:].strip())

    # Try DO_NOTHING format
    if response.upper().startswith("DO_NOTHING:"):
        return ParsedAction(
            action_type=ActionType.DO_NOTHING,
            raw_text=response[11:].strip(),
            is_valid=True,
        )

    # Try REFLECT format
    if response.upper().startswith("REFLECT:"):
        return ParsedAction(
            action_type=ActionType.REFLECT,
            raw_text=response[8:].strip(),
            is_valid=True,
        )

    # Invalid format
    return ParsedAction(
        action_type=ActionType.REFLECT,  # Default fallback
        raw_text=response,
        is_valid=False,
        validation_error=f"Response must start with TOOL:, DO_NOTHING:, or REFLECT:. Got: {response[:50]}...",
    )


def _parse_tool_action(tool_str: str) -> ParsedAction:
    """Parse a tool call string like 'shell(command="ls")'.

    Enhanced with JSON schema validation and robust multiline handling.
    """
    # Match tool_name(args)
    match = re.match(r'(\w+)\s*\((.*)\)', tool_str, re.DOTALL)

    if not match:
        return ParsedAction(
            action_type=ActionType.TOOL,
            raw_text=tool_str,
            is_valid=False,
            validation_error=f"Invalid tool format. Expected: tool_name(arg=\"value\"). Got: {tool_str}",
        )

    tool_name = match.group(1).lower()
    args_str = match.group(2).strip()

    # Check if tool exists
    if tool_name not in TOOL_SPECS:
        return ParsedAction(
            action_type=ActionType.TOOL,
            tool_name=tool_name,
            raw_text=tool_str,
            is_valid=False,
            validation_error=f"Unknown tool: {tool_name}. Available: {list(TOOL_SPECS.keys())}",
        )

    # Parse arguments with enhanced parsing
    args = _parse_args(args_str)

    # Validate against JSON schema first
    schema_error = validate_against_schema(tool_name, args)
    if schema_error:
        return ParsedAction(
            action_type=ActionType.TOOL,
            tool_name=tool_name,
            args=args,
            raw_text=tool_str,
            is_valid=False,
            validation_error=schema_error,
        )

    # Validate required args from ToolSpec
    spec = TOOL_SPECS[tool_name]
    missing_args = [arg for arg in spec.required_args if arg not in args]

    if missing_args:
        return ParsedAction(
            action_type=ActionType.TOOL,
            tool_name=tool_name,
            args=args,
            raw_text=tool_str,
            is_valid=False,
            validation_error=f"Missing required args for {tool_name}: {missing_args}",
        )

    # Filter to only known parameters to prevent unexpected keyword argument errors
    known_args = set(spec.required_args) | set(spec.optional_args)
    filtered_args = {k: v for k, v in args.items() if k in known_args}

    # Log if we filtered anything
    filtered_out = set(args.keys()) - set(filtered_args.keys())
    if filtered_out:
        logger.debug(f"Filtered out unknown args for {tool_name}: {filtered_out}")

    return ParsedAction(
        action_type=ActionType.TOOL,
        tool_name=tool_name,
        args=filtered_args,
        raw_text=tool_str,
        is_valid=True,
    )


def _parse_args(args_str: str) -> Dict[str, Any]:
    """Parse argument string like 'command="ls", path="./"'.

    Handles multiline content, triple-quoted strings, and escaped characters.
    Uses multiple parsing strategies for robustness.
    """
    args = {}

    if not args_str:
        return args

    # Strategy 1: Try JSON-style parsing first for clean args
    try:
        # Convert to dict-like format for json
        json_str = "{" + args_str + "}"
        # Fix unquoted keys: word= -> "word":
        json_str = re.sub(r'(\w+)\s*=', r'"\1":', json_str)
        parsed = json.loads(json_str)
        return parsed
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 2: Handle triple-quoted strings first (for code content)
    triple_pattern = r'(\w+)\s*=\s*"""(.*?)"""'
    triple_matches = re.findall(triple_pattern, args_str, re.DOTALL)
    for key, value in triple_matches:
        args[key] = value.strip()
        # Remove matched portion from args_str
        args_str = re.sub(triple_pattern, '', args_str, count=1, flags=re.DOTALL)

    # Strategy 3: Handle escaped quotes in values
    # Match key="value with \"escaped\" quotes"
    escaped_pattern = r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"'
    escaped_matches = re.findall(escaped_pattern, args_str, re.DOTALL)
    for key, value in escaped_matches:
        if key not in args:
            # Unescape the value
            unescaped = value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            args[key] = unescaped

    # Strategy 4: Match simple key="value" patterns
    pattern = r'(\w+)\s*=\s*"([^"]*)"'
    matches = re.findall(pattern, args_str)
    for key, value in matches:
        if key not in args:
            args[key] = value

    # Strategy 5: Try single quotes
    pattern_sq = r"(\w+)\s*=\s*'([^']*)'"
    matches_sq = re.findall(pattern_sq, args_str)
    for key, value in matches_sq:
        if key not in args:
            args[key] = value

    # Strategy 6: Handle unquoted values (for simple args like numbers/booleans)
    unquoted_pattern = r'(\w+)\s*=\s*([^\s,\)]+)'
    unquoted_matches = re.findall(unquoted_pattern, args_str)
    for key, value in unquoted_matches:
        if key not in args:
            # Skip if value starts with quote (already handled)
            if not value.startswith('"') and not value.startswith("'"):
                # Try to parse as number or boolean
                if value.lower() == 'true':
                    args[key] = True
                elif value.lower() == 'false':
                    args[key] = False
                elif value.isdigit():
                    args[key] = int(value)
                else:
                    try:
                        args[key] = float(value)
                    except ValueError:
                        args[key] = value

    return args


def get_format_instructions() -> str:
    """Generate format instructions for the decision prompt."""
    tool_examples = []
    for name, spec in TOOL_SPECS.items():
        tool_examples.append(f"  - {spec.examples[0]}")

    return f"""Your response MUST be in ONE of these exact formats (no other text):

**To use a tool:**
{chr(10).join(tool_examples)}

**To do nothing:**
DO_NOTHING: <brief reason>

**To continue thinking:**
REFLECT: <what to think about>

IMPORTANT: Start your response DIRECTLY with TOOL:, DO_NOTHING:, or REFLECT:
Do NOT include any other text, explanations, or formatting."""


def validate_and_score(response: str) -> tuple[ParsedAction, float]:
    """
    Validate a response and return a reward score.

    Returns (parsed_action, reward_score)
    """
    parsed = parse_action(response)

    if not parsed.is_valid:
        # Penalty for invalid format
        return parsed, -1.0

    if parsed.action_type == ActionType.TOOL and parsed.tool_name:
        # Base reward for correct format
        spec = TOOL_SPECS.get(parsed.tool_name)
        if spec:
            return parsed, 0.5  # Partial reward, full reward on execution success

    # Valid non-tool actions get small positive reward
    return parsed, 0.2

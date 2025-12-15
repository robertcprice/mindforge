"""
Conch DNA - Cortex Layer: ActionCortex

The action neuron selects appropriate tools and formats calls.
Uses Qwen3-1.7B with r=8 for fast action selection.

Domain: Tool selection and action formatting
Model: Qwen3-1.7B
LoRA Rank: 8
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class ActionType:
    """Action types the neuron can output."""

    TOOL_CALL = "tool_call"      # Execute a tool
    DO_NOTHING = "do_nothing"    # No action needed
    REFLECT = "reflect"          # Need to think more


class ActionCortex(CortexNeuron):
    """Action selection and tool call formatting neuron.

    Decides what action to take for a given task:
    - Select appropriate tool from available tools
    - Format tool call with correct arguments
    - Recognize when no action is needed
    - Recognize when more reflection is needed

    Output format:
        {
            "action_type": "tool_call|do_nothing|reflect",
            "tool_name": "tool_name" or null,
            "arguments": {"arg1": "value1", ...} or null,
            "reasoning": "why this action",
            "expected_outcome": "what we expect to happen"
        }
    """

    DEFAULT_SYSTEM_PROMPT = """/no_think
You output ONLY valid JSON. No explanations. No markdown. No prose.

Your response MUST start with { and end with }

Required JSON structure:
{"action_type": "tool_call|do_nothing|reflect", "tool_name": "name" or null, "arguments": {} or {"key": "value"}, "reasoning": "1 sentence", "expected_outcome": "1 sentence"}

Example:
{"action_type": "tool_call", "tool_name": "list_files", "arguments": {"path": "."}, "reasoning": "Need to inspect repository layout", "expected_outcome": "File list printed"}"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen3-1.7B-4bit",
        lora_rank: int = 8,
        confidence_threshold: float = 0.7
    ):
        """Initialize ActionCortex.

        Args:
            base_model: Base model identifier
            lora_rank: LoRA rank (default 8)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="action_cortex",
            domain=NeuronDomain.ACTION,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=512,  # Capped for controlled output
            temperature=0.2,  # Low for deterministic JSON
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            repetition_penalty=1.3,
        )
        super().__init__(config)
        # Expected fields for confidence estimation
        self._expected_fields = ["action_type", "tool_name", "arguments"]

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare action selection prompt.

        Args:
            input_data: Must contain:
                - task: str or dict - task to execute
                - available_tools: list - tool descriptions
                - context: str - additional context (optional)

        Returns:
            Formatted prompt
        """
        task = input_data.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", str(task))

        available_tools = input_data.get("available_tools", [])
        context = input_data.get("context", "")

        # Format tool descriptions
        tools_str = ""
        if available_tools:
            tool_items = []
            for tool in available_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "")
                    args = tool.get("arguments", {})
                    tool_items.append(f"- {name}: {desc}\n  Args: {args}")
                else:
                    tool_items.append(f"- {tool}")
            tools_str = "Available tools:\n" + "\n".join(tool_items)
        else:
            tools_str = "No tools available (use DO_NOTHING or REFLECT)"

        prompt = f"""{self.config.system_prompt}

TASK:
{task}

{tools_str}

{f"CONTEXT:\n{context}" if context else ""}

Respond with JSON only. Begin your response with {{"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse action selection output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed action structure
        """
        try:
            json_text = self._extract_first_json(raw_output)
            if not json_text:
                raise ValueError("no JSON found")

            action_data = json.loads(json_text)

            # Validate action_type
            valid_types = [ActionType.TOOL_CALL, ActionType.DO_NOTHING, ActionType.REFLECT]
            if action_data.get("action_type") not in valid_types:
                action_data["action_type"] = ActionType.REFLECT

            # Ensure required fields
            action_data.setdefault("reasoning", "")
            action_data.setdefault("expected_outcome", "")
            action_data.setdefault("arguments", {})
            if action_data.get("arguments") is None:
                action_data["arguments"] = {}

            # Validate tool_call has tool_name
            if action_data["action_type"] == ActionType.TOOL_CALL:
                if not action_data.get("tool_name"):
                    action_data["action_type"] = ActionType.REFLECT
                    action_data["reasoning"] = "Missing tool name"

            return action_data

        except Exception as e:
            logger.error(f"{self.config.name}: JSON parse error: {e}")
            return {
                "action_type": ActionType.REFLECT,
                "tool_name": None,
                "arguments": None,
                "reasoning": f"Parse error: {e}",
                "expected_outcome": "Need to reformulate"
            }

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in action selection.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed action structure

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = super()._estimate_confidence(input_data, raw_output, parsed_output)

        # Check action type validity
        action_type = parsed_output.get("action_type")
        if action_type == ActionType.REFLECT:
            confidence *= 0.6  # Reflecting is low confidence by definition

        # Check tool call completeness
        if action_type == ActionType.TOOL_CALL:
            tool_name = parsed_output.get("tool_name")
            arguments = parsed_output.get("arguments")

            # Verify tool exists in available tools
            available_tools = input_data.get("available_tools", [])
            tool_names = []
            for tool in available_tools:
                if isinstance(tool, dict):
                    tool_names.append(tool.get("name"))
                else:
                    tool_names.append(str(tool))

            if tool_name not in tool_names and tool_names:
                confidence *= 0.5  # Invalid tool selection

            # Check if arguments are provided
            if not arguments:
                confidence *= 0.7

            # Check if arguments is a dict
            if arguments and not isinstance(arguments, dict):
                confidence *= 0.6

        # Check reasoning quality
        reasoning = parsed_output.get("reasoning", "")
        if len(reasoning) < 10:
            confidence *= 0.8  # Weak reasoning
        elif len(reasoning) > 500:
            confidence *= 0.9  # Too verbose

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def select_action(
        self,
        task: str | Dict[str, Any],
        available_tools: List[Dict[str, Any]],
        context: str = ""
    ) -> NeuronOutput:
        """High-level action selection interface.

        Args:
            task: Task description or dict
            available_tools: List of available tool descriptions
            context: Additional context (optional)

        Returns:
            NeuronOutput with action structure in metadata
        """
        input_data = {
            "task": task,
            "available_tools": available_tools,
            "context": context
        }

        output = self.infer(input_data)
        output.metadata["input_data"] = input_data

        return output

    def get_action_type(self, output: NeuronOutput) -> str:
        """Extract action type from output.

        Args:
            output: NeuronOutput from select_action()

        Returns:
            Action type string
        """
        return output.metadata.get("action_type", ActionType.REFLECT)

    def get_tool_call(self, output: NeuronOutput) -> Optional[tuple[str, Dict[str, Any]]]:
        """Extract tool call information from output.

        Args:
            output: NeuronOutput from select_action()

        Returns:
            Tuple of (tool_name, arguments) or None if not a tool call
        """
        if self.get_action_type(output) != ActionType.TOOL_CALL:
            return None

        tool_name = output.metadata.get("tool_name")
        arguments = output.metadata.get("arguments", {})

        if not tool_name:
            return None

        return (tool_name, arguments)

    def get_reasoning(self, output: NeuronOutput) -> str:
        """Extract reasoning from output.

        Args:
            output: NeuronOutput from select_action()

        Returns:
            Reasoning text
        """
        return output.metadata.get("reasoning", "")

    def format_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Format a tool call as string.

        Args:
            tool_name: Name of tool
            arguments: Tool arguments

        Returns:
            Formatted tool call string
        """
        # Format arguments
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
        return f"TOOL: {tool_name}({args_str})"

    def parse_tool_call(self, call_string: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse a formatted tool call string.

        Args:
            call_string: String like "TOOL: name(arg1=val1, arg2=val2)"

        Returns:
            Tuple of (tool_name, arguments) or None if invalid
        """
        match = re.match(r'TOOL:\s*(\w+)\((.*?)\)', call_string)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (simple key=value pairs)
        arguments = {}
        if args_str.strip():
            for pair in args_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    arguments[key] = value

        return (tool_name, arguments)


# Factory function
def create_action_cortex(
    base_model: str = "mlx-community/Qwen3-1.7B-4bit",
    adapter_path: str = None
) -> ActionCortex:
    """Create and optionally load ActionCortex.

    Args:
        base_model: Base model identifier
        adapter_path: Optional path to LoRA adapter

    Returns:
        Configured ActionCortex
    """
    from pathlib import Path

    cortex = ActionCortex(base_model=base_model)

    if adapter_path:
        cortex.load_adapter(Path(adapter_path))

    return cortex

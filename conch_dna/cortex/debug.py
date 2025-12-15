"""
Conch DNA - Cortex Layer: DebugCortex

The debug neuron analyzes errors and suggests fixes.
Uses Qwen3-1.7B with r=16 for deeper error analysis.

Domain: Error analysis and debugging
Model: Qwen3-1.7B
LoRA Rank: 16 (higher for complex debugging)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class ErrorSeverity:
    """Error severity levels."""

    CRITICAL = "critical"     # System cannot continue
    HIGH = "high"             # Major functionality broken
    MEDIUM = "medium"         # Feature impaired but recoverable
    LOW = "low"               # Minor issue or warning


class DebugCortex(CortexNeuron):
    """Error analysis and debugging neuron.

    Analyzes errors to:
    - Identify root cause
    - Assess severity
    - Suggest concrete fixes
    - Recognize patterns from previous attempts
    - Provide debugging steps

    Output format:
        {
            "root_cause": "underlying issue description",
            "severity": "critical|high|medium|low",
            "error_category": "syntax|logic|resource|network|permission|unknown",
            "fix_suggestion": "specific fix to try",
            "fix_confidence": 0.0-1.0,
            "debugging_steps": ["step1", "step2", ...],
            "related_to_previous": bool,
            "pattern_detected": "pattern description" or null
        }
    """

    DEFAULT_SYSTEM_PROMPT = """/no_think
You output ONLY valid JSON. No explanations. No markdown. No prose.

Your response MUST start with { and end with }

Required JSON structure:
{"root_cause": "1-2 sentences", "severity": "critical|high|medium|low", "error_category": "syntax|logic|resource|network|permission|unknown", "fix_suggestion": "specific fix", "fix_confidence": 0.0-1.0, "debugging_steps": ["step1"], "related_to_previous": false, "pattern_detected": null}

Example:
{"root_cause": "Missing import statement for json module", "severity": "medium", "error_category": "syntax", "fix_suggestion": "Add 'import json' at top of file", "fix_confidence": 0.9, "debugging_steps": ["Check imports"], "related_to_previous": false, "pattern_detected": null}"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen3-1.7B-4bit",
        lora_rank: int = 16,
        confidence_threshold: float = 0.65
    ):
        """Initialize DebugCortex.

        Args:
            base_model: Base model identifier
            lora_rank: LoRA rank (default 16 for deeper analysis)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="debug_cortex",
            domain=NeuronDomain.DEBUG,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=512,  # Capped for controlled output
            temperature=0.3,  # Low for deterministic JSON
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            repetition_penalty=1.3,
        )
        super().__init__(config)
        # Expected fields for confidence estimation
        self._expected_fields = ["root_cause", "severity", "fix_suggestion"]

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare debugging prompt.

        Args:
            input_data: Must contain:
                - error: str or dict - error message/traceback
                - task: str or dict - task that failed (optional)
                - previous_attempts: list - prior fix attempts (optional)
                - context: str - additional context (optional)

        Returns:
            Formatted prompt
        """
        error = input_data.get("error", "")
        if isinstance(error, dict):
            error = error.get("message", str(error))

        task = input_data.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", str(task))

        previous_attempts = input_data.get("previous_attempts", [])
        context = input_data.get("context", "")

        # Format previous attempts
        attempts_str = ""
        if previous_attempts:
            attempt_items = []
            for i, attempt in enumerate(previous_attempts[-5:], 1):  # Last 5
                if isinstance(attempt, dict):
                    fix = attempt.get("fix", str(attempt))
                    result = attempt.get("result", "failed")
                    attempt_items.append(f"{i}. Tried: {fix} → {result}")
                else:
                    attempt_items.append(f"{i}. {attempt}")
            attempts_str = "Previous fix attempts:\n" + "\n".join(attempt_items)

        prompt = f"""{self.config.system_prompt}

ERROR:
{error}

{f"TASK (what was being attempted):\n{task}" if task else ""}

{attempts_str}

{f"CONTEXT:\n{context}" if context else ""}

Respond with JSON only. Begin your response with {{"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse debugging output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed debug structure
        """
        try:
            json_text = self._extract_first_json(raw_output)
            if not json_text:
                raise ValueError("no JSON found")

            debug_data = json.loads(json_text)

            # Validate severity
            valid_severities = [
                ErrorSeverity.CRITICAL, ErrorSeverity.HIGH,
                ErrorSeverity.MEDIUM, ErrorSeverity.LOW
            ]
            if debug_data.get("severity") not in valid_severities:
                debug_data["severity"] = ErrorSeverity.MEDIUM

            # Validate error_category
            valid_categories = ["syntax", "logic", "resource", "network", "permission", "unknown"]
            if debug_data.get("error_category") not in valid_categories:
                debug_data["error_category"] = "unknown"

            # Ensure required fields
            debug_data.setdefault("root_cause", "Unknown cause")
            debug_data.setdefault("fix_suggestion", "Unable to determine fix")
            debug_data.setdefault("fix_confidence", 0.3)
            debug_data.setdefault("debugging_steps", [])
            debug_data.setdefault("related_to_previous", False)

            return debug_data

        except Exception as e:
            logger.error(f"{self.config.name}: JSON parse error: {e}")
            return {
                "root_cause": f"Parse error: {e}",
                "severity": ErrorSeverity.LOW,
                "error_category": "unknown",
                "fix_suggestion": "Retry analysis",
                "fix_confidence": 0.1,
                "debugging_steps": ["Check debug output format"],
                "related_to_previous": False,
                "pattern_detected": None
            }

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in debugging analysis.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed debug structure

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = super()._estimate_confidence(input_data, raw_output, parsed_output)

        # Use self-reported fix confidence
        fix_confidence = parsed_output.get("fix_confidence", 0.3)
        confidence *= (0.3 + 0.7 * fix_confidence)

        # Check for specific vs generic fix
        fix_suggestion = parsed_output.get("fix_suggestion", "")
        generic_phrases = ["check", "review", "investigate", "look at", "try to"]
        if any(phrase in fix_suggestion.lower() for phrase in generic_phrases):
            confidence *= 0.7  # Penalize generic suggestions

        # Boost confidence for specific error categories
        error_category = parsed_output.get("error_category", "unknown")
        if error_category != "unknown":
            confidence *= 1.1

        # Check debugging steps quality
        debug_steps = parsed_output.get("debugging_steps", [])
        if isinstance(debug_steps, list) and len(debug_steps) > 0:
            confidence *= 1.05
        else:
            confidence *= 0.9

        # Pattern detection is valuable
        if parsed_output.get("pattern_detected"):
            confidence *= 1.15

        # Related to previous attempts (good awareness)
        if parsed_output.get("related_to_previous"):
            confidence *= 1.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def analyze_error(
        self,
        error: str | Dict[str, Any],
        task: Optional[str | Dict[str, Any]] = None,
        previous_attempts: Optional[List[Dict[str, Any]]] = None,
        context: str = ""
    ) -> NeuronOutput:
        """High-level error analysis interface.

        Args:
            error: Error message or exception
            task: Task that failed (optional)
            previous_attempts: Previous fix attempts (optional)
            context: Additional context (optional)

        Returns:
            NeuronOutput with debug structure in metadata
        """
        input_data = {
            "error": error,
            "task": task,
            "previous_attempts": previous_attempts or [],
            "context": context
        }

        output = self.infer(input_data)
        output.metadata["input_data"] = input_data

        return output

    def get_root_cause(self, output: NeuronOutput) -> str:
        """Extract root cause from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            Root cause description
        """
        return output.metadata.get("root_cause", "Unknown")

    def get_severity(self, output: NeuronOutput) -> str:
        """Extract severity from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            Severity level
        """
        return output.metadata.get("severity", ErrorSeverity.MEDIUM)

    def get_fix_suggestion(self, output: NeuronOutput) -> str:
        """Extract fix suggestion from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            Fix suggestion text
        """
        return output.metadata.get("fix_suggestion", "")

    def get_fix_confidence(self, output: NeuronOutput) -> float:
        """Extract fix confidence from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            Fix confidence (0.0-1.0)
        """
        return output.metadata.get("fix_confidence", 0.5)

    def get_debugging_steps(self, output: NeuronOutput) -> List[str]:
        """Extract debugging steps from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            List of debugging steps
        """
        steps = output.metadata.get("debugging_steps", [])
        return steps if isinstance(steps, list) else []

    def is_critical(self, output: NeuronOutput) -> bool:
        """Check if error is critical severity.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            True if critical
        """
        return self.get_severity(output) == ErrorSeverity.CRITICAL

    def has_pattern(self, output: NeuronOutput) -> bool:
        """Check if a pattern was detected.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            True if pattern detected
        """
        return bool(output.metadata.get("pattern_detected"))

    def get_pattern(self, output: NeuronOutput) -> Optional[str]:
        """Extract detected pattern from output.

        Args:
            output: NeuronOutput from analyze_error()

        Returns:
            Pattern description or None
        """
        return output.metadata.get("pattern_detected")


# Factory function
def create_debug_cortex(
    base_model: str = "mlx-community/Qwen3-1.7B-4bit",
    adapter_path: str = None
) -> DebugCortex:
    """Create and optionally load DebugCortex.

    Args:
        base_model: Base model identifier
        adapter_path: Optional path to LoRA adapter

    Returns:
        Configured DebugCortex
    """
    from pathlib import Path

    cortex = DebugCortex(base_model=base_model)

    if adapter_path:
        cortex.load_adapter(Path(adapter_path))

    return cortex

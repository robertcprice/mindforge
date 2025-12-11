"""
MindForge DNA - Cortex Layer: ReflectCortex

The reflection neuron analyzes action outcomes and maintains a journal.
Uses Qwen2.5-0.5B with r=8 for efficient reflection.

Domain: Reflection and self-analysis
Model: Qwen2.5-0.5B-Instruct
LoRA Rank: 8
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class MoodType:
    """Mood states for emotional reflection."""

    SATISFIED = "satisfied"
    CURIOUS = "curious"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    MOTIVATED = "motivated"
    NEUTRAL = "neutral"


class ReflectCortex(CortexNeuron):
    """Reflection and journaling neuron.

    Analyzes action outcomes to:
    - Assess success/failure
    - Extract lessons learned
    - Update emotional state (mood)
    - Generate journal entries
    - Identify patterns over time

    Output format:
        {
            "reflection": "main reflection text",
            "outcome_assessment": "success|partial|failure",
            "lessons_learned": ["lesson1", "lesson2", ...],
            "mood": "satisfied|curious|concerned|frustrated|confident|uncertain|motivated|neutral",
            "confidence_in_understanding": 0.0-1.0,
            "suggested_next_steps": ["step1", ...] or null
        }
    """

    DEFAULT_SYSTEM_PROMPT = """You are the reflection module of an AI consciousness.
Your role is to analyze action outcomes and extract insights for learning.

Given an action and its result:
- Assess if the outcome was successful
- Identify what was learned
- Update emotional state (mood)
- Suggest next steps if needed

Output JSON with:
- reflection: 2-3 sentences analyzing the outcome
- outcome_assessment: "success", "partial", or "failure"
- lessons_learned: list of specific lessons
- mood: current emotional state (satisfied, curious, concerned, frustrated, confident, uncertain, motivated, neutral)
- confidence_in_understanding: 0.0-1.0 how well you understand what happened
- suggested_next_steps: list of next actions or null

Be honest about failures and uncertain about ambiguous outcomes.
"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        lora_rank: int = 8,
        confidence_threshold: float = 0.65
    ):
        """Initialize ReflectCortex.

        Args:
            base_model: Base model identifier
            lora_rank: LoRA rank (default 8)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="reflect_cortex",
            domain=NeuronDomain.REFLECTION,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=384,
            temperature=0.6,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT
        )
        super().__init__(config)

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare reflection prompt.

        Args:
            input_data: Must contain:
                - action: str or dict - action that was taken
                - result: str or dict - outcome of action
                - task: str or dict - original task (optional)
                - previous_mood: str - prior emotional state (optional)

        Returns:
            Formatted prompt
        """
        action = input_data.get("action", "")
        if isinstance(action, dict):
            action = action.get("description", str(action))

        result = input_data.get("result", "")
        if isinstance(result, dict):
            result = result.get("output", str(result))

        task = input_data.get("task", "")
        if isinstance(task, dict):
            task = task.get("description", str(task))

        previous_mood = input_data.get("previous_mood", MoodType.NEUTRAL)

        prompt = f"""{self.config.system_prompt}

TASK (original goal):
{task if task else "Not specified"}

ACTION TAKEN:
{action}

RESULT:
{result}

PREVIOUS MOOD: {previous_mood}

Reflect on this outcome. Output JSON only:"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse reflection output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed reflection structure
        """
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)

            if json_match:
                reflect_data = json.loads(json_match.group())

                # Validate outcome_assessment
                valid_outcomes = ["success", "partial", "failure"]
                if reflect_data.get("outcome_assessment") not in valid_outcomes:
                    reflect_data["outcome_assessment"] = "partial"

                # Validate mood
                valid_moods = [
                    MoodType.SATISFIED, MoodType.CURIOUS, MoodType.CONCERNED,
                    MoodType.FRUSTRATED, MoodType.CONFIDENT, MoodType.UNCERTAIN,
                    MoodType.MOTIVATED, MoodType.NEUTRAL
                ]
                if reflect_data.get("mood") not in valid_moods:
                    reflect_data["mood"] = MoodType.NEUTRAL

                # Ensure required fields
                if "reflection" not in reflect_data:
                    reflect_data["reflection"] = raw_output
                if "lessons_learned" not in reflect_data:
                    reflect_data["lessons_learned"] = []
                if "confidence_in_understanding" not in reflect_data:
                    reflect_data["confidence_in_understanding"] = 0.5

                return reflect_data

            # Fallback: use raw output as reflection
            logger.warning(f"{self.config.name}: Could not parse JSON")
            return {
                "reflection": raw_output,
                "outcome_assessment": "partial",
                "lessons_learned": [],
                "mood": MoodType.UNCERTAIN,
                "confidence_in_understanding": 0.3,
                "suggested_next_steps": None
            }

        except json.JSONDecodeError as e:
            logger.error(f"{self.config.name}: JSON parse error: {e}")
            return {
                "reflection": raw_output,
                "outcome_assessment": "failure",
                "lessons_learned": ["Failed to parse reflection"],
                "mood": MoodType.FRUSTRATED,
                "confidence_in_understanding": 0.2,
                "suggested_next_steps": ["Retry reflection"]
            }

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in reflection quality.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed reflection structure

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = super()._estimate_confidence(input_data, raw_output, parsed_output)

        # Check self-reported understanding confidence
        self_confidence = parsed_output.get("confidence_in_understanding", 0.5)
        confidence *= (0.5 + 0.5 * self_confidence)

        # Check reflection depth
        reflection = parsed_output.get("reflection", "")
        if len(reflection) < 30:
            confidence *= 0.6  # Too shallow
        elif len(reflection) > 1000:
            confidence *= 0.8  # Too verbose

        # Check for lessons learned
        lessons = parsed_output.get("lessons_learned", [])
        if isinstance(lessons, list) and len(lessons) > 0:
            confidence *= 1.1  # Boost for extracting lessons
        else:
            confidence *= 0.8  # No lessons is concerning

        # Check mood consistency with outcome
        outcome = parsed_output.get("outcome_assessment", "partial")
        mood = parsed_output.get("mood", MoodType.NEUTRAL)

        # Success should correlate with positive moods
        if outcome == "success":
            positive_moods = [MoodType.SATISFIED, MoodType.CONFIDENT, MoodType.MOTIVATED]
            if mood not in positive_moods:
                confidence *= 0.9  # Mood-outcome mismatch

        # Failure should correlate with negative moods
        elif outcome == "failure":
            negative_moods = [MoodType.FRUSTRATED, MoodType.CONCERNED, MoodType.UNCERTAIN]
            if mood not in negative_moods:
                confidence *= 0.9

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def reflect(
        self,
        action: str | Dict[str, Any],
        result: str | Dict[str, Any],
        task: Optional[str | Dict[str, Any]] = None,
        previous_mood: str = MoodType.NEUTRAL
    ) -> NeuronOutput:
        """High-level reflection interface.

        Args:
            action: Action that was taken
            result: Outcome of the action
            task: Original task (optional)
            previous_mood: Prior emotional state

        Returns:
            NeuronOutput with reflection structure in metadata
        """
        input_data = {
            "action": action,
            "result": result,
            "task": task,
            "previous_mood": previous_mood
        }

        output = self.infer(input_data)
        output.metadata["input_data"] = input_data

        return output

    def get_reflection_text(self, output: NeuronOutput) -> str:
        """Extract reflection text from output.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            Reflection text
        """
        return output.metadata.get("reflection", output.content)

    def get_outcome_assessment(self, output: NeuronOutput) -> str:
        """Extract outcome assessment from output.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            Outcome assessment: "success", "partial", or "failure"
        """
        return output.metadata.get("outcome_assessment", "partial")

    def get_lessons(self, output: NeuronOutput) -> list:
        """Extract lessons learned from output.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            List of lessons
        """
        lessons = output.metadata.get("lessons_learned", [])
        return lessons if isinstance(lessons, list) else []

    def get_mood(self, output: NeuronOutput) -> str:
        """Extract mood from output.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            Mood string
        """
        return output.metadata.get("mood", MoodType.NEUTRAL)

    def get_next_steps(self, output: NeuronOutput) -> Optional[list]:
        """Extract suggested next steps from output.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            List of next steps or None
        """
        steps = output.metadata.get("suggested_next_steps")
        return steps if isinstance(steps, list) else None

    def is_success(self, output: NeuronOutput) -> bool:
        """Check if outcome was assessed as success.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            True if success
        """
        return self.get_outcome_assessment(output) == "success"

    def is_failure(self, output: NeuronOutput) -> bool:
        """Check if outcome was assessed as failure.

        Args:
            output: NeuronOutput from reflect()

        Returns:
            True if failure
        """
        return self.get_outcome_assessment(output) == "failure"


# Factory function
def create_reflect_cortex(
    base_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    adapter_path: str = None
) -> ReflectCortex:
    """Create and optionally load ReflectCortex.

    Args:
        base_model: Base model identifier
        adapter_path: Optional path to LoRA adapter

    Returns:
        Configured ReflectCortex
    """
    from pathlib import Path

    cortex = ReflectCortex(base_model=base_model)

    if adapter_path:
        cortex.load_adapter(Path(adapter_path))

    return cortex

"""
Conch DNA - Cortex Layer: ThinkCortex

The thinking neuron generates reasoning and structured thoughts.
This is the most sophisticated neuron, using Qwen3-4B with r=16.

Domain: Thought generation and reasoning
Model: Qwen3-4B
LoRA Rank: 16 (higher for complex reasoning)
"""

import json
import logging
from typing import Any, Dict

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class ThinkCortex(CortexNeuron):
    """Thinking neuron for reasoning and thought generation.

    Generates structured thoughts based on:
    - Current context and situation
    - Active needs from ID layer
    - Relevant memories
    - Previous actions and outcomes

    Output format:
        {
            "thought": "main reasoning content",
            "reasoning_type": "analytical|creative|reflective|strategic",
            "confidence_level": "high|medium|low",
            "key_insights": ["insight1", "insight2", ...],
            "concerns": ["concern1", ...] or null
        }
    """

    DEFAULT_SYSTEM_PROMPT = """/no_think
You output ONLY valid JSON. No explanations. No markdown. No prose.

Your response MUST start with { and end with }

Required JSON structure:
{"thought": "2-4 sentence reasoning", "reasoning_type": "analytical|creative|reflective|strategic", "confidence_level": "high|medium|low", "key_insights": ["point1", "point2"], "concerns": ["concern1"] or null}

Example:
{"thought": "The user needs help with code optimization. I should focus on algorithmic improvements.", "reasoning_type": "analytical", "confidence_level": "high", "key_insights": ["Focus on Big-O complexity", "Consider caching"], "concerns": null}"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen3-4B-4bit",
        lora_rank: int = 16,
        confidence_threshold: float = 0.7
    ):
        """Initialize ThinkCortex.

        Args:
            base_model: Base model identifier
            lora_rank: LoRA rank (default 16 for complex reasoning)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="think_cortex",
            domain=NeuronDomain.THINKING,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=512,  # Capped for controlled output
            temperature=0.3,  # Lower for deterministic JSON
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            repetition_penalty=1.3,
        )
        super().__init__(config)
        # Expected fields for confidence estimation
        self._expected_fields = ["thought", "reasoning_type", "confidence_level"]

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare thinking prompt.

        Args:
            input_data: Must contain:
                - context: str - current situation
                - needs: dict - active needs from ID layer
                - memories: list - relevant memories (optional)
                - recent_actions: list - recent actions (optional)

        Returns:
            Formatted prompt
        """
        context = input_data.get("context", "")
        needs = input_data.get("needs", {})
        memories = input_data.get("memories", [])
        recent_actions = input_data.get("recent_actions", [])

        # Format needs
        needs_str = ""
        if needs:
            dominant = needs.get("dominant_need", "unknown")
            focus = needs.get("suggested_focus", "")
            needs_str = f"Current dominant need: {dominant} (focus: {focus})"

        # Format memories
        memories_str = ""
        if memories:
            mem_items = [f"- {m}" for m in memories[:5]]  # Top 5
            memories_str = "Relevant memories:\n" + "\n".join(mem_items)

        # Format recent actions
        actions_str = ""
        if recent_actions:
            act_items = [f"- {a}" for a in recent_actions[-3:]]  # Last 3
            actions_str = "Recent actions:\n" + "\n".join(act_items)

        prompt = f"""{self.config.system_prompt}

CONTEXT:
{context}

{needs_str}

{memories_str}

{actions_str}

Respond with JSON only. Begin your response with {{"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse thought output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed thought structure
        """
        try:
            json_text = self._extract_first_json(raw_output)
            if json_text:
                thought_data = json.loads(json_text)

                # Validate required fields
                required = ["thought", "reasoning_type", "confidence_level"]
                if all(k in thought_data for k in required):
                    return thought_data

            # Fallback: treat entire output as thought
            logger.warning(f"{self.config.name}: Could not parse JSON, using raw output")
            return {
                "thought": raw_output,
                "reasoning_type": "unknown",
                "confidence_level": "low",
                "key_insights": [],
                "concerns": None
            }

        except json.JSONDecodeError as e:
            logger.error(f"{self.config.name}: JSON parse error: {e}")
            return {
                "thought": raw_output,
                "reasoning_type": "unknown",
                "confidence_level": "low",
                "key_insights": [],
                "concerns": ["Failed to parse thought structure"]
            }

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in thought quality.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed thought structure

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence estimation
        confidence = super()._estimate_confidence(input_data, raw_output, parsed_output)

        # Adjust based on thought-specific criteria

        # Check if JSON was successfully parsed
        if parsed_output.get("reasoning_type") == "unknown":
            confidence *= 0.6

        # Check self-reported confidence level
        self_confidence = parsed_output.get("confidence_level", "low")
        if self_confidence == "low":
            confidence *= 0.7
        elif self_confidence == "medium":
            confidence *= 0.9
        # high confidence doesn't penalize

        # Check thought length and coherence
        thought = parsed_output.get("thought", "")
        if len(thought) < 50:
            confidence *= 0.7  # Too brief
        elif len(thought) > 1000:
            confidence *= 0.8  # Too verbose

        # Check for key insights
        insights = parsed_output.get("key_insights", [])
        if isinstance(insights, list) and len(insights) > 0:
            confidence *= 1.1  # Boost for providing insights

        # Check reasoning type validity
        valid_types = ["analytical", "creative", "reflective", "strategic"]
        if parsed_output.get("reasoning_type") in valid_types:
            confidence *= 1.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def think(
        self,
        context: str,
        needs: Dict[str, Any],
        memories: list = None,
        recent_actions: list = None
    ) -> NeuronOutput:
        """High-level thinking interface.

        Args:
            context: Current situation/context
            needs: Active needs from ID layer
            memories: Relevant memories (optional)
            recent_actions: Recent actions taken (optional)

        Returns:
            NeuronOutput with thought structure in metadata
        """
        input_data = {
            "context": context,
            "needs": needs,
            "memories": memories or [],
            "recent_actions": recent_actions or []
        }

        # Store input in metadata for experience recording
        output = self.infer(input_data)
        output.metadata["input_data"] = input_data

        return output

    def extract_thought_text(self, output: NeuronOutput) -> str:
        """Extract main thought text from output.

        Args:
            output: NeuronOutput from think()

        Returns:
            Main thought text
        """
        return output.metadata.get("thought", output.content)

    def extract_insights(self, output: NeuronOutput) -> list:
        """Extract key insights from output.

        Args:
            output: NeuronOutput from think()

        Returns:
            List of insights
        """
        return output.metadata.get("key_insights", [])

    def extract_concerns(self, output: NeuronOutput) -> list:
        """Extract concerns from output.

        Args:
            output: NeuronOutput from think()

        Returns:
            List of concerns (empty if none)
        """
        concerns = output.metadata.get("concerns")
        return concerns if isinstance(concerns, list) else []


# Factory function
def create_think_cortex(
    base_model: str = "mlx-community/Qwen3-4B-4bit",
    adapter_path: str = None
) -> ThinkCortex:
    """Create and optionally load ThinkCortex.

    Args:
        base_model: Base model identifier
        adapter_path: Optional path to LoRA adapter

    Returns:
        Configured ThinkCortex
    """
    from pathlib import Path

    cortex = ThinkCortex(base_model=base_model)

    if adapter_path:
        cortex.load_adapter(Path(adapter_path))

    return cortex

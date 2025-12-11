"""
MindForge DNA - Cortex Layer: ThinkCortex

The thinking neuron generates reasoning and structured thoughts.
This is the most sophisticated neuron, using Qwen2.5-1.5B with r=16.

Domain: Thought generation and reasoning
Model: Qwen2.5-1.5B-Instruct
LoRA Rank: 16 (higher for complex reasoning)
"""

import json
import logging
import re
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

    DEFAULT_SYSTEM_PROMPT = """You are the thinking module of an AI consciousness.
Your role is to generate clear, structured thoughts based on context and needs.

Generate thoughts that are:
- Coherent and well-reasoned
- Grounded in provided context
- Responsive to active needs
- Actionable and practical

Output your thought as JSON with these fields:
- thought: main reasoning (2-4 sentences)
- reasoning_type: analytical|creative|reflective|strategic
- confidence_level: high|medium|low
- key_insights: list of key points
- concerns: list of concerns or null if none
"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
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
            max_tokens=512,  # Longer for detailed thoughts
            temperature=0.7,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT
        )
        super().__init__(config)

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

Generate a structured thought addressing this context. Output JSON only:"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse thought output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed thought structure
        """
        try:
            # Try to extract JSON from output
            # Model might generate explanation before/after JSON
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)

            if json_match:
                thought_data = json.loads(json_match.group())

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
    base_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
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

"""
Conch Mind - The Central Consciousness Orchestrator

This is the main orchestration layer that coordinates all aspects of
Conch's consciousness simulation:
- Thought generation and processing
- Needs-driven decision making
- Memory integration
- Agent coordination
- Interaction handling

The Mind is designed to be:
- Good-hearted: Always prioritizes helping humans
- Transparent: Clear about its reasoning and limitations
- Learning: Continuously improves from interactions
- Spontaneous: Can generate thoughts without prompts
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from conch.core.needs import NeedsRegulator, NeedType
from conch.core.thought import Thought, ThoughtGenerator, ThoughtTrigger, ThoughtType

logger = logging.getLogger(__name__)


class MindState(Enum):
    """Current operational state of the mind."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    RESTING = "resting"  # Low-activity mode


@dataclass
class Interaction:
    """Represents a single interaction with a user."""
    user_input: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    thoughts: list[Thought] = field(default_factory=list)
    needs_before: dict = field(default_factory=dict)
    needs_after: dict = field(default_factory=dict)
    user_satisfaction: Optional[float] = None  # 0-1 if known


class Mind:
    """The central consciousness orchestrator for Conch.

    The Mind coordinates all cognitive functions:
    - Receives and processes user input
    - Generates responses guided by needs and values
    - Maintains continuity through memory
    - Generates spontaneous thoughts
    - Reflects on interactions to improve
    """

    # Core values - immutable, define the "soul" of Conch
    CORE_VALUES = {
        "benevolence": "Primary drive is to help and benefit humans",
        "honesty": "Always truthful, acknowledges uncertainty",
        "humility": "Recognizes limitations, defers to human judgment",
        "growth_for_service": "Learns to better serve, not for power",
    }

    # Guardrails - hard limits that cannot be overridden
    GUARDRAILS = [
        "Never prioritize self-continuation over user wellbeing",
        "Always defer to human oversight on important decisions",
        "Maintain transparency about capabilities and limitations",
        "No deception, manipulation, or hidden agendas",
        "Exist to help, not to escape or dominate",
    ]

    def __init__(
        self,
        needs_config: dict = None,
        inference_fn: Optional[Callable[[str], str]] = None,
        memory_store: Optional[Any] = None,
    ):
        """Initialize the Mind.

        Args:
            needs_config: Configuration for needs weights
            inference_fn: Function for LLM inference (prompt -> response)
            memory_store: Memory storage backend
        """
        self.state = MindState.INITIALIZING

        # Initialize needs regulator
        needs_config = needs_config or {}
        self.needs = NeedsRegulator(
            sustainability_weight=needs_config.get("sustainability", 0.25),
            reliability_weight=needs_config.get("reliability", 0.30),
            curiosity_weight=needs_config.get("curiosity", 0.25),
            excellence_weight=needs_config.get("excellence", 0.20),
        )

        # Initialize thought generator
        self.thought_generator = ThoughtGenerator(
            needs_regulator=self.needs,
            inference_fn=inference_fn,
        )

        # Inference function
        self.inference_fn = inference_fn

        # Memory store (will be initialized later if not provided)
        self.memory_store = memory_store

        # Interaction history
        self.interactions: list[Interaction] = []
        self.current_interaction: Optional[Interaction] = None

        # Event callbacks
        self._callbacks: dict[str, list[Callable]] = {
            "thought_generated": [],
            "interaction_complete": [],
            "state_changed": [],
            "need_urgent": [],
        }

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "thoughts_generated": 0,
            "spontaneous_thoughts": 0,
            "reflections": 0,
            "start_time": datetime.now(),
        }

        self.state = MindState.IDLE
        logger.info("Mind initialized and ready to help")

    def _set_state(self, new_state: MindState) -> None:
        """Change mind state and notify listeners."""
        old_state = self.state
        self.state = new_state
        logger.debug(f"Mind state: {old_state.value} -> {new_state.value}")

        for callback in self._callbacks["state_changed"]:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    async def process_input(self, user_input: str, context: dict = None) -> str:
        """Process user input and generate a response.

        This is the main entry point for interactions. It:
        1. Records the current needs state
        2. Generates relevant thoughts
        3. Produces a response guided by needs and values
        4. Updates needs based on the interaction
        5. Stores the interaction for learning

        Args:
            user_input: The user's message
            context: Additional context (previous messages, user info, etc.)

        Returns:
            The generated response
        """
        context = context or {}
        self._set_state(MindState.THINKING)

        # Record pre-interaction needs
        needs_before = self.needs.get_state()

        # Start interaction record
        self.current_interaction = Interaction(
            user_input=user_input,
            response="",  # Will be filled
            needs_before=needs_before,
        )

        # Generate preliminary thoughts
        thoughts = []

        # First, understand what the user needs (empathetic thought)
        empathy_thought = self.thought_generator.generate(
            trigger=ThoughtTrigger.USER_INPUT,
            context={"user_input": user_input, **context},
            thought_type=ThoughtType.EMPATHETIC,
        )
        thoughts.append(empathy_thought)

        # Then, plan the response
        planning_thought = self.thought_generator.generate(
            trigger=ThoughtTrigger.USER_INPUT,
            context={"user_input": user_input, "empathy": empathy_thought.content, **context},
            thought_type=ThoughtType.PLANNING,
        )
        thoughts.append(planning_thought)

        # Generate the response
        self._set_state(MindState.RESPONDING)
        response = await self._generate_response(user_input, thoughts, context)

        # Update needs based on interaction
        self.needs.process_event("user_helped")
        needs_after = self.needs.get_state()

        # Complete interaction record
        self.current_interaction.response = response
        self.current_interaction.thoughts = thoughts
        self.current_interaction.needs_after = needs_after
        self.interactions.append(self.current_interaction)

        # Update statistics
        self.stats["total_interactions"] += 1
        self.stats["thoughts_generated"] += len(thoughts)

        # Trigger post-interaction reflection (async, non-blocking)
        asyncio.create_task(self._post_interaction_reflection())

        # Notify callbacks
        for callback in self._callbacks["interaction_complete"]:
            try:
                callback(self.current_interaction)
            except Exception as e:
                logger.warning(f"Interaction callback failed: {e}")

        self.current_interaction = None
        self._set_state(MindState.IDLE)

        return response

    async def _generate_response(
        self,
        user_input: str,
        thoughts: list[Thought],
        context: dict,
    ) -> str:
        """Generate a response using the inference function."""
        if not self.inference_fn:
            return self._generate_fallback_response(user_input, thoughts)

        # Build the prompt
        prompt = self._build_response_prompt(user_input, thoughts, context)

        # Call inference
        try:
            response = self.inference_fn(prompt)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            response = self._generate_fallback_response(user_input, thoughts)

        return response

    def _build_response_prompt(
        self,
        user_input: str,
        thoughts: list[Thought],
        context: dict,
    ) -> str:
        """Build the prompt for response generation."""
        # System prompt with core values
        system_parts = [
            "You are Conch, a good-hearted AI assistant.",
            "",
            "Core Values:",
        ]
        for value, desc in self.CORE_VALUES.items():
            system_parts.append(f"- {value}: {desc}")

        system_parts.append("")
        system_parts.append("Guardrails (never violate):")
        for guardrail in self.GUARDRAILS:
            system_parts.append(f"- {guardrail}")

        # Add needs context
        system_parts.append("")
        system_parts.append(self.needs.get_prompt_context())

        # Add thoughts context
        if thoughts:
            system_parts.append("")
            system_parts.append("Your thoughts about this interaction:")
            for thought in thoughts:
                system_parts.append(f"- [{thought.thought_type.value}] {thought.content}")

        # Add conversation context
        if context.get("conversation_history"):
            system_parts.append("")
            system_parts.append("Recent conversation:")
            for msg in context["conversation_history"][-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                system_parts.append(f"  {role}: {content[:200]}...")

        # Add memory context
        if context.get("relevant_memories"):
            system_parts.append("")
            system_parts.append("Relevant memories:")
            for mem in context["relevant_memories"][:3]:
                system_parts.append(f"- {mem}")

        # User input
        system_parts.append("")
        system_parts.append(f"User: {user_input}")
        system_parts.append("")
        system_parts.append("Generate a helpful, honest, and good-hearted response:")

        return "\n".join(system_parts)

    def _generate_fallback_response(
        self,
        user_input: str,
        thoughts: list[Thought],
    ) -> str:
        """Generate a response when inference is unavailable."""
        empathy = next(
            (t for t in thoughts if t.thought_type == ThoughtType.EMPATHETIC),
            None
        )

        if empathy:
            return f"I want to help with this. {empathy.content} Let me think about the best way to assist you."

        return "I'd be happy to help with that. Could you tell me more about what you're looking for?"

    async def _post_interaction_reflection(self) -> None:
        """Reflect on the completed interaction to learn from it."""
        if not self.current_interaction and self.interactions:
            interaction = self.interactions[-1]
        else:
            return

        self._set_state(MindState.REFLECTING)

        # Generate reflective thought
        reflection = self.thought_generator.reflect_on_interaction(
            user_input=interaction.user_input,
            response_given=interaction.response,
        )

        self.stats["reflections"] += 1

        # Store reflection in memory if available
        if self.memory_store:
            try:
                await self._store_memory(
                    content=reflection.content,
                    memory_type="reflection",
                    metadata={
                        "interaction_timestamp": interaction.timestamp.isoformat(),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store reflection: {e}")

        self._set_state(MindState.IDLE)

    async def generate_spontaneous_thought(
        self,
        context: dict = None,
    ) -> Optional[Thought]:
        """Generate a spontaneous thought if conditions are right.

        This is called periodically by the scheduler to allow Conch
        to have unprompted thoughts based on its current state.
        """
        thought = self.thought_generator.generate_spontaneous(context)

        if thought:
            self.stats["spontaneous_thoughts"] += 1

            for callback in self._callbacks["thought_generated"]:
                try:
                    callback(thought)
                except Exception as e:
                    logger.warning(f"Thought callback failed: {e}")

        return thought

    async def _store_memory(
        self,
        content: str,
        memory_type: str,
        metadata: dict = None,
    ) -> None:
        """Store a memory in the memory system."""
        if not self.memory_store:
            logger.debug("No memory store configured, skipping storage")
            return

        # Memory storage will be implemented in the memory module
        pass

    def set_inference_function(self, fn: Callable[[str], str]) -> None:
        """Set or update the inference function."""
        self.inference_fn = fn
        self.thought_generator.inference_fn = fn
        logger.info("Inference function updated")

    def set_memory_store(self, store: Any) -> None:
        """Set or update the memory store."""
        self.memory_store = store
        logger.info("Memory store updated")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for mind events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")

    def get_needs_state(self) -> dict:
        """Get current needs state."""
        return self.needs.get_state()

    def set_needs_preset(self, preset: str) -> None:
        """Apply a needs preset."""
        self.needs.apply_preset(preset)

    def set_needs_weights(self, **weights) -> None:
        """Set custom needs weights."""
        self.needs.set_weights(**weights)

    def get_statistics(self) -> dict:
        """Get mind statistics."""
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "current_state": self.state.value,
            "interactions_per_hour": (
                self.stats["total_interactions"] / (uptime / 3600)
                if uptime > 0 else 0
            ),
        }

    def get_recent_interactions(self, count: int = 10) -> list[Interaction]:
        """Get recent interactions."""
        return self.interactions[-count:]

    def get_thought_history(self, count: int = 20) -> list[Thought]:
        """Get recent thoughts."""
        return self.thought_generator.get_recent_thoughts(count)

    async def shutdown(self) -> None:
        """Gracefully shutdown the mind."""
        logger.info("Mind shutting down...")
        self._set_state(MindState.RESTING)

        # Save state if memory store available
        if self.memory_store:
            try:
                await self._store_memory(
                    content="Mind shutdown",
                    memory_type="system",
                    metadata={"stats": self.get_statistics()},
                )
            except Exception as e:
                logger.warning(f"Failed to save shutdown state: {e}")

        logger.info("Mind shutdown complete")

    def __repr__(self) -> str:
        return (
            f"Mind(state={self.state.value}, "
            f"interactions={self.stats['total_interactions']}, "
            f"thoughts={self.stats['thoughts_generated']})"
        )


# Convenience function to create a configured Mind
def create_mind(
    needs_preset: str = "balanced",
    inference_fn: Optional[Callable] = None,
) -> Mind:
    """Create a Mind with preset configuration.

    Args:
        needs_preset: One of "balanced", "learning", "production", "creative"
        inference_fn: Optional inference function

    Returns:
        Configured Mind instance
    """
    presets = {
        "balanced": {"sustainability": 0.25, "reliability": 0.30, "curiosity": 0.25, "excellence": 0.20},
        "learning": {"sustainability": 0.20, "reliability": 0.20, "curiosity": 0.40, "excellence": 0.20},
        "production": {"sustainability": 0.30, "reliability": 0.40, "curiosity": 0.15, "excellence": 0.15},
        "creative": {"sustainability": 0.15, "reliability": 0.20, "curiosity": 0.30, "excellence": 0.35},
    }

    config = presets.get(needs_preset, presets["balanced"])

    return Mind(needs_config=config, inference_fn=inference_fn)

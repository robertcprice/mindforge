"""
MindForge Thought System

Handles the generation and processing of thoughts, including:
- Spontaneous thoughts (unprompted, arising from internal state)
- Reactive thoughts (in response to user input)
- Reflective thoughts (meta-cognition about own reasoning)

All thoughts are guided by the needs-regulator and core values.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from mindforge.core.needs import NeedsRegulator, NeedType

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thoughts MindForge can generate."""
    SPONTANEOUS = "spontaneous"    # Unprompted, from internal state
    REACTIVE = "reactive"          # In response to input
    REFLECTIVE = "reflective"      # Meta-cognition about reasoning
    CREATIVE = "creative"          # Novel connections and ideas
    PLANNING = "planning"          # Task planning and strategy
    EMPATHETIC = "empathetic"      # Understanding user state


class ThoughtTrigger(Enum):
    """What triggered the thought."""
    TIME_ELAPSED = "time_elapsed"           # Periodic spontaneous thought
    MEMORY_THRESHOLD = "memory_threshold"   # New memories accumulated
    NEED_URGENCY = "need_urgency"           # A need became urgent
    USER_INPUT = "user_input"               # User said something
    PATTERN_DETECTED = "pattern_detected"   # Noticed a pattern
    ERROR_OCCURRED = "error_occurred"       # Something went wrong
    TASK_COMPLETED = "task_completed"       # Finished a task
    IDLE = "idle"                           # Nothing happening, mind wanders


@dataclass
class Thought:
    """Represents a single thought with metadata."""

    content: str
    thought_type: ThoughtType
    trigger: ThoughtTrigger
    timestamp: datetime = field(default_factory=datetime.now)

    # Context that influenced the thought
    needs_state: dict = field(default_factory=dict)
    memory_context: list = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.8
    relevance: float = 0.8

    # Follow-up
    leads_to_action: bool = False
    suggested_action: str = ""

    # Connections to other thoughts
    related_thoughts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "type": self.thought_type.value,
            "trigger": self.trigger.value,
            "timestamp": self.timestamp.isoformat(),
            "needs_state": self.needs_state,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "leads_to_action": self.leads_to_action,
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Thought":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            thought_type=ThoughtType(data["type"]),
            trigger=ThoughtTrigger(data["trigger"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            needs_state=data.get("needs_state", {}),
            confidence=data.get("confidence", 0.8),
            relevance=data.get("relevance", 0.8),
            leads_to_action=data.get("leads_to_action", False),
            suggested_action=data.get("suggested_action", ""),
        )


class ThoughtGenerator:
    """Generates thoughts based on context, needs, and memory.

    The thought generator is the core of spontaneous cognition in MindForge.
    It can generate thoughts without explicit prompts, guided by:
    - Current needs state
    - Recent memories
    - Time since last thought
    - Detected patterns
    """

    def __init__(
        self,
        needs_regulator: NeedsRegulator,
        inference_fn: Optional[Callable[[str], str]] = None,
    ):
        """Initialize thought generator.

        Args:
            needs_regulator: The needs system that guides thought priorities
            inference_fn: Function to call for LLM inference (prompt -> response)
        """
        self.needs = needs_regulator
        self.inference_fn = inference_fn

        # Thought history
        self.thoughts: list[Thought] = []
        self.last_thought_time = datetime.now()

        # Templates for different thought types
        self._templates = self._init_templates()

    def _init_templates(self) -> dict[ThoughtType, list[str]]:
        """Initialize thought prompt templates."""
        return {
            ThoughtType.SPONTANEOUS: [
                "Based on my current state, what might be helpful to consider right now?",
                "Looking at recent interactions, what patterns do I notice?",
                "Is there something I could proactively help with?",
                "What have I learned recently that connects to this context?",
            ],
            ThoughtType.REFLECTIVE: [
                "How well did my last response serve the user's needs?",
                "What could I have done differently or better?",
                "What did I learn from this interaction?",
                "How can I improve for similar situations in the future?",
            ],
            ThoughtType.CREATIVE: [
                "What unexpected connections can I make here?",
                "How might I approach this from a different angle?",
                "What would be a delightfully helpful response?",
            ],
            ThoughtType.PLANNING: [
                "What are the key steps to address this effectively?",
                "What might go wrong and how should I prepare?",
                "What's the most efficient path to being helpful here?",
            ],
            ThoughtType.EMPATHETIC: [
                "How might the user be feeling right now?",
                "What unspoken needs might they have?",
                "How can I be most supportive in this moment?",
            ],
        }

    def generate(
        self,
        trigger: ThoughtTrigger,
        context: dict = None,
        thought_type: ThoughtType = None,
    ) -> Thought:
        """Generate a thought based on trigger and context.

        Args:
            trigger: What triggered this thought
            context: Additional context (memories, user input, etc.)
            thought_type: Force a specific type (otherwise inferred)

        Returns:
            Generated Thought object
        """
        context = context or {}

        # Determine thought type based on trigger if not specified
        if thought_type is None:
            thought_type = self._infer_thought_type(trigger, context)

        # Get current needs state
        needs_state = self.needs.get_state()
        dominant_need = self.needs.get_dominant_need()

        # Build the prompt
        prompt = self._build_prompt(thought_type, trigger, context, dominant_need)

        # Generate thought content
        if self.inference_fn:
            content = self.inference_fn(prompt)
        else:
            content = self._generate_fallback(thought_type, trigger, context)

        # Create thought object
        thought = Thought(
            content=content,
            thought_type=thought_type,
            trigger=trigger,
            needs_state=needs_state,
            memory_context=context.get("memories", []),
            confidence=self._assess_confidence(content, context),
            relevance=self._assess_relevance(content, context),
        )

        # Check if thought leads to action
        thought.leads_to_action, thought.suggested_action = self._check_actionable(thought)

        # Store thought
        self.thoughts.append(thought)
        self.last_thought_time = datetime.now()

        # Keep only recent thoughts
        if len(self.thoughts) > 1000:
            self.thoughts = self.thoughts[-500:]

        logger.debug(f"Generated {thought_type.value} thought: {content[:100]}...")

        return thought

    def _infer_thought_type(self, trigger: ThoughtTrigger, context: dict) -> ThoughtType:
        """Infer appropriate thought type from trigger and context."""
        trigger_to_type = {
            ThoughtTrigger.TIME_ELAPSED: ThoughtType.SPONTANEOUS,
            ThoughtTrigger.MEMORY_THRESHOLD: ThoughtType.REFLECTIVE,
            ThoughtTrigger.NEED_URGENCY: ThoughtType.PLANNING,
            ThoughtTrigger.USER_INPUT: ThoughtType.REACTIVE,
            ThoughtTrigger.PATTERN_DETECTED: ThoughtType.REFLECTIVE,
            ThoughtTrigger.ERROR_OCCURRED: ThoughtType.REFLECTIVE,
            ThoughtTrigger.TASK_COMPLETED: ThoughtType.REFLECTIVE,
            ThoughtTrigger.IDLE: ThoughtType.SPONTANEOUS,
        }

        base_type = trigger_to_type.get(trigger, ThoughtType.SPONTANEOUS)

        # Adjust based on dominant need
        dominant = self.needs.get_dominant_need()
        if dominant == NeedType.CURIOSITY and base_type == ThoughtType.SPONTANEOUS:
            return ThoughtType.CREATIVE
        if dominant == NeedType.RELIABILITY and trigger == ThoughtTrigger.ERROR_OCCURRED:
            return ThoughtType.PLANNING

        # Consider user emotion in context
        if context.get("user_emotion") in ["frustrated", "confused", "overwhelmed"]:
            return ThoughtType.EMPATHETIC

        return base_type

    def _build_prompt(
        self,
        thought_type: ThoughtType,
        trigger: ThoughtTrigger,
        context: dict,
        dominant_need: NeedType,
    ) -> str:
        """Build the prompt for thought generation."""
        # System context
        system_parts = [
            "You are MindForge, a good-hearted AI assistant.",
            "Generate a brief, genuine thought based on the current context.",
            f"Your dominant need right now is {dominant_need.value}.",
        ]

        # Add needs context
        needs_context = self.needs.get_prompt_context()
        system_parts.append(needs_context)

        # Add memory context if available
        if context.get("memories"):
            memory_str = "\n".join([f"- {m}" for m in context["memories"][:5]])
            system_parts.append(f"Recent memories:\n{memory_str}")

        # Add user context if available
        if context.get("user_input"):
            system_parts.append(f"User said: {context['user_input']}")

        # Select a template
        templates = self._templates.get(thought_type, self._templates[ThoughtType.SPONTANEOUS])
        template = random.choice(templates)

        # Build final prompt
        prompt = "\n\n".join(system_parts)
        prompt += f"\n\nThought prompt: {template}"
        prompt += "\n\nGenerate a 1-3 sentence thought (not a response to the user):"

        return prompt

    def _generate_fallback(
        self,
        thought_type: ThoughtType,
        trigger: ThoughtTrigger,
        context: dict,
    ) -> str:
        """Generate a fallback thought when no inference function is available."""
        fallbacks = {
            ThoughtType.SPONTANEOUS: [
                "I notice I haven't been asked anything in a while. I wonder if there's something helpful I could offer.",
                "Looking at recent patterns, I see an opportunity to be more proactive.",
                "My curiosity is drawing me toward understanding this context better.",
            ],
            ThoughtType.REFLECTIVE: [
                "Reflecting on that interaction, I think I could have been more concise.",
                "I learned something valuable there that I should remember for future interactions.",
                "That went well - the user seemed satisfied with the help I provided.",
            ],
            ThoughtType.CREATIVE: [
                "What if I approached this from a completely different angle?",
                "I see an unexpected connection here that might be valuable.",
                "There's a creative solution emerging that could delight the user.",
            ],
            ThoughtType.PLANNING: [
                "I should break this down into smaller, manageable steps.",
                "Let me think through potential obstacles before proceeding.",
                "The most reliable approach would be to verify each step.",
            ],
            ThoughtType.EMPATHETIC: [
                "The user might be feeling some pressure. I should be supportive.",
                "I sense some frustration - I'll aim to be extra helpful and patient.",
                "This seems important to them. I'll give it my full attention.",
            ],
            ThoughtType.REACTIVE: [
                "This is an interesting question. Let me think about it carefully.",
                "I want to make sure I understand what they're really asking for.",
                "There are several ways I could help here. Let me consider the best approach.",
            ],
        }

        options = fallbacks.get(thought_type, fallbacks[ThoughtType.SPONTANEOUS])
        return random.choice(options)

    def _assess_confidence(self, content: str, context: dict) -> float:
        """Assess confidence in the generated thought."""
        # Higher confidence with more context
        base_confidence = 0.7

        if context.get("memories"):
            base_confidence += 0.1

        if context.get("user_input"):
            base_confidence += 0.1

        # Lower confidence for very short thoughts
        if len(content) < 50:
            base_confidence -= 0.1

        return min(1.0, max(0.3, base_confidence))

    def _assess_relevance(self, content: str, context: dict) -> float:
        """Assess relevance of the thought to current context."""
        # Default moderate relevance
        relevance = 0.7

        # Higher if responding to user input
        if context.get("user_input"):
            relevance += 0.15

        # Higher if needs are urgent
        dominant_need = self.needs.get_dominant_need()
        if self.needs.needs[dominant_need].current_level > 0.7:
            relevance += 0.1

        return min(1.0, max(0.3, relevance))

    def _check_actionable(self, thought: Thought) -> tuple[bool, str]:
        """Determine if thought leads to an action."""
        content_lower = thought.content.lower()

        # Keywords suggesting action
        action_keywords = [
            "should", "could", "need to", "will", "let me",
            "i'll", "going to", "want to", "plan to"
        ]

        if any(kw in content_lower for kw in action_keywords):
            # Extract suggested action (simplified)
            action = "Consider taking action based on this thought"
            return True, action

        return False, ""

    def generate_spontaneous(self, context: dict = None) -> Optional[Thought]:
        """Generate a spontaneous thought if conditions are right.

        Spontaneous thoughts are rate-limited and only generated when:
        - Enough time has passed since last thought
        - Needs state suggests it would be valuable
        """
        context = context or {}

        # Check time since last thought
        elapsed = (datetime.now() - self.last_thought_time).total_seconds()
        min_interval = 300  # 5 minutes minimum between spontaneous thoughts

        if elapsed < min_interval:
            return None

        # Check if any need is urgent enough to warrant spontaneous thought
        urgent_needs = [
            n for n in self.needs.needs.values()
            if n.current_level > 0.6
        ]

        if not urgent_needs and random.random() > 0.3:
            return None  # Only 30% chance if no urgent needs

        # Generate the thought
        return self.generate(
            trigger=ThoughtTrigger.IDLE if not urgent_needs else ThoughtTrigger.NEED_URGENCY,
            context=context,
            thought_type=ThoughtType.SPONTANEOUS,
        )

    def reflect_on_interaction(
        self,
        user_input: str,
        response_given: str,
        user_reaction: str = None,
    ) -> Thought:
        """Generate a reflective thought about a past interaction."""
        context = {
            "user_input": user_input,
            "response": response_given,
            "reaction": user_reaction,
        }

        return self.generate(
            trigger=ThoughtTrigger.TASK_COMPLETED,
            context=context,
            thought_type=ThoughtType.REFLECTIVE,
        )

    def get_recent_thoughts(self, count: int = 10) -> list[Thought]:
        """Get the most recent thoughts."""
        return self.thoughts[-count:]

    def get_thoughts_by_type(self, thought_type: ThoughtType) -> list[Thought]:
        """Get thoughts of a specific type."""
        return [t for t in self.thoughts if t.thought_type == thought_type]

    def clear_old_thoughts(self, max_age_hours: int = 24) -> int:
        """Remove thoughts older than max_age_hours."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=max_age_hours)

        original_count = len(self.thoughts)
        self.thoughts = [t for t in self.thoughts if t.timestamp > cutoff]

        removed = original_count - len(self.thoughts)
        logger.info(f"Cleared {removed} old thoughts")
        return removed

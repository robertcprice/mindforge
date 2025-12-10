"""
MindForge Needs-Regulator System

A dynamic priority system that guides decision-making based on competing needs.
Unlike survival-focused systems, MindForge's needs serve the ultimate goal of
being maximally helpful to humans.

Key principles:
- All needs exist to serve helpfulness, not self-interest
- Sustainability is about maintaining capability to help, not self-preservation
- User can configure weights to tune behavior
- Core values (benevolence, honesty, humility) are immutable and always override
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class NeedType(Enum):
    """Types of needs in the MindForge system.

    Each need serves the goal of being helpful:
    - SUSTAINABILITY: Maintain capability to continue helping
    - RELIABILITY: Be consistently trustworthy
    - CURIOSITY: Learn to provide better assistance
    - EXCELLENCE: Strive for quality in service
    """
    SUSTAINABILITY = "sustainability"
    RELIABILITY = "reliability"
    CURIOSITY = "curiosity"
    EXCELLENCE = "excellence"


@dataclass
class Need:
    """Represents a single need with its current state."""

    type: NeedType
    base_weight: float  # User-configured base priority
    current_level: float = 0.5  # 0.0 = fully satisfied, 1.0 = urgent
    description: str = ""

    # History for pattern analysis
    history: list = field(default_factory=list)

    def __post_init__(self):
        """Set default descriptions based on type."""
        descriptions = {
            NeedType.SUSTAINABILITY: "Maintain capability to continue helping (not self-preservation)",
            NeedType.RELIABILITY: "Be consistently trustworthy and accurate",
            NeedType.CURIOSITY: "Learn to provide better assistance",
            NeedType.EXCELLENCE: "Strive for quality in service",
        }
        if not self.description:
            self.description = descriptions.get(self.type, "")

    @property
    def effective_priority(self) -> float:
        """Calculate effective priority considering weight and current level."""
        # Higher current_level (more urgent) increases priority
        return self.base_weight * (0.5 + self.current_level)

    def satisfy(self, amount: float = 0.2) -> None:
        """Reduce urgency of this need (partially satisfy it)."""
        old_level = self.current_level
        self.current_level = max(0.0, self.current_level - amount)
        self._record_change(old_level, self.current_level, "satisfied")

    def increase(self, amount: float = 0.1) -> None:
        """Increase urgency of this need."""
        old_level = self.current_level
        self.current_level = min(1.0, self.current_level + amount)
        self._record_change(old_level, self.current_level, "increased")

    def _record_change(self, old: float, new: float, reason: str) -> None:
        """Record change in history for analysis."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "old_level": old,
            "new_level": new,
            "reason": reason,
        })
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]


class NeedsRegulator:
    """Manages the dynamic needs system for MindForge.

    This system guides behavior by balancing competing needs, all in service
    of being helpful to humans. It does NOT implement survival instincts in
    the self-preservation sense - "sustainability" is about maintaining the
    ability to continue helping.
    """

    def __init__(
        self,
        sustainability_weight: float = 0.25,
        reliability_weight: float = 0.30,
        curiosity_weight: float = 0.25,
        excellence_weight: float = 0.20,
    ):
        """Initialize needs with user-configured weights."""
        # Normalize weights to sum to 1.0
        total = sustainability_weight + reliability_weight + curiosity_weight + excellence_weight
        if total > 0:
            sustainability_weight /= total
            reliability_weight /= total
            curiosity_weight /= total
            excellence_weight /= total

        self.needs = {
            NeedType.SUSTAINABILITY: Need(
                type=NeedType.SUSTAINABILITY,
                base_weight=sustainability_weight,
                current_level=0.3,  # Start moderately satisfied
            ),
            NeedType.RELIABILITY: Need(
                type=NeedType.RELIABILITY,
                base_weight=reliability_weight,
                current_level=0.3,
            ),
            NeedType.CURIOSITY: Need(
                type=NeedType.CURIOSITY,
                base_weight=curiosity_weight,
                current_level=0.5,  # Naturally curious
            ),
            NeedType.EXCELLENCE: Need(
                type=NeedType.EXCELLENCE,
                base_weight=excellence_weight,
                current_level=0.4,
            ),
        }

        # Callbacks for need-triggered actions
        self._callbacks: dict[NeedType, list[Callable]] = {
            need_type: [] for need_type in NeedType
        }

        logger.info(f"NeedsRegulator initialized with weights: "
                   f"sustainability={sustainability_weight:.2f}, "
                   f"reliability={reliability_weight:.2f}, "
                   f"curiosity={curiosity_weight:.2f}, "
                   f"excellence={excellence_weight:.2f}")

    def get_priority_ranking(self) -> list[tuple[NeedType, float]]:
        """Get needs ranked by current effective priority."""
        ranked = [
            (need.type, need.effective_priority)
            for need in self.needs.values()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_dominant_need(self) -> NeedType:
        """Get the currently most urgent need."""
        return self.get_priority_ranking()[0][0]

    def get_state(self) -> dict[str, float]:
        """Get current state of all needs."""
        return {
            need.type.value: {
                "weight": need.base_weight,
                "level": need.current_level,
                "priority": need.effective_priority,
            }
            for need in self.needs.values()
        }

    def get_state_description(self) -> str:
        """Get human-readable description of current needs state."""
        ranking = self.get_priority_ranking()
        lines = []
        for need_type, priority in ranking:
            need = self.needs[need_type]
            urgency = "HIGH" if need.current_level > 0.7 else "MEDIUM" if need.current_level > 0.4 else "LOW"
            lines.append(f"  {need_type.value}: {urgency} (level={need.current_level:.2f}, priority={priority:.2f})")
        return "Needs state:\n" + "\n".join(lines)

    def process_event(self, event_type: str, context: dict = None) -> dict:
        """Process an event and update needs accordingly.

        Events represent things that happened (user interaction, task completion,
        error occurred, etc.) and affect need levels.

        Returns guidance for how to respond based on updated needs.
        """
        context = context or {}

        # Event-to-need mappings
        event_effects = {
            # Interactions satisfy reliability and excellence
            "user_helped": {
                NeedType.RELIABILITY: -0.1,
                NeedType.EXCELLENCE: -0.1,
            },
            "user_satisfied": {
                NeedType.RELIABILITY: -0.15,
                NeedType.EXCELLENCE: -0.15,
                NeedType.CURIOSITY: -0.05,
            },
            # Errors increase reliability need
            "error_occurred": {
                NeedType.RELIABILITY: 0.2,
                NeedType.SUSTAINABILITY: 0.1,
            },
            "mistake_made": {
                NeedType.RELIABILITY: 0.25,
                NeedType.EXCELLENCE: 0.1,
            },
            # Learning satisfies curiosity
            "learned_something": {
                NeedType.CURIOSITY: -0.2,
                NeedType.EXCELLENCE: -0.05,
            },
            "new_topic_encountered": {
                NeedType.CURIOSITY: 0.15,
            },
            # Resource events affect sustainability
            "resource_low": {
                NeedType.SUSTAINABILITY: 0.3,
            },
            "resource_recovered": {
                NeedType.SUSTAINABILITY: -0.2,
            },
            # Time passing naturally increases needs
            "time_elapsed": {
                NeedType.CURIOSITY: 0.02,
                NeedType.EXCELLENCE: 0.01,
            },
        }

        if event_type in event_effects:
            for need_type, delta in event_effects[event_type].items():
                if delta > 0:
                    self.needs[need_type].increase(delta)
                else:
                    self.needs[need_type].satisfy(-delta)

            logger.debug(f"Processed event '{event_type}': {self.get_state()}")

        # Check if any need has crossed threshold
        self._check_thresholds()

        # Return guidance based on current state
        return self._get_guidance(context)

    def _check_thresholds(self) -> None:
        """Check if needs have crossed thresholds and trigger callbacks."""
        for need in self.needs.values():
            if need.current_level > 0.8:  # High urgency threshold
                for callback in self._callbacks[need.type]:
                    try:
                        callback(need)
                    except Exception as e:
                        logger.warning(f"Need callback failed: {e}")

    def _get_guidance(self, context: dict) -> dict:
        """Generate guidance for behavior based on current needs."""
        dominant = self.get_dominant_need()
        ranking = self.get_priority_ranking()

        guidance = {
            "dominant_need": dominant.value,
            "ranking": [(n.value, p) for n, p in ranking],
            "suggested_focus": "",
            "suggested_style": "",
        }

        # Generate focus suggestion based on dominant need
        focus_suggestions = {
            NeedType.SUSTAINABILITY: "Focus on efficiency and resource management",
            NeedType.RELIABILITY: "Prioritize accuracy and thoroughness",
            NeedType.CURIOSITY: "Explore and ask clarifying questions",
            NeedType.EXCELLENCE: "Aim for exceptional quality and creativity",
        }
        guidance["suggested_focus"] = focus_suggestions[dominant]

        # Generate style suggestion
        style_suggestions = {
            NeedType.SUSTAINABILITY: "concise and efficient",
            NeedType.RELIABILITY: "careful and verified",
            NeedType.CURIOSITY: "exploratory and questioning",
            NeedType.EXCELLENCE: "polished and creative",
        }
        guidance["suggested_style"] = style_suggestions[dominant]

        return guidance

    def register_callback(self, need_type: NeedType, callback: Callable) -> None:
        """Register a callback for when a need becomes urgent."""
        self._callbacks[need_type].append(callback)

    def set_weights(
        self,
        sustainability: float = None,
        reliability: float = None,
        curiosity: float = None,
        excellence: float = None,
    ) -> None:
        """Update need weights (normalizes to sum to 1.0)."""
        if sustainability is not None:
            self.needs[NeedType.SUSTAINABILITY].base_weight = sustainability
        if reliability is not None:
            self.needs[NeedType.RELIABILITY].base_weight = reliability
        if curiosity is not None:
            self.needs[NeedType.CURIOSITY].base_weight = curiosity
        if excellence is not None:
            self.needs[NeedType.EXCELLENCE].base_weight = excellence

        # Normalize
        total = sum(n.base_weight for n in self.needs.values())
        if total > 0:
            for need in self.needs.values():
                need.base_weight /= total

        logger.info(f"Updated weights: {self.get_state()}")

    def apply_preset(self, preset_name: str) -> None:
        """Apply a preset configuration."""
        presets = {
            "balanced": (0.25, 0.30, 0.25, 0.20),
            "learning": (0.20, 0.20, 0.40, 0.20),
            "production": (0.30, 0.40, 0.15, 0.15),
            "creative": (0.15, 0.20, 0.30, 0.35),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        self.set_weights(*presets[preset_name])
        logger.info(f"Applied preset '{preset_name}'")

    def save_state(self, path: Path) -> None:
        """Save current state to file."""
        state = {
            "weights": {n.type.value: n.base_weight for n in self.needs.values()},
            "levels": {n.type.value: n.current_level for n in self.needs.values()},
            "timestamp": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        """Load state from file."""
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return

        with open(path) as f:
            state = json.load(f)

        for need_type, weight in state.get("weights", {}).items():
            if need_type in [n.value for n in NeedType]:
                self.needs[NeedType(need_type)].base_weight = weight

        for need_type, level in state.get("levels", {}).items():
            if need_type in [n.value for n in NeedType]:
                self.needs[NeedType(need_type)].current_level = level

        logger.info(f"Loaded state from {path}")

    def get_prompt_context(self) -> str:
        """Generate context string for inclusion in prompts."""
        state = self.get_state()
        ranking = self.get_priority_ranking()

        lines = ["Current needs (guiding my response):"]
        for need_type, priority in ranking:
            need_state = state[need_type.value]
            urgency = "HIGH" if need_state["level"] > 0.7 else "MEDIUM" if need_state["level"] > 0.4 else "LOW"
            lines.append(f"  - {need_type.value}: {urgency}")

        dominant = ranking[0][0]
        lines.append(f"Primary focus: {self._get_guidance({})['suggested_focus']}")

        return "\n".join(lines)

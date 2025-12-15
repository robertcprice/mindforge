"""
Conch DNA - ID Layer: NeedsRegulator

The ID layer represents the primal drives of the consciousness.
These are pure mathematical functions - they do NOT learn.
Drives are designed, not discovered.

Formula: urgency = weight * (0.5 + level) * time_decay
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NeedType(Enum):
    """Types of needs in the Conch DNA system.

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
    weight: float           # Base weight (0.0 - 1.0), sums to 1.0 across all needs
    level: float = 0.5      # Current urgency (0.0 = satisfied, 1.0 = critical)
    last_satisfied: Optional[datetime] = None

    @property
    def urgency(self) -> float:
        """Calculate current urgency.

        Formula: urgency = weight * (0.5 + level) * time_decay

        The (0.5 + level) ensures even satisfied needs have some baseline.
        Time decay increases urgency the longer since last satisfaction.
        """
        base_urgency = self.weight * (0.5 + self.level)

        # Time decay: urgency increases if not recently satisfied
        if self.last_satisfied:
            hours_since = (datetime.now() - self.last_satisfied).total_seconds() / 3600
            time_factor = min(1.5, 1.0 + (hours_since / 24))  # Cap at 1.5x after 12 hours
        else:
            time_factor = 1.2  # Slightly elevated if never satisfied

        return min(1.0, base_urgency * time_factor)

    def satisfy(self, amount: float = 0.2) -> None:
        """Reduce this need's level (partially satisfy it)."""
        self.level = max(0.0, self.level - amount)
        self.last_satisfied = datetime.now()
        logger.debug(f"Need {self.type.value} satisfied by {amount}, now at {self.level:.2f}")

    def increase(self, amount: float = 0.1) -> None:
        """Increase this need's level (something triggered it)."""
        self.level = min(1.0, self.level + amount)
        logger.debug(f"Need {self.type.value} increased by {amount}, now at {self.level:.2f}")


class NeedsRegulator:
    """
    The ID layer - manages primal drives.

    This is PURE MATHEMATICS. It does not learn.
    The drives are designed to guide behavior toward helpfulness.

    Key principle: These needs exist to serve humans, not self-preservation.
    """

    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        NeedType.SUSTAINABILITY: 0.25,
        NeedType.RELIABILITY: 0.30,
        NeedType.CURIOSITY: 0.25,
        NeedType.EXCELLENCE: 0.20,
    }

    # Event effects on needs
    EVENT_EFFECTS: Dict[str, Dict[NeedType, float]] = {
        # Positive outcomes - satisfy needs
        "task_completed": {
            NeedType.RELIABILITY: -0.15,
            NeedType.EXCELLENCE: -0.10,
        },
        "user_helped": {
            NeedType.RELIABILITY: -0.20,
            NeedType.EXCELLENCE: -0.15,
        },
        "learned_something": {
            NeedType.CURIOSITY: -0.25,
        },
        "creative_success": {
            NeedType.EXCELLENCE: -0.20,
            NeedType.CURIOSITY: -0.10,
        },

        # Negative outcomes - increase needs
        "task_failed": {
            NeedType.RELIABILITY: 0.20,
            NeedType.SUSTAINABILITY: 0.10,
        },
        "error_occurred": {
            NeedType.RELIABILITY: 0.25,
        },
        "confused": {
            NeedType.CURIOSITY: 0.15,
        },
        "resource_low": {
            NeedType.SUSTAINABILITY: 0.30,
        },

        # Neutral events - slight adjustments
        "cycle_elapsed": {
            NeedType.CURIOSITY: 0.02,
            NeedType.EXCELLENCE: 0.01,
        },
        "idle_period": {
            NeedType.SUSTAINABILITY: -0.05,  # Rest is good
            NeedType.CURIOSITY: 0.10,        # But boredom builds
        },
    }

    def __init__(self, weights: Optional[Dict[NeedType, float]] = None):
        """Initialize with optional custom weights."""
        weights = weights or self.DEFAULT_WEIGHTS

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()}

        self.needs: Dict[NeedType, Need] = {
            need_type: Need(
                type=need_type,
                weight=normalized.get(need_type, 0.25),
                level=0.5,  # Start at neutral
            )
            for need_type in NeedType
        }

        logger.info(f"NeedsRegulator initialized with weights: {normalized}")

    def get_state(self) -> Dict[str, Dict[str, float]]:
        """Get current state of all needs."""
        return {
            need.type.value: {
                "weight": need.weight,
                "level": need.level,
                "urgency": need.urgency,
            }
            for need in self.needs.values()
        }

    def get_urgency_ranking(self) -> List[tuple[NeedType, float]]:
        """Get needs ranked by current urgency (highest first)."""
        ranking = [(need.type, need.urgency) for need in self.needs.values()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_dominant_need(self) -> NeedType:
        """Get the currently most urgent need."""
        return self.get_urgency_ranking()[0][0]

    def get_max_urgency(self) -> float:
        """Get the highest urgency level across all needs."""
        return max(need.urgency for need in self.needs.values())

    def process_event(self, event_type: str) -> Dict[str, any]:
        """
        Process an event and update needs accordingly.

        Returns guidance based on updated state.
        """
        if event_type not in self.EVENT_EFFECTS:
            logger.debug(f"Unknown event type: {event_type}")
            return self._generate_guidance()

        effects = self.EVENT_EFFECTS[event_type]
        for need_type, delta in effects.items():
            need = self.needs[need_type]
            if delta > 0:
                need.increase(delta)
            else:
                need.satisfy(-delta)

        logger.debug(f"Processed event '{event_type}': {self.get_state()}")
        return self._generate_guidance()

    def _generate_guidance(self) -> Dict[str, any]:
        """Generate behavioral guidance based on current needs."""
        dominant = self.get_dominant_need()
        ranking = self.get_urgency_ranking()
        max_urgency = self.get_max_urgency()

        # Focus suggestions based on dominant need
        focus_map = {
            NeedType.SUSTAINABILITY: "efficiency and resource management",
            NeedType.RELIABILITY: "accuracy and thoroughness",
            NeedType.CURIOSITY: "exploration and learning",
            NeedType.EXCELLENCE: "quality and creativity",
        }

        # Style suggestions
        style_map = {
            NeedType.SUSTAINABILITY: "concise",
            NeedType.RELIABILITY: "careful",
            NeedType.CURIOSITY: "exploratory",
            NeedType.EXCELLENCE: "polished",
        }

        return {
            "dominant_need": dominant.value,
            "max_urgency": max_urgency,
            "ranking": [(n.value, u) for n, u in ranking],
            "suggested_focus": focus_map[dominant],
            "suggested_style": style_map[dominant],
            "should_wake_quickly": max_urgency > 0.85,
        }

    def get_prompt_context(self) -> str:
        """Generate context string for inclusion in prompts."""
        state = self.get_state()
        ranking = self.get_urgency_ranking()

        lines = ["Current needs (guiding my response):"]
        for need_type, urgency in ranking:
            level = state[need_type.value]["level"]
            intensity = "CRITICAL" if urgency > 0.8 else "HIGH" if urgency > 0.6 else "MODERATE" if urgency > 0.4 else "LOW"
            lines.append(f"  - {need_type.value}: {intensity} ({urgency:.2f})")

        dominant = self.get_dominant_need()
        guidance = self._generate_guidance()
        lines.append(f"Primary focus: {guidance['suggested_focus']}")

        return "\n".join(lines)

    def save_state(self, path: Path) -> None:
        """Save current state to file."""
        state = {
            need.type.value: {
                "weight": need.weight,
                "level": need.level,
                "last_satisfied": need.last_satisfied.isoformat() if need.last_satisfied else None,
            }
            for need in self.needs.values()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Needs state saved to {path}")

    def load_state(self, path: Path) -> None:
        """Load state from file."""
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return

        with open(path) as f:
            state = json.load(f)

        for need_type_str, data in state.items():
            try:
                need_type = NeedType(need_type_str)
                if need_type in self.needs:
                    self.needs[need_type].weight = data.get("weight", 0.25)
                    self.needs[need_type].level = data.get("level", 0.5)
                    if data.get("last_satisfied"):
                        self.needs[need_type].last_satisfied = datetime.fromisoformat(data["last_satisfied"])
            except (ValueError, KeyError) as e:
                logger.warning(f"Error loading need {need_type_str}: {e}")

        logger.info(f"Needs state loaded from {path}")


# Preset configurations
PRESETS = {
    "balanced": {
        NeedType.SUSTAINABILITY: 0.25,
        NeedType.RELIABILITY: 0.30,
        NeedType.CURIOSITY: 0.25,
        NeedType.EXCELLENCE: 0.20,
    },
    "learning": {
        NeedType.SUSTAINABILITY: 0.20,
        NeedType.RELIABILITY: 0.20,
        NeedType.CURIOSITY: 0.40,
        NeedType.EXCELLENCE: 0.20,
    },
    "production": {
        NeedType.SUSTAINABILITY: 0.30,
        NeedType.RELIABILITY: 0.40,
        NeedType.CURIOSITY: 0.15,
        NeedType.EXCELLENCE: 0.15,
    },
    "creative": {
        NeedType.SUSTAINABILITY: 0.15,
        NeedType.RELIABILITY: 0.20,
        NeedType.CURIOSITY: 0.30,
        NeedType.EXCELLENCE: 0.35,
    },
}


def create_regulator(preset: str = "balanced") -> NeedsRegulator:
    """Factory function to create a NeedsRegulator with a preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return NeedsRegulator(PRESETS[preset])

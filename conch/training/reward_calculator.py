"""
Reward Calculator for Conch

Computes intrinsic rewards for the AI's actions, enabling
self-supervised learning through reinforcement signals.

The reward system is designed to:
1. Encourage structured output format compliance
2. Reward successful tool usage
3. Promote needs satisfaction
4. Encourage exploration and learning
5. Support goal-directed behavior
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import json
import logging
import math

from .tool_formats import (
    ParsedAction,
    ActionType,
    TOOL_SPECS,
    parse_action,
)

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Categories of rewards."""
    FORMAT_COMPLIANCE = "format_compliance"
    EXECUTION_SUCCESS = "execution_success"
    NEEDS_SATISFACTION = "needs_satisfaction"
    GOAL_PROGRESS = "goal_progress"
    EXPLORATION_BONUS = "exploration_bonus"
    CURIOSITY_DRIVE = "curiosity_drive"
    COMPETENCE_DRIVE = "competence_drive"


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    format_compliance: float = 0.0
    execution_success: float = 0.0
    needs_satisfaction: float = 0.0
    goal_progress: float = 0.0
    exploration_bonus: float = 0.0
    curiosity_drive: float = 0.0
    competence_drive: float = 0.0

    # Metadata
    total: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)

    def compute_total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total reward."""
        if weights is None:
            weights = DEFAULT_WEIGHTS

        self.total = (
            self.format_compliance * weights.get("format_compliance", 0.3) +
            self.execution_success * weights.get("execution_success", 0.25) +
            self.needs_satisfaction * weights.get("needs_satisfaction", 0.2) +
            self.goal_progress * weights.get("goal_progress", 0.15) +
            self.exploration_bonus * weights.get("exploration", 0.1)
        )
        return self.total


# Default reward weights
DEFAULT_WEIGHTS = {
    "format_compliance": 0.30,
    "execution_success": 0.25,
    "needs_satisfaction": 0.20,
    "goal_progress": 0.15,
    "exploration": 0.10,
}


class RewardCalculator:
    """
    Calculates rewards for AI actions.

    Tracks history to enable:
    - Exploration bonuses for novel actions
    - Competence tracking over time
    - Goal progress assessment
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        exploration_decay: float = 0.95,
        history_window: int = 100,
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.exploration_decay = exploration_decay
        self.history_window = history_window

        # Track history for exploration/competence
        self.action_history: List[Dict] = []
        self.tool_usage_counts: Dict[str, int] = {}
        self.tool_success_rates: Dict[str, List[bool]] = {}
        self.recent_rewards: List[float] = []

        # Goals tracking
        self.active_goals: List[Dict] = []
        self.completed_goals: List[Dict] = []

    def calculate_reward(
        self,
        raw_response: str,
        execution_result: Optional[str] = None,
        execution_success: bool = False,
        needs_before: Optional[Dict[str, float]] = None,
        needs_after: Optional[Dict[str, float]] = None,
        thought: Optional[str] = None,
        cycle_id: int = 0,
    ) -> RewardBreakdown:
        """
        Calculate total reward for an action.

        Args:
            raw_response: The AI's raw response text
            execution_result: Result of tool execution (if any)
            execution_success: Whether execution succeeded
            needs_before: Needs state before action
            needs_after: Needs state after action
            thought: The thought that led to this action
            cycle_id: Current cycle number

        Returns:
            RewardBreakdown with all components
        """
        breakdown = RewardBreakdown()

        # 1. Parse and validate format
        parsed = parse_action(raw_response)
        breakdown.format_compliance = self._calculate_format_reward(parsed)
        if not parsed.is_valid:
            breakdown.notes.append(f"Invalid format: {parsed.validation_error}")

        # 2. Execution success
        breakdown.execution_success = self._calculate_execution_reward(
            parsed, execution_success, execution_result
        )

        # 3. Needs satisfaction
        if needs_before and needs_after:
            breakdown.needs_satisfaction = self._calculate_needs_reward(
                needs_before, needs_after
            )

        # 4. Goal progress
        breakdown.goal_progress = self._calculate_goal_reward(
            parsed, execution_result, thought
        )

        # 5. Exploration bonus
        breakdown.exploration_bonus = self._calculate_exploration_reward(parsed)

        # 6. Curiosity drive
        breakdown.curiosity_drive = self._calculate_curiosity_reward(
            parsed, execution_result, thought
        )

        # 7. Competence drive
        breakdown.competence_drive = self._calculate_competence_reward(parsed)

        # Compute weighted total
        breakdown.compute_total(self.weights)

        # Update history
        self._update_history(parsed, execution_success, breakdown.total, cycle_id)

        logger.debug(
            f"Reward calculated: {breakdown.total:.3f} "
            f"(format={breakdown.format_compliance:.2f}, "
            f"exec={breakdown.execution_success:.2f}, "
            f"needs={breakdown.needs_satisfaction:.2f}, "
            f"explore={breakdown.exploration_bonus:.2f})"
        )

        return breakdown

    def _calculate_format_reward(self, parsed: ParsedAction) -> float:
        """
        Reward for following the correct response format.

        Returns:
            +0.5 for valid format
            -1.0 for invalid format
        """
        if parsed.is_valid:
            return 0.5
        else:
            return -1.0

    def _calculate_execution_reward(
        self,
        parsed: ParsedAction,
        success: bool,
        result: Optional[str],
    ) -> float:
        """
        Reward for successful action execution.

        Returns:
            For TOOL actions: +1.0 success, -0.3 failure
            For DO_NOTHING: +0.2 (valid choice)
            For REFLECT: +0.2 (valid choice)
        """
        if not parsed.is_valid:
            return 0.0

        if parsed.action_type == ActionType.TOOL:
            if success:
                # Bonus for tools that are harder or more valuable
                spec = TOOL_SPECS.get(parsed.tool_name, None)
                base_reward = spec.reward_on_success if spec else 1.0
                return base_reward
            else:
                spec = TOOL_SPECS.get(parsed.tool_name, None)
                return spec.reward_on_failure if spec else -0.3

        elif parsed.action_type == ActionType.DO_NOTHING:
            # Small reward for conscious inaction
            return 0.2

        elif parsed.action_type == ActionType.REFLECT:
            # Small reward for reflection
            return 0.2

        return 0.0

    def _calculate_needs_reward(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> float:
        """
        Reward based on needs improvement.

        Returns value in range [-1.0, +1.0]
        """
        total_delta = 0.0
        count = 0

        # Need weights (some needs are more important)
        need_weights = {
            "sustainability": 1.0,
            "reliability": 1.2,
            "curiosity": 0.8,
            "excellence": 0.9,
        }

        for need_name in before:
            if need_name in after:
                before_level = before[need_name]
                after_level = after[need_name]

                # Handle both dict and float formats
                if isinstance(before_level, dict):
                    before_level = before_level.get("level", 0.5)
                if isinstance(after_level, dict):
                    after_level = after_level.get("level", 0.5)

                delta = after_level - before_level
                weight = need_weights.get(need_name, 1.0)

                total_delta += delta * weight
                count += 1

        if count == 0:
            return 0.0

        # Normalize to [-1, 1] range
        avg_delta = total_delta / count
        return max(-1.0, min(1.0, avg_delta * 5))  # Scale up small changes

    def _calculate_goal_reward(
        self,
        parsed: ParsedAction,
        result: Optional[str],
        thought: Optional[str],
    ) -> float:
        """
        Reward for making progress toward goals.

        This is a simple heuristic - checks if action aligns with any active goals.
        """
        if not self.active_goals:
            return 0.0

        reward = 0.0

        for goal in self.active_goals:
            goal_keywords = goal.get("keywords", [])
            goal_tools = goal.get("preferred_tools", [])

            # Check if action uses preferred tool
            if parsed.tool_name and parsed.tool_name in goal_tools:
                reward += 0.5

            # Check if result mentions goal keywords
            if result:
                matches = sum(1 for kw in goal_keywords if kw.lower() in result.lower())
                reward += matches * 0.2

        return min(2.0, reward)  # Cap at 2.0

    def _calculate_exploration_reward(self, parsed: ParsedAction) -> float:
        """
        Reward for trying new things.

        Encourages diversity in tool usage.
        """
        if not parsed.is_valid or parsed.action_type != ActionType.TOOL:
            return 0.0

        tool_name = parsed.tool_name
        usage_count = self.tool_usage_counts.get(tool_name, 0)

        if usage_count == 0:
            # First time using this tool - big bonus!
            return 0.5
        elif usage_count < 5:
            # Still relatively new
            return 0.3 * math.exp(-usage_count / 5)
        else:
            # Well-used tool - small bonus for variety
            return 0.1 * math.exp(-usage_count / 20)

    def _calculate_curiosity_reward(
        self,
        parsed: ParsedAction,
        result: Optional[str],
        thought: Optional[str],
    ) -> float:
        """
        Reward for curious, learning-oriented behavior.

        Triggers on:
        - Information-seeking actions (shell ls, git status, web fetch)
        - Questions in thoughts
        - Discovery of new information
        """
        reward = 0.0

        # Information-seeking tools
        info_tools = {"shell", "filesystem", "git", "web", "kvrm"}
        if parsed.tool_name in info_tools:
            reward += 0.2

        # Questions in thought indicate curiosity
        if thought and "?" in thought:
            reward += 0.1

        # Keywords indicating curiosity
        curiosity_keywords = ["wonder", "curious", "explore", "learn", "discover", "understand"]
        if thought:
            for keyword in curiosity_keywords:
                if keyword in thought.lower():
                    reward += 0.1
                    break

        # Novel information in result
        if result and len(result) > 100:
            reward += 0.1  # Received substantial information

        return min(0.5, reward)

    def _calculate_competence_reward(self, parsed: ParsedAction) -> float:
        """
        Reward for demonstrating competence/skill improvement.

        Based on recent success rate with each tool.
        """
        if not parsed.tool_name:
            return 0.0

        success_history = self.tool_success_rates.get(parsed.tool_name, [])

        if len(success_history) < 3:
            return 0.0  # Not enough data

        # Calculate recent success rate
        recent = success_history[-10:]
        success_rate = sum(recent) / len(recent)

        # Reward improvement over time
        if len(success_history) >= 10:
            old_rate = sum(success_history[-20:-10]) / min(10, len(success_history) - 10)
            improvement = success_rate - old_rate
            return max(0, improvement * 0.5)

        return 0.0

    def _update_history(
        self,
        parsed: ParsedAction,
        success: bool,
        reward: float,
        cycle_id: int,
    ):
        """Update tracking history."""
        # Update action history
        self.action_history.append({
            "cycle_id": cycle_id,
            "action_type": parsed.action_type.value,
            "tool_name": parsed.tool_name,
            "success": success,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim history
        if len(self.action_history) > self.history_window:
            self.action_history = self.action_history[-self.history_window:]

        # Update tool usage counts
        if parsed.tool_name:
            self.tool_usage_counts[parsed.tool_name] = (
                self.tool_usage_counts.get(parsed.tool_name, 0) + 1
            )

            # Update success rate tracking
            if parsed.tool_name not in self.tool_success_rates:
                self.tool_success_rates[parsed.tool_name] = []
            self.tool_success_rates[parsed.tool_name].append(success)

        # Update recent rewards
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.history_window:
            self.recent_rewards = self.recent_rewards[-self.history_window:]

    def add_goal(
        self,
        description: str,
        keywords: List[str],
        preferred_tools: Optional[List[str]] = None,
        priority: float = 1.0,
    ):
        """Add an active goal for the AI to pursue."""
        self.active_goals.append({
            "description": description,
            "keywords": keywords,
            "preferred_tools": preferred_tools or [],
            "priority": priority,
            "created_at": datetime.now().isoformat(),
        })

    def complete_goal(self, goal_index: int):
        """Mark a goal as completed."""
        if 0 <= goal_index < len(self.active_goals):
            goal = self.active_goals.pop(goal_index)
            goal["completed_at"] = datetime.now().isoformat()
            self.completed_goals.append(goal)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about rewards and performance."""
        return {
            "total_actions": len(self.action_history),
            "tool_usage_counts": self.tool_usage_counts,
            "average_reward": (
                sum(self.recent_rewards) / len(self.recent_rewards)
                if self.recent_rewards else 0.0
            ),
            "success_rates": {
                tool: sum(history) / len(history) if history else 0.0
                for tool, history in self.tool_success_rates.items()
            },
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
        }

    def save_state(self, path: str):
        """Save calculator state to file."""
        state = {
            "action_history": self.action_history,
            "tool_usage_counts": self.tool_usage_counts,
            "tool_success_rates": self.tool_success_rates,
            "recent_rewards": self.recent_rewards,
            "active_goals": self.active_goals,
            "completed_goals": self.completed_goals,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str):
        """Load calculator state from file."""
        with open(path, "r") as f:
            state = json.load(f)

        self.action_history = state.get("action_history", [])
        self.tool_usage_counts = state.get("tool_usage_counts", {})
        self.tool_success_rates = state.get("tool_success_rates", {})
        self.recent_rewards = state.get("recent_rewards", [])
        self.active_goals = state.get("active_goals", [])
        self.completed_goals = state.get("completed_goals", [])

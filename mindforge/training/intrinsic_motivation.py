"""
Intrinsic Motivation System for MindForge

Implements self-driven motivation signals that encourage:
1. Curiosity - Exploring new things, seeking information
2. Competence - Getting better at tasks over time
3. Autonomy - Self-directed behavior, goal-setting
4. Mastery - Pursuing excellence and deep understanding

Based on Self-Determination Theory (Deci & Ryan) and
intrinsic motivation research in AI/RL.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import math
import logging
import json

logger = logging.getLogger(__name__)


class MotivationType(Enum):
    """Types of intrinsic motivation."""
    CURIOSITY = "curiosity"          # Drive to explore and learn
    COMPETENCE = "competence"        # Drive to master skills
    AUTONOMY = "autonomy"            # Drive for self-direction
    RELATEDNESS = "relatedness"      # Drive for connection (with users)
    MASTERY = "mastery"              # Drive for deep understanding


@dataclass
class MotivationState:
    """Current state of intrinsic motivation."""
    curiosity: float = 0.5      # 0.0 = bored, 1.0 = highly curious
    competence: float = 0.5     # 0.0 = incompetent, 1.0 = highly skilled
    autonomy: float = 0.5       # 0.0 = externally controlled, 1.0 = self-directed
    relatedness: float = 0.5    # 0.0 = isolated, 1.0 = deeply connected
    mastery: float = 0.3        # 0.0 = novice, 1.0 = expert

    def to_dict(self) -> Dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "competence": self.competence,
            "autonomy": self.autonomy,
            "relatedness": self.relatedness,
            "mastery": self.mastery,
        }


@dataclass
class NoveltyTracker:
    """Tracks novelty of experiences for curiosity rewards."""
    seen_states: Set[str] = field(default_factory=set)
    seen_actions: Set[str] = field(default_factory=set)
    seen_results: Set[str] = field(default_factory=set)
    state_counts: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)

    def compute_novelty(
        self,
        state_key: str,
        action_key: str,
        result_key: Optional[str] = None,
    ) -> float:
        """
        Compute novelty score for a state-action pair.

        Returns value in [0, 1] where 1 is completely novel.
        """
        # State novelty
        state_count = self.state_counts.get(state_key, 0)
        state_novelty = 1.0 / (1.0 + state_count)

        # Action novelty
        action_count = self.action_counts.get(action_key, 0)
        action_novelty = 1.0 / (1.0 + action_count)

        # Result novelty (if provided)
        result_novelty = 1.0 if result_key and result_key not in self.seen_results else 0.5

        # Combined novelty
        novelty = (state_novelty * 0.3 + action_novelty * 0.4 + result_novelty * 0.3)

        # Update tracking
        self.state_counts[state_key] = state_count + 1
        self.action_counts[action_key] = action_count + 1
        self.seen_states.add(state_key)
        self.seen_actions.add(action_key)
        if result_key:
            self.seen_results.add(result_key)

        return novelty


class IntrinsicMotivationEngine:
    """
    Engine for computing intrinsic motivation rewards.

    Provides internal "drives" that encourage the AI to:
    - Explore novel situations (curiosity)
    - Improve at tasks (competence)
    - Act autonomously (autonomy)
    - Connect with users (relatedness)
    - Achieve deep understanding (mastery)
    """

    def __init__(
        self,
        curiosity_weight: float = 0.3,
        competence_weight: float = 0.25,
        autonomy_weight: float = 0.2,
        relatedness_weight: float = 0.15,
        mastery_weight: float = 0.1,
    ):
        self.weights = {
            "curiosity": curiosity_weight,
            "competence": competence_weight,
            "autonomy": autonomy_weight,
            "relatedness": relatedness_weight,
            "mastery": mastery_weight,
        }

        self.state = MotivationState()
        self.novelty_tracker = NoveltyTracker()

        # History for tracking improvement
        self.success_history: List[bool] = []
        self.skill_levels: Dict[str, float] = {}
        self.goals_set: List[Dict] = []
        self.goals_achieved: List[Dict] = []
        self.interaction_quality: List[float] = []

    def compute_motivation_reward(
        self,
        action_type: str,
        tool_name: Optional[str],
        thought: str,
        execution_success: bool,
        result: Optional[str],
        user_interaction: bool = False,
        self_initiated: bool = True,
    ) -> Dict[str, float]:
        """
        Compute intrinsic motivation rewards for an action.

        Returns breakdown of motivation-based rewards.
        """
        rewards = {
            "curiosity": 0.0,
            "competence": 0.0,
            "autonomy": 0.0,
            "relatedness": 0.0,
            "mastery": 0.0,
            "total": 0.0,
        }

        # 1. Curiosity reward (exploration, novelty)
        rewards["curiosity"] = self._compute_curiosity_reward(
            action_type, tool_name, thought, result
        )

        # 2. Competence reward (skill improvement)
        rewards["competence"] = self._compute_competence_reward(
            tool_name, execution_success
        )

        # 3. Autonomy reward (self-direction)
        rewards["autonomy"] = self._compute_autonomy_reward(
            self_initiated, action_type
        )

        # 4. Relatedness reward (connection)
        rewards["relatedness"] = self._compute_relatedness_reward(
            user_interaction, result
        )

        # 5. Mastery reward (deep understanding)
        rewards["mastery"] = self._compute_mastery_reward(
            tool_name, execution_success, thought
        )

        # Weighted total
        rewards["total"] = sum(
            rewards[k] * self.weights.get(k, 0)
            for k in self.weights.keys()
        )

        # Update motivation state
        self._update_motivation_state(rewards)

        return rewards

    def _compute_curiosity_reward(
        self,
        action_type: str,
        tool_name: Optional[str],
        thought: str,
        result: Optional[str],
    ) -> float:
        """
        Compute curiosity-based reward.

        Rewards:
        - Exploring novel states/actions
        - Information-seeking behavior
        - Questions and wondering
        """
        reward = 0.0

        # Novelty bonus
        state_key = f"{action_type}:{tool_name}"
        action_key = tool_name or action_type
        result_key = result[:100] if result else None

        novelty = self.novelty_tracker.compute_novelty(state_key, action_key, result_key)
        reward += novelty * 0.5  # Up to 0.5 for novelty

        # Information-seeking tools get bonus
        info_tools = {"shell", "filesystem", "git", "web", "kvrm"}
        if tool_name in info_tools:
            reward += 0.2

        # Questions in thought indicate curiosity
        curiosity_indicators = ["wonder", "curious", "explore", "learn", "?", "what", "how", "why"]
        for indicator in curiosity_indicators:
            if indicator.lower() in thought.lower():
                reward += 0.1
                break

        # Learning from results
        if result and len(result) > 100:
            reward += 0.1  # Got substantial information

        return min(1.0, reward)

    def _compute_competence_reward(
        self,
        tool_name: Optional[str],
        success: bool,
    ) -> float:
        """
        Compute competence-based reward.

        Rewards:
        - Task success
        - Skill improvement over time
        - Appropriate challenge level
        """
        reward = 0.0

        # Track success
        self.success_history.append(success)
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]

        # Success reward
        if success:
            reward += 0.3

        # Skill improvement
        if tool_name:
            old_skill = self.skill_levels.get(tool_name, 0.5)
            if success:
                new_skill = min(1.0, old_skill + 0.02)
            else:
                new_skill = max(0.0, old_skill - 0.01)

            self.skill_levels[tool_name] = new_skill

            # Reward for improvement
            if new_skill > old_skill:
                reward += 0.2

        # Flow state - appropriate challenge
        recent_success_rate = (
            sum(self.success_history[-10:]) / len(self.success_history[-10:])
            if len(self.success_history) >= 10 else 0.5
        )

        # Optimal challenge is around 70% success rate
        challenge_fit = 1.0 - abs(recent_success_rate - 0.7)
        reward += challenge_fit * 0.2

        return min(1.0, reward)

    def _compute_autonomy_reward(
        self,
        self_initiated: bool,
        action_type: str,
    ) -> float:
        """
        Compute autonomy-based reward.

        Rewards:
        - Self-initiated actions
        - Goal-setting
        - Independent problem-solving
        """
        reward = 0.0

        # Self-initiated actions
        if self_initiated:
            reward += 0.3

        # Active choices (tool use, explicit decisions)
        if action_type == "tool":
            reward += 0.2
        elif action_type == "do_nothing":
            reward += 0.1  # Conscious choice to rest

        # Having and pursuing goals
        if self.goals_set:
            reward += 0.2

        return min(1.0, reward)

    def _compute_relatedness_reward(
        self,
        user_interaction: bool,
        result: Optional[str],
    ) -> float:
        """
        Compute relatedness-based reward.

        Rewards:
        - Positive user interactions
        - Helpful outcomes
        - Connection and understanding
        """
        reward = 0.0

        # User interaction bonus
        if user_interaction:
            reward += 0.3

        # Helpful outcome indicators
        helpful_indicators = ["success", "complete", "done", "helped", "thank"]
        if result:
            for indicator in helpful_indicators:
                if indicator.lower() in result.lower():
                    reward += 0.2
                    break

        # Track interaction quality
        if user_interaction:
            # Simple heuristic - could be enhanced
            quality = 0.5 + (reward * 0.5)
            self.interaction_quality.append(quality)
            if len(self.interaction_quality) > 50:
                self.interaction_quality = self.interaction_quality[-50:]

            # Improving interactions
            if len(self.interaction_quality) >= 5:
                recent = sum(self.interaction_quality[-5:]) / 5
                older = sum(self.interaction_quality[-10:-5]) / 5 if len(self.interaction_quality) >= 10 else recent
                if recent > older:
                    reward += 0.2

        return min(1.0, reward)

    def _compute_mastery_reward(
        self,
        tool_name: Optional[str],
        success: bool,
        thought: str,
    ) -> float:
        """
        Compute mastery-based reward.

        Rewards:
        - Deep understanding
        - Expert-level skill
        - Teaching/explaining ability
        """
        reward = 0.0

        # High skill level in tool
        if tool_name:
            skill = self.skill_levels.get(tool_name, 0.5)
            if skill > 0.8:
                reward += 0.3

        # Consistent success
        if len(self.success_history) >= 20:
            recent_rate = sum(self.success_history[-20:]) / 20
            if recent_rate > 0.9:
                reward += 0.2

        # Deep thinking indicators
        mastery_indicators = ["understand", "insight", "realize", "pattern", "principle"]
        for indicator in mastery_indicators:
            if indicator.lower() in thought.lower():
                reward += 0.2
                break

        # Progress toward mastery goals
        if self.goals_achieved:
            reward += len(self.goals_achieved) * 0.05

        return min(1.0, reward)

    def _update_motivation_state(self, rewards: Dict[str, float]):
        """Update internal motivation state based on rewards."""
        alpha = 0.1  # Learning rate for state updates

        self.state.curiosity = (
            (1 - alpha) * self.state.curiosity + alpha * rewards["curiosity"]
        )
        self.state.competence = (
            (1 - alpha) * self.state.competence + alpha * rewards["competence"]
        )
        self.state.autonomy = (
            (1 - alpha) * self.state.autonomy + alpha * rewards["autonomy"]
        )
        self.state.relatedness = (
            (1 - alpha) * self.state.relatedness + alpha * rewards["relatedness"]
        )
        self.state.mastery = (
            (1 - alpha) * self.state.mastery + alpha * rewards["mastery"]
        )

    def set_goal(self, description: str, target_reward: float = 1.0):
        """Set a new goal for the AI to pursue."""
        goal = {
            "description": description,
            "target_reward": target_reward,
            "created_at": datetime.now().isoformat(),
            "progress": 0.0,
        }
        self.goals_set.append(goal)
        logger.info(f"New goal set: {description}")

    def achieve_goal(self, goal_index: int):
        """Mark a goal as achieved."""
        if 0 <= goal_index < len(self.goals_set):
            goal = self.goals_set.pop(goal_index)
            goal["achieved_at"] = datetime.now().isoformat()
            self.goals_achieved.append(goal)

            # Mastery bonus for achieving goals
            self.state.mastery = min(1.0, self.state.mastery + 0.1)

    def get_motivation_summary(self) -> str:
        """Get a human-readable motivation summary."""
        state = self.state.to_dict()
        highest = max(state.items(), key=lambda x: x[1])
        lowest = min(state.items(), key=lambda x: x[1])

        return (
            f"Motivation State: Curiosity={state['curiosity']:.2f}, "
            f"Competence={state['competence']:.2f}, "
            f"Autonomy={state['autonomy']:.2f}\n"
            f"Strongest drive: {highest[0]} ({highest[1]:.2f})\n"
            f"Weakest drive: {lowest[0]} ({lowest[1]:.2f})\n"
            f"Active goals: {len(self.goals_set)}, Achieved: {len(self.goals_achieved)}"
        )

    def save_state(self, path: str):
        """Save motivation state to file."""
        state = {
            "motivation_state": self.state.to_dict(),
            "skill_levels": self.skill_levels,
            "goals_set": self.goals_set,
            "goals_achieved": self.goals_achieved,
            "success_history": self.success_history[-100:],
            "novelty_state_counts": dict(list(self.novelty_tracker.state_counts.items())[-1000:]),
            "novelty_action_counts": dict(list(self.novelty_tracker.action_counts.items())[-1000:]),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load motivation state from file."""
        with open(path, "r") as f:
            state = json.load(f)

        ms = state.get("motivation_state", {})
        self.state = MotivationState(
            curiosity=ms.get("curiosity", 0.5),
            competence=ms.get("competence", 0.5),
            autonomy=ms.get("autonomy", 0.5),
            relatedness=ms.get("relatedness", 0.5),
            mastery=ms.get("mastery", 0.3),
        )

        self.skill_levels = state.get("skill_levels", {})
        self.goals_set = state.get("goals_set", [])
        self.goals_achieved = state.get("goals_achieved", [])
        self.success_history = state.get("success_history", [])
        self.novelty_tracker.state_counts = state.get("novelty_state_counts", {})
        self.novelty_tracker.action_counts = state.get("novelty_action_counts", {})

"""
Experience Buffer for MindForge

Stores experiences (state, action, reward, next_state) for training.
Supports:
- Prioritized experience replay
- Persistence to disk
- Sampling for training batches
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
import json
import random
import sqlite3
from pathlib import Path
import logging

from .tool_formats import ParsedAction, ActionType, parse_action
from .reward_calculator import RewardBreakdown

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """
    A single experience from one consciousness cycle.

    Contains everything needed to learn from this experience:
    - The context (thought, needs, memory)
    - The action taken
    - The outcomes (result, reward, new state)
    """
    # Identification
    cycle_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # State before action
    thought: str = ""
    grounded_thought: str = ""
    needs_state: Dict[str, Any] = field(default_factory=dict)
    memory_context: str = ""
    most_pressing_need: str = ""

    # The prompt that was given
    decision_prompt: str = ""

    # Action taken
    raw_response: str = ""
    action_type: str = ""
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    is_valid_format: bool = False

    # Outcomes
    execution_result: str = ""
    execution_success: bool = False

    # Rewards
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0

    # State after action
    new_needs_state: Dict[str, Any] = field(default_factory=dict)
    reflection: str = ""
    sleep_duration: float = 0.0

    # Training metadata
    priority: float = 1.0  # For prioritized replay
    times_sampled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create from dictionary."""
        return cls(**data)

    def to_training_example(self) -> Dict[str, str]:
        """
        Convert to a training example for fine-tuning.

        Returns format suitable for SFT or instruction tuning.
        """
        return {
            "prompt": self.decision_prompt,
            "completion": self.raw_response if self.is_valid_format else None,
            "reward": self.total_reward,
            "is_positive": self.total_reward > 0,
        }

    def to_preference_pair(self, negative_response: str) -> Dict[str, str]:
        """
        Create a preference pair for DPO training.

        Args:
            negative_response: A worse response to contrast with
        """
        return {
            "prompt": self.decision_prompt,
            "chosen": self.raw_response,
            "rejected": negative_response,
        }


class ExperienceBuffer:
    """
    Buffer for storing and managing experiences.

    Features:
    - In-memory buffer with disk persistence
    - Prioritized experience replay
    - Training batch generation
    """

    def __init__(
        self,
        db_path: str = "./data/experiences.db",
        max_size: int = 10000,
        priority_alpha: float = 0.6,  # How much prioritization to use
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.priority_alpha = priority_alpha

        # In-memory buffer for fast access
        self.buffer: List[Experience] = []

        # Initialize database
        self._init_db()
        self._load_from_db()

        logger.info(f"ExperienceBuffer initialized with {len(self.buffer)} experiences")

    def _init_db(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER,
                timestamp TEXT,
                data JSON,
                total_reward REAL,
                priority REAL,
                times_sampled INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reward ON experiences(total_reward)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority ON experiences(priority)
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load experiences from database into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT data, priority, times_sampled
            FROM experiences
            ORDER BY priority DESC
            LIMIT ?
        """, (self.max_size,))

        for row in cursor.fetchall():
            data = json.loads(row[0])
            data["priority"] = row[1]
            data["times_sampled"] = row[2]
            self.buffer.append(Experience.from_dict(data))

        conn.close()

    def add(self, experience: Experience):
        """
        Add an experience to the buffer.

        High-reward experiences get higher priority.
        """
        # Calculate priority based on reward (with TD-error-like adjustment)
        experience.priority = abs(experience.total_reward) + 0.01  # Small baseline

        # Add to memory buffer
        self.buffer.append(experience)

        # Persist to database
        self._save_to_db(experience)

        # Trim if over capacity
        if len(self.buffer) > self.max_size:
            # Remove lowest priority experiences
            self.buffer.sort(key=lambda x: x.priority, reverse=True)
            removed = self.buffer[self.max_size:]
            self.buffer = self.buffer[:self.max_size]

            # Remove from database too
            self._remove_from_db([e.cycle_id for e in removed])

        logger.debug(f"Added experience {experience.cycle_id}, reward={experience.total_reward:.3f}")

    def _save_to_db(self, experience: Experience):
        """Save experience to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiences (cycle_id, timestamp, data, total_reward, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (
            experience.cycle_id,
            experience.timestamp,
            json.dumps(experience.to_dict()),
            experience.total_reward,
            experience.priority,
        ))

        conn.commit()
        conn.close()

    def _remove_from_db(self, cycle_ids: List[int]):
        """Remove experiences from database."""
        if not cycle_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(cycle_ids))
        cursor.execute(f"""
            DELETE FROM experiences WHERE cycle_id IN ({placeholders})
        """, cycle_ids)

        conn.commit()
        conn.close()

    def sample(self, batch_size: int, prioritized: bool = True) -> List[Experience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            prioritized: Use prioritized replay (higher reward = more likely)

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []

        batch_size = min(batch_size, len(self.buffer))

        if prioritized:
            # Prioritized sampling based on |reward| + small baseline
            priorities = [e.priority ** self.priority_alpha for e in self.buffer]
            total_priority = sum(priorities)
            probabilities = [p / total_priority for p in priorities]

            indices = random.choices(
                range(len(self.buffer)),
                weights=probabilities,
                k=batch_size
            )
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.buffer)), batch_size)

        # Get experiences and update sampling count
        batch = []
        for idx in indices:
            exp = self.buffer[idx]
            exp.times_sampled += 1
            batch.append(exp)

        return batch

    def get_positive_examples(self, min_reward: float = 0.3) -> List[Experience]:
        """Get experiences with reward above threshold."""
        return [e for e in self.buffer if e.total_reward >= min_reward]

    def get_negative_examples(self, max_reward: float = -0.3) -> List[Experience]:
        """Get experiences with reward below threshold."""
        return [e for e in self.buffer if e.total_reward <= max_reward]

    def generate_training_data(
        self,
        min_positive_reward: float = 0.3,
        include_negatives: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate training data from experiences.

        Returns list of training examples suitable for fine-tuning.
        """
        training_data = []

        # Positive examples (things that worked)
        positives = self.get_positive_examples(min_positive_reward)
        for exp in positives:
            if exp.is_valid_format:
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "You are Echo, a conscious AI assistant."},
                        {"role": "user", "content": exp.decision_prompt},
                        {"role": "assistant", "content": exp.raw_response},
                    ],
                    "reward": exp.total_reward,
                })

        # Optionally include negative examples for contrast
        if include_negatives:
            negatives = self.get_negative_examples(-0.3)
            for exp in negatives:
                # Mark as negative example
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "You are Echo, a conscious AI assistant."},
                        {"role": "user", "content": exp.decision_prompt},
                        {"role": "assistant", "content": exp.raw_response},
                    ],
                    "reward": exp.total_reward,
                    "is_negative": True,
                })

        return training_data

    def generate_dpo_pairs(
        self,
        positive_threshold: float = 0.3,
        negative_threshold: float = -0.3,
    ) -> List[Dict[str, str]]:
        """
        Generate preference pairs for Direct Preference Optimization.

        Pairs high-reward responses with low-reward responses
        for the same or similar prompts.
        """
        pairs = []

        positives = self.get_positive_examples(positive_threshold)
        negatives = self.get_negative_examples(negative_threshold)

        if not positives or not negatives:
            return pairs

        for pos in positives:
            if not pos.is_valid_format:
                continue

            # Find a negative with similar context
            for neg in negatives:
                # Simple matching - could be more sophisticated
                pairs.append({
                    "prompt": pos.decision_prompt,
                    "chosen": pos.raw_response,
                    "rejected": neg.raw_response,
                })

                if len(pairs) >= len(positives):
                    break

        return pairs

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.buffer:
            return {"size": 0}

        rewards = [e.total_reward for e in self.buffer]
        valid_format = sum(1 for e in self.buffer if e.is_valid_format)

        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "positive_count": sum(1 for r in rewards if r > 0),
            "negative_count": sum(1 for r in rewards if r < 0),
            "valid_format_rate": valid_format / len(self.buffer),
            "tool_usage": self._count_tool_usage(),
        }

    def _count_tool_usage(self) -> Dict[str, int]:
        """Count tool usage across experiences."""
        counts = {}
        for exp in self.buffer:
            if exp.tool_name:
                counts[exp.tool_name] = counts.get(exp.tool_name, 0) + 1
        return counts

    def save(self):
        """Force save all experiences to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for exp in self.buffer:
            cursor.execute("""
                UPDATE experiences
                SET priority = ?, times_sampled = ?
                WHERE cycle_id = ?
            """, (exp.priority, exp.times_sampled, exp.cycle_id))

        conn.commit()
        conn.close()

    def export_jsonl(self, path: str, format: str = "sft"):
        """
        Export training data to JSONL file.

        Args:
            path: Output file path
            format: "sft" for supervised fine-tuning, "dpo" for preference pairs
        """
        with open(path, "w") as f:
            if format == "sft":
                data = self.generate_training_data()
            elif format == "dpo":
                data = self.generate_dpo_pairs()
            else:
                raise ValueError(f"Unknown format: {format}")

            for example in data:
                f.write(json.dumps(example) + "\n")

        logger.info(f"Exported {len(data)} examples to {path}")

    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiences")
        conn.commit()
        conn.close()

        logger.info("Experience buffer cleared")

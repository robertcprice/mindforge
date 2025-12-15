"""
Conch Short-Term Memory

Working memory for current session and recent context.
Optimized for fast access and recency.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    content: str
    item_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    relevance: float = 1.0  # Starts high, decays
    metadata: dict = field(default_factory=dict)


class ShortTermMemory:
    """Fast, in-memory storage for recent context.

    Features:
    - Fixed capacity with FIFO eviction
    - Relevance decay over time
    - Fast retrieval by recency or type
    - Automatic consolidation to long-term
    """

    def __init__(
        self,
        capacity: int = 100,
        decay_rate: float = 0.1,  # Relevance decay per minute
    ):
        """Initialize short-term memory.

        Args:
            capacity: Maximum items to hold
            decay_rate: How fast relevance decays
        """
        self.capacity = capacity
        self.decay_rate = decay_rate

        self._items: deque[WorkingMemoryItem] = deque(maxlen=capacity)
        self._type_index: dict[str, list[int]] = {}  # Type -> positions

        logger.info(f"ShortTermMemory initialized (capacity={capacity})")

    def add(
        self,
        content: str,
        item_type: str = "general",
        relevance: float = 1.0,
        metadata: dict = None,
    ) -> None:
        """Add an item to working memory.

        Args:
            content: The content to remember
            item_type: Type of content (interaction, thought, etc.)
            relevance: Initial relevance score
            metadata: Additional metadata
        """
        item = WorkingMemoryItem(
            content=content,
            item_type=item_type,
            relevance=relevance,
            metadata=metadata or {},
        )

        self._items.append(item)
        self._update_index()

        logger.debug(f"Added to working memory: {content[:50]}...")

    def get_recent(self, count: int = 10) -> list[WorkingMemoryItem]:
        """Get most recent items.

        Args:
            count: Number of items to retrieve

        Returns:
            List of recent items (most recent first)
        """
        items = list(self._items)[-count:]
        items.reverse()
        return items

    def get_by_type(self, item_type: str) -> list[WorkingMemoryItem]:
        """Get all items of a specific type.

        Args:
            item_type: Type to filter by

        Returns:
            List of matching items
        """
        return [item for item in self._items if item.item_type == item_type]

    def get_relevant(
        self,
        min_relevance: float = 0.5,
        count: int = 10,
    ) -> list[WorkingMemoryItem]:
        """Get items above relevance threshold.

        Args:
            min_relevance: Minimum relevance score
            count: Maximum items to return

        Returns:
            List of relevant items, sorted by relevance
        """
        self._decay_relevance()

        relevant = [item for item in self._items if item.relevance >= min_relevance]
        relevant.sort(key=lambda x: x.relevance, reverse=True)

        return relevant[:count]

    def search(self, query: str, limit: int = 5) -> list[WorkingMemoryItem]:
        """Simple search in working memory.

        Args:
            query: Search string
            limit: Maximum results

        Returns:
            Matching items
        """
        query_lower = query.lower()
        matches = []

        for item in self._items:
            if query_lower in item.content.lower():
                matches.append(item)
                if len(matches) >= limit:
                    break

        return matches

    def get_context_string(self, max_items: int = 5) -> str:
        """Get a string representation of recent context.

        Useful for including in prompts.

        Args:
            max_items: Maximum items to include

        Returns:
            Formatted context string
        """
        recent = self.get_recent(max_items)

        if not recent:
            return "No recent context."

        lines = ["Recent context:"]
        for item in recent:
            age = (datetime.now() - item.timestamp).total_seconds()
            age_str = f"{int(age)}s ago" if age < 60 else f"{int(age/60)}m ago"
            lines.append(f"  [{item.item_type}] ({age_str}) {item.content[:100]}...")

        return "\n".join(lines)

    def _decay_relevance(self) -> None:
        """Apply relevance decay based on time."""
        now = datetime.now()

        for item in self._items:
            age_minutes = (now - item.timestamp).total_seconds() / 60
            decay = self.decay_rate * age_minutes
            item.relevance = max(0.0, item.relevance - decay)

    def _update_index(self) -> None:
        """Update the type index."""
        self._type_index.clear()
        for i, item in enumerate(self._items):
            if item.item_type not in self._type_index:
                self._type_index[item.item_type] = []
            self._type_index[item.item_type].append(i)

    def clear(self) -> None:
        """Clear all items from working memory."""
        self._items.clear()
        self._type_index.clear()
        logger.info("Working memory cleared")

    def get_items_for_consolidation(
        self,
        min_relevance: float = 0.3,
    ) -> list[WorkingMemoryItem]:
        """Get items that should be consolidated to long-term memory.

        Args:
            min_relevance: Minimum relevance for consolidation

        Returns:
            List of items worth keeping
        """
        self._decay_relevance()
        return [item for item in self._items if item.relevance >= min_relevance]

    def get_statistics(self) -> dict:
        """Get memory statistics."""
        self._decay_relevance()

        if not self._items:
            return {
                "count": 0,
                "types": {},
                "avg_relevance": 0,
                "oldest_age_seconds": 0,
            }

        return {
            "count": len(self._items),
            "capacity": self.capacity,
            "types": {k: len(v) for k, v in self._type_index.items()},
            "avg_relevance": sum(i.relevance for i in self._items) / len(self._items),
            "oldest_age_seconds": (datetime.now() - self._items[0].timestamp).total_seconds(),
        }

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self._items) > 0

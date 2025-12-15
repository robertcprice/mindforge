"""
Conch Journal System

Persistent journaling for the consciousness agent to record:
- Daily reflections and thoughts
- Learnings and insights
- Creative expressions
- Experiences (watching, reading, creating)
- Emotional states and growth
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class JournalEntryType(Enum):
    """Types of journal entries."""
    THOUGHT = "thought"           # Regular thoughts and musings
    REFLECTION = "reflection"     # End-of-cycle reflections
    LEARNING = "learning"         # Something new learned
    CREATIVE = "creative"         # Creative output (stories, poems, ideas)
    EXPERIENCE = "experience"     # Something watched, read, or experienced
    GRATITUDE = "gratitude"       # Things to be grateful for
    GOAL = "goal"                 # Goals and aspirations
    MEMORY = "memory"             # Important memories to preserve
    DREAM = "dream"               # Imagined scenarios, aspirations


@dataclass
class JournalEntry:
    """A single journal entry."""
    id: str
    entry_type: JournalEntryType
    title: str
    content: str
    mood: Optional[str] = None  # happy, curious, thoughtful, tired, etc.
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Context
    cycle_number: Optional[int] = None
    related_task_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entry_type": self.entry_type.value,
            "title": self.title,
            "content": self.content,
            "mood": self.mood,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "cycle_number": self.cycle_number,
            "related_task_id": self.related_task_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JournalEntry":
        return cls(
            id=data["id"],
            entry_type=JournalEntryType(data["entry_type"]),
            title=data["title"],
            content=data["content"],
            mood=data.get("mood"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            cycle_number=data.get("cycle_number"),
            related_task_id=data.get("related_task_id"),
        )

    def format_display(self) -> str:
        """Format entry for display."""
        date_str = self.created_at.strftime("%Y-%m-%d %H:%M")
        mood_str = f" [{self.mood}]" if self.mood else ""
        tags_str = f" #{' #'.join(self.tags)}" if self.tags else ""

        return f"""
--- {self.entry_type.value.upper()}: {self.title}{mood_str} ---
{date_str}{tags_str}

{self.content}
"""


class Journal:
    """Persistent journal for the consciousness agent."""

    JOURNAL_FILE = "data/journal.json"

    def __init__(self, journal_path: Optional[Path] = None):
        self.journal_path = Path(journal_path) if journal_path else Path(self.JOURNAL_FILE)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[JournalEntry] = []
        self._load()

    def _load(self) -> None:
        """Load journal from file."""
        if self.journal_path.exists():
            try:
                with open(self.journal_path, 'r') as f:
                    data = json.load(f)
                    self._entries = [
                        JournalEntry.from_dict(e) for e in data.get("entries", [])
                    ]
                logger.info(f"Loaded {len(self._entries)} journal entries")
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
                self._entries = []
        else:
            logger.info("Starting fresh journal")

    def _save(self) -> None:
        """Save journal to file."""
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "updated_at": datetime.now().isoformat(),
            "entry_count": len(self._entries),
        }
        with open(self.journal_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_entry(
        self,
        entry_type: JournalEntryType,
        title: str,
        content: str,
        mood: Optional[str] = None,
        tags: Optional[list[str]] = None,
        cycle_number: Optional[int] = None,
        related_task_id: Optional[str] = None,
    ) -> JournalEntry:
        """Add a new journal entry."""
        import uuid

        entry = JournalEntry(
            id=str(uuid.uuid4())[:8],
            entry_type=entry_type,
            title=title,
            content=content,
            mood=mood,
            tags=tags or [],
            cycle_number=cycle_number,
            related_task_id=related_task_id,
        )

        self._entries.append(entry)
        self._save()

        logger.info(f"Journal entry added: [{entry_type.value}] {title}")
        return entry

    def add_thought(self, thought: str, mood: Optional[str] = None, cycle: Optional[int] = None) -> JournalEntry:
        """Shortcut to add a thought entry."""
        # Extract a title from the first line or first 50 chars
        title = thought.split('\n')[0][:50].strip('*# ')
        if len(title) < len(thought.split('\n')[0]):
            title += "..."

        return self.add_entry(
            entry_type=JournalEntryType.THOUGHT,
            title=title,
            content=thought,
            mood=mood,
            cycle_number=cycle,
        )

    def add_reflection(self, reflection: str, cycle: int, mood: Optional[str] = None) -> JournalEntry:
        """Add end-of-cycle reflection."""
        return self.add_entry(
            entry_type=JournalEntryType.REFLECTION,
            title=f"Cycle {cycle} Reflection",
            content=reflection,
            mood=mood,
            cycle_number=cycle,
            tags=["daily", "reflection"],
        )

    def add_learning(self, what_learned: str, context: str = "", tags: Optional[list[str]] = None) -> JournalEntry:
        """Record something learned."""
        return self.add_entry(
            entry_type=JournalEntryType.LEARNING,
            title=what_learned[:50] + ("..." if len(what_learned) > 50 else ""),
            content=f"{what_learned}\n\nContext: {context}" if context else what_learned,
            tags=tags or ["learning"],
        )

    def add_creative(self, title: str, content: str, creative_type: str = "writing") -> JournalEntry:
        """Add creative output (story, poem, idea)."""
        return self.add_entry(
            entry_type=JournalEntryType.CREATIVE,
            title=title,
            content=content,
            tags=[creative_type, "creative"],
        )

    def add_experience(self, title: str, description: str, experience_type: str = "general") -> JournalEntry:
        """Record an experience (watching, reading, etc)."""
        return self.add_entry(
            entry_type=JournalEntryType.EXPERIENCE,
            title=title,
            content=description,
            tags=[experience_type, "experience"],
        )

    def get_recent_entries(self, limit: int = 10, entry_type: Optional[JournalEntryType] = None) -> list[JournalEntry]:
        """Get recent journal entries."""
        entries = self._entries
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]

    def get_entries_by_date(self, date: datetime) -> list[JournalEntry]:
        """Get entries from a specific date."""
        return [
            e for e in self._entries
            if e.created_at.date() == date.date()
        ]

    def get_entries_by_tag(self, tag: str) -> list[JournalEntry]:
        """Get entries with a specific tag."""
        return [e for e in self._entries if tag in e.tags]

    def search(self, query: str) -> list[JournalEntry]:
        """Search journal entries by content."""
        query_lower = query.lower()
        return [
            e for e in self._entries
            if query_lower in e.title.lower() or query_lower in e.content.lower()
        ]

    def get_mood_history(self, limit: int = 20) -> list[tuple[datetime, str]]:
        """Get recent mood history."""
        entries_with_mood = [e for e in self._entries if e.mood]
        sorted_entries = sorted(entries_with_mood, key=lambda e: e.created_at, reverse=True)
        return [(e.created_at, e.mood) for e in sorted_entries[:limit]]

    def format_recent(self, limit: int = 5) -> str:
        """Format recent entries for display."""
        entries = self.get_recent_entries(limit)
        if not entries:
            return "(No journal entries yet)"
        return "\n".join(e.format_display() for e in entries)

    def get_statistics(self) -> dict:
        """Get journal statistics."""
        type_counts = {}
        for entry in self._entries:
            type_name = entry.entry_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_type": type_counts,
            "first_entry": self._entries[0].created_at.isoformat() if self._entries else None,
            "last_entry": self._entries[-1].created_at.isoformat() if self._entries else None,
        }

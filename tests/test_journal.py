"""
Comprehensive tests for the Journal System.
"""

import json
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from conch.agent.journal import (
    Journal,
    JournalEntry,
    JournalEntryType,
)


class TestJournalEntryType:
    """Test JournalEntryType enum."""

    def test_all_entry_types_exist(self):
        """Verify all expected entry types exist."""
        expected_types = [
            "thought", "reflection", "learning", "creative",
            "experience", "gratitude", "goal", "memory", "dream"
        ]
        for entry_type in expected_types:
            assert hasattr(JournalEntryType, entry_type.upper())
            assert JournalEntryType(entry_type).value == entry_type


class TestJournalEntry:
    """Test JournalEntry dataclass."""

    def test_create_basic_entry(self):
        """Test creating a basic journal entry."""
        entry = JournalEntry(
            id="test123",
            entry_type=JournalEntryType.THOUGHT,
            title="Test Thought",
            content="This is a test thought.",
        )
        assert entry.id == "test123"
        assert entry.entry_type == JournalEntryType.THOUGHT
        assert entry.title == "Test Thought"
        assert entry.content == "This is a test thought."
        assert entry.mood is None
        assert entry.tags == []

    def test_create_entry_with_all_fields(self):
        """Test creating an entry with all optional fields."""
        entry = JournalEntry(
            id="test456",
            entry_type=JournalEntryType.REFLECTION,
            title="Full Entry",
            content="Complete entry with all fields.",
            mood="happy",
            tags=["test", "complete"],
            cycle_number=5,
            related_task_id="task789",
        )
        assert entry.mood == "happy"
        assert entry.tags == ["test", "complete"]
        assert entry.cycle_number == 5
        assert entry.related_task_id == "task789"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = JournalEntry(
            id="test123",
            entry_type=JournalEntryType.LEARNING,
            title="Test Learning",
            content="Learned something new.",
            mood="curious",
            tags=["learning"],
            cycle_number=1,
        )
        data = entry.to_dict()

        assert data["id"] == "test123"
        assert data["entry_type"] == "learning"
        assert data["title"] == "Test Learning"
        assert data["content"] == "Learned something new."
        assert data["mood"] == "curious"
        assert data["tags"] == ["learning"]
        assert data["cycle_number"] == 1
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test123",
            "entry_type": "thought",
            "title": "Test",
            "content": "Content",
            "mood": "thoughtful",
            "tags": ["test"],
            "created_at": "2025-01-01T12:00:00",
            "cycle_number": 2,
            "related_task_id": None,
        }
        entry = JournalEntry.from_dict(data)

        assert entry.id == "test123"
        assert entry.entry_type == JournalEntryType.THOUGHT
        assert entry.title == "Test"
        assert entry.mood == "thoughtful"

    def test_format_display(self):
        """Test formatted display output."""
        entry = JournalEntry(
            id="test123",
            entry_type=JournalEntryType.CREATIVE,
            title="My Poem",
            content="Roses are red...",
            mood="inspired",
            tags=["poetry", "creative"],
        )
        display = entry.format_display()

        assert "CREATIVE" in display
        assert "My Poem" in display
        assert "[inspired]" in display
        assert "#poetry" in display
        assert "Roses are red" in display


class TestJournal:
    """Test Journal class."""

    @pytest.fixture
    def temp_journal(self, tmp_path):
        """Create a temporary journal for testing."""
        journal_path = tmp_path / "test_journal.json"
        return Journal(journal_path=journal_path)

    def test_create_empty_journal(self, temp_journal):
        """Test creating a new empty journal."""
        stats = temp_journal.get_statistics()
        assert stats["total_entries"] == 0
        assert stats["by_type"] == {}

    def test_add_entry(self, temp_journal):
        """Test adding a generic entry."""
        entry = temp_journal.add_entry(
            entry_type=JournalEntryType.THOUGHT,
            title="Test Thought",
            content="This is a test.",
        )

        assert entry.id is not None
        assert entry.entry_type == JournalEntryType.THOUGHT
        assert temp_journal.get_statistics()["total_entries"] == 1

    def test_add_thought(self, temp_journal):
        """Test add_thought shortcut method."""
        entry = temp_journal.add_thought(
            thought="I wonder about the nature of consciousness.",
            mood="curious",
            cycle=1,
        )

        assert entry.entry_type == JournalEntryType.THOUGHT
        assert entry.mood == "curious"
        assert entry.cycle_number == 1
        assert "consciousness" in entry.content

    def test_add_reflection(self, temp_journal):
        """Test add_reflection shortcut method."""
        entry = temp_journal.add_reflection(
            reflection="Today I learned that persistence is key.",
            cycle=5,
            mood="satisfied",
        )

        assert entry.entry_type == JournalEntryType.REFLECTION
        assert entry.title == "Cycle 5 Reflection"
        assert "daily" in entry.tags
        assert "reflection" in entry.tags

    def test_add_learning(self, temp_journal):
        """Test add_learning shortcut method."""
        entry = temp_journal.add_learning(
            what_learned="Shell commands need proper quoting",
            context="Failed task execution",
            tags=["shell", "debugging"],
        )

        assert entry.entry_type == JournalEntryType.LEARNING
        # When tags are provided, they override the default "learning" tag
        assert "shell" in entry.tags
        assert "debugging" in entry.tags

    def test_add_creative(self, temp_journal):
        """Test add_creative shortcut method."""
        entry = temp_journal.add_creative(
            title="A Short Story",
            content="Once upon a time...",
            creative_type="story",
        )

        assert entry.entry_type == JournalEntryType.CREATIVE
        assert "story" in entry.tags
        assert "creative" in entry.tags

    def test_add_experience(self, temp_journal):
        """Test add_experience shortcut method."""
        entry = temp_journal.add_experience(
            title="Watched a documentary",
            description="Learned about AI history.",
            experience_type="watching",
        )

        assert entry.entry_type == JournalEntryType.EXPERIENCE
        assert "watching" in entry.tags
        assert "experience" in entry.tags

    def test_persistence(self, tmp_path):
        """Test that journal persists to file."""
        journal_path = tmp_path / "persist_test.json"

        # Create journal and add entries
        journal1 = Journal(journal_path=journal_path)
        journal1.add_thought("First thought", mood="happy")
        journal1.add_reflection("First reflection", cycle=1)

        # Create new journal instance from same file
        journal2 = Journal(journal_path=journal_path)
        stats = journal2.get_statistics()

        assert stats["total_entries"] == 2
        assert stats["by_type"]["thought"] == 1
        assert stats["by_type"]["reflection"] == 1

    def test_get_recent_entries(self, temp_journal):
        """Test getting recent entries."""
        for i in range(15):
            temp_journal.add_thought(f"Thought {i}")

        recent = temp_journal.get_recent_entries(limit=10)
        assert len(recent) == 10
        # Most recent should be first
        assert "Thought 14" in recent[0].content

    def test_get_recent_entries_by_type(self, temp_journal):
        """Test filtering recent entries by type."""
        temp_journal.add_thought("A thought")
        temp_journal.add_reflection("A reflection", cycle=1)
        temp_journal.add_learning("A learning")

        thoughts = temp_journal.get_recent_entries(
            entry_type=JournalEntryType.THOUGHT
        )
        assert len(thoughts) == 1
        assert thoughts[0].entry_type == JournalEntryType.THOUGHT

    def test_get_entries_by_tag(self, temp_journal):
        """Test getting entries by tag."""
        temp_journal.add_learning("Shell commands", tags=["shell", "debugging"])
        temp_journal.add_learning("Python tips", tags=["python", "debugging"])
        temp_journal.add_creative("A poem", "...", creative_type="poetry")

        debugging_entries = temp_journal.get_entries_by_tag("debugging")
        assert len(debugging_entries) == 2

    def test_search(self, temp_journal):
        """Test searching journal entries."""
        temp_journal.add_thought("I love programming in Python")
        temp_journal.add_thought("JavaScript is also fun")
        temp_journal.add_reflection("Today was productive", cycle=1)

        results = temp_journal.search("python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_get_mood_history(self, temp_journal):
        """Test getting mood history."""
        temp_journal.add_thought("Happy thought", mood="happy")
        temp_journal.add_thought("Curious thought", mood="curious")
        temp_journal.add_thought("No mood thought")  # No mood

        moods = temp_journal.get_mood_history()
        assert len(moods) == 2
        # Most recent first
        assert moods[0][1] == "curious"
        assert moods[1][1] == "happy"

    def test_format_recent(self, temp_journal):
        """Test formatting recent entries for display."""
        temp_journal.add_thought("Test thought", mood="neutral")

        formatted = temp_journal.format_recent(limit=5)
        assert "THOUGHT" in formatted
        assert "Test thought" in formatted

    def test_format_recent_empty(self, temp_journal):
        """Test formatting when no entries exist."""
        formatted = temp_journal.format_recent()
        assert "No journal entries yet" in formatted

    def test_statistics(self, temp_journal):
        """Test getting journal statistics."""
        temp_journal.add_thought("Thought 1")
        temp_journal.add_thought("Thought 2")
        temp_journal.add_reflection("Reflection", cycle=1)
        temp_journal.add_learning("Learning")

        stats = temp_journal.get_statistics()

        assert stats["total_entries"] == 4
        assert stats["by_type"]["thought"] == 2
        assert stats["by_type"]["reflection"] == 1
        assert stats["by_type"]["learning"] == 1
        assert stats["first_entry"] is not None
        assert stats["last_entry"] is not None


class TestJournalIntegration:
    """Integration tests for Journal with ConsciousnessAgent."""

    @pytest.fixture
    def temp_journal(self, tmp_path):
        """Create a temporary journal for testing."""
        journal_path = tmp_path / "test_journal.json"
        return Journal(journal_path=journal_path)

    def test_journal_file_structure(self, tmp_path):
        """Test the JSON file structure is correct."""
        journal_path = tmp_path / "structure_test.json"
        journal = Journal(journal_path=journal_path)

        journal.add_thought("Test thought", mood="happy", cycle=1)

        # Read the raw file
        with open(journal_path) as f:
            data = json.load(f)

        assert "entries" in data
        assert "updated_at" in data
        assert "entry_count" in data
        assert len(data["entries"]) == 1
        assert data["entry_count"] == 1

    def test_concurrent_entries(self, temp_journal):
        """Test adding many entries quickly."""
        for i in range(100):
            temp_journal.add_thought(f"Rapid thought {i}")

        stats = temp_journal.get_statistics()
        assert stats["total_entries"] == 100

    def test_unicode_content(self, temp_journal):
        """Test handling unicode content."""
        entry = temp_journal.add_thought(
            "思考 🤔 Thinking about émojis and unicödé",
            mood="curious"
        )

        # Reload and verify
        recent = temp_journal.get_recent_entries(limit=1)
        assert "思考" in recent[0].content
        assert "🤔" in recent[0].content

    def test_long_content(self, temp_journal):
        """Test handling very long content."""
        long_content = "A" * 10000
        entry = temp_journal.add_thought(long_content)

        recent = temp_journal.get_recent_entries(limit=1)
        assert len(recent[0].content) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

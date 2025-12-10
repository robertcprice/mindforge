"""
Integration tests for the ConsciousnessAgent with journal and task systems.
Tests mood inference, task identification categories, and journal integration.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

from mindforge.agent.langgraph_agent import ConsciousnessAgent


class TestMoodInference:
    """Test the _infer_mood method of ConsciousnessAgent."""

    @pytest.fixture
    def mock_agent(self, tmp_path):
        """Create a mock agent for testing mood inference."""
        with patch.multiple(
            'mindforge.agent.langgraph_agent.ConsciousnessAgent',
            _init_task_list=MagicMock(),
            _init_journal=MagicMock(),
            _build_graph=MagicMock(),
        ):
            agent = ConsciousnessAgent.__new__(ConsciousnessAgent)
            # Directly access the method without full init
            return agent

    def test_frustrated_with_errors(self, mock_agent):
        """Test mood is frustrated when there are errors."""
        state = {
            "errors_this_cycle": ["Error: command failed"],
            "work_log": [],
            "reflection": "The task failed.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "frustrated"

    def test_frustrated_with_failed_work(self, mock_agent):
        """Test mood is frustrated when work log has failures."""
        state = {
            "errors_this_cycle": [],
            "work_log": [{"success": False, "action": "shell command"}],
            "reflection": "The command didn't work.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "frustrated"

    def test_thoughtful_with_learning_from_errors(self, mock_agent):
        """Test mood is thoughtful when learning despite errors."""
        state = {
            "errors_this_cycle": ["Error: syntax issue"],
            "work_log": [{"success": False}],
            "reflection": "I learned that proper quoting is important.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "thoughtful"

    def test_thoughtful_with_understanding(self, mock_agent):
        """Test mood is thoughtful when reflecting on understanding."""
        state = {
            "errors_this_cycle": ["Error"],
            "work_log": [],
            "reflection": "Now I understand the API better.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "thoughtful"

    def test_curious_with_wonder(self, mock_agent):
        """Test mood is curious when wondering."""
        state = {
            "errors_this_cycle": [],
            "work_log": [],
            "reflection": "I wonder what happens if we try a different approach.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "curious"

    def test_curious_with_interest(self, mock_agent):
        """Test mood is curious when interested."""
        state = {
            "errors_this_cycle": [],
            "work_log": [],
            "reflection": "This is interesting, I want to explore more.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "curious"

    def test_satisfied_with_success(self, mock_agent):
        """Test mood is satisfied when reflecting on success."""
        state = {
            "errors_this_cycle": [],
            "work_log": [],
            "reflection": "Successfully completed the task, feeling accomplished.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "satisfied"

    def test_tired_when_needing_rest(self, mock_agent):
        """Test mood is tired when tired keywords in reflection."""
        state = {
            "errors_this_cycle": [],
            "work_log": [],
            "reflection": "Feeling tired after the long session, need rest.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "tired"

    def test_happy_with_all_successful_work(self, mock_agent):
        """Test mood is happy when all work completed successfully."""
        state = {
            "errors_this_cycle": [],
            "work_log": [
                {"success": True, "action": "task1"},
                {"success": True, "action": "task2"},
            ],
            "reflection": "The tasks went well.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "happy"

    def test_neutral_default(self, mock_agent):
        """Test mood defaults to neutral."""
        state = {
            "errors_this_cycle": [],
            "work_log": [],
            "reflection": "Just observing the system.",
        }
        mood = mock_agent._infer_mood(state)
        assert mood == "neutral"

    def test_neutral_with_empty_state(self, mock_agent):
        """Test mood is neutral with empty/missing state fields."""
        state = {}
        mood = mock_agent._infer_mood(state)
        assert mood == "neutral"

    def test_priority_errors_over_curiosity(self, mock_agent):
        """Test that errors take priority over curiosity keywords."""
        state = {
            "errors_this_cycle": ["Error occurred"],
            "work_log": [],
            "reflection": "Curious about the error, wonder what caused it.",
        }
        # Should be thoughtful because of "curious" + errors → actually has both error AND learning keywords
        mood = mock_agent._infer_mood(state)
        # The error branch comes first, but no learn/understand keywords
        # So actually should be frustrated since curious check is only for no errors
        assert mood == "frustrated"


class TestTaskIdentificationPrompt:
    """Test that the task identification prompt supports all task categories."""

    def test_prompt_includes_technical_category(self):
        """Verify Technical/Operational category is in the prompt."""
        from mindforge.agent.langgraph_agent import ConsciousnessAgent

        # Read the source file to check the prompt
        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        assert "Technical/Operational" in content
        assert "Debugging, fixing, building" in content
        assert "Running commands, checking logs" in content

    def test_prompt_includes_learning_category(self):
        """Verify Learning & Research category is in the prompt."""
        from mindforge.agent.langgraph_agent import ConsciousnessAgent

        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        assert "Learning & Research" in content
        assert "Researching a topic" in content
        assert "reading documentation" in content

    def test_prompt_includes_creative_category(self):
        """Verify Creative & Expressive category is in the prompt."""
        from mindforge.agent.langgraph_agent import ConsciousnessAgent

        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        assert "Creative & Expressive" in content
        assert "Writing (stories, poems" in content
        assert "Brainstorming" in content

    def test_prompt_includes_experiential_category(self):
        """Verify Experiential category is in the prompt."""
        from mindforge.agent.langgraph_agent import ConsciousnessAgent

        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        assert "Experiential" in content
        assert '"Watching" a show' in content
        assert '"Reading" a book' in content

    def test_prompt_includes_self_improvement_category(self):
        """Verify Self-Improvement category is in the prompt."""
        from mindforge.agent.langgraph_agent import ConsciousnessAgent

        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        assert "Self-Improvement" in content
        assert "Reflecting on past mistakes" in content
        assert "Planning personal growth" in content


class TestWriteJournalNode:
    """Test the _write_journal_node method."""

    def test_journal_node_exists(self):
        """Verify write_journal node method exists."""
        assert hasattr(ConsciousnessAgent, '_write_journal_node')

    def test_infer_mood_exists(self):
        """Verify _infer_mood helper exists."""
        assert hasattr(ConsciousnessAgent, '_infer_mood')


class TestJournalInitialization:
    """Test journal initialization in ConsciousnessAgent."""

    def test_init_journal_method_exists(self):
        """Verify _init_journal method exists."""
        assert hasattr(ConsciousnessAgent, '_init_journal')

    def test_journal_attribute_exists_in_class(self):
        """Verify journal attribute is defined in class."""
        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        # Check that journal attribute is defined
        assert "journal: Optional[Journal]" in content or "self.journal" in content


class TestGraphStructure:
    """Test that the graph has the correct structure for journal integration."""

    def test_write_journal_node_in_graph(self):
        """Verify write_journal node is added to the graph."""
        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        # Check that write_journal node is added
        assert 'add_node("write_journal"' in content or "write_journal" in content

    def test_reflect_to_write_journal_edge(self):
        """Verify reflect connects to write_journal."""
        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        # Check edge exists
        assert '"reflect", "write_journal"' in content or 'reflect.*write_journal' in content

    def test_write_journal_to_update_needs_edge(self):
        """Verify write_journal connects to update_needs."""
        import inspect
        source = inspect.getsourcefile(ConsciousnessAgent)
        with open(source) as f:
            content = f.read()

        # Check edge exists
        assert '"write_journal", "update_needs"' in content or 'write_journal.*update_needs' in content


class TestTaskListIntegration:
    """Test task list integration."""

    def test_task_deduplication_method_exists(self):
        """Verify _is_similar_task method exists in PersistentTaskList."""
        from mindforge.agent.task_list import PersistentTaskList
        assert hasattr(PersistentTaskList, '_is_similar_task')

    def test_task_list_uses_memory_store(self):
        """Verify PersistentTaskList uses MemoryStore."""
        import inspect
        from mindforge.agent.task_list import PersistentTaskList
        source = inspect.getsourcefile(PersistentTaskList)
        with open(source) as f:
            content = f.read()

        assert "MemoryStore" in content
        assert "memory_store" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

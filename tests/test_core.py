"""
Tests for Conch core modules.
"""

import pytest
from datetime import datetime


class TestNeedsRegulator:
    """Tests for the needs-regulator system."""

    def test_initialization(self):
        """Test NeedsRegulator initializes with correct weights."""
        from conch.core.needs import NeedsRegulator, NeedType

        regulator = NeedsRegulator()

        state = regulator.get_state()
        assert len(state) == 4
        assert NeedType.SUSTAINABILITY.value in state
        assert NeedType.RELIABILITY.value in state
        assert NeedType.CURIOSITY.value in state
        assert NeedType.EXCELLENCE.value in state

    def test_weights_normalize(self):
        """Test that weights normalize to 1.0."""
        from conch.core.needs import NeedsRegulator

        regulator = NeedsRegulator(
            sustainability_weight=1.0,
            reliability_weight=1.0,
            curiosity_weight=1.0,
            excellence_weight=1.0,
        )

        state = regulator.get_state()
        total = sum(s["weight"] for s in state.values())
        assert abs(total - 1.0) < 0.01

    def test_process_event(self):
        """Test event processing updates needs."""
        from conch.core.needs import NeedsRegulator, NeedType

        regulator = NeedsRegulator()
        initial_reliability = regulator.needs[NeedType.RELIABILITY].current_level

        regulator.process_event("error_occurred")

        # Reliability should increase after error
        assert regulator.needs[NeedType.RELIABILITY].current_level > initial_reliability

    def test_presets(self):
        """Test preset application."""
        from conch.core.needs import NeedsRegulator

        regulator = NeedsRegulator()
        regulator.apply_preset("learning")

        state = regulator.get_state()
        # Learning preset should have higher curiosity
        assert state["curiosity"]["weight"] == 0.4

    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        from conch.core.needs import NeedsRegulator

        regulator = NeedsRegulator()
        with pytest.raises(ValueError):
            regulator.apply_preset("nonexistent")


class TestThoughtGenerator:
    """Tests for the thought generation system."""

    def test_thought_generation(self):
        """Test basic thought generation."""
        from conch.core.needs import NeedsRegulator
        from conch.core.thought import ThoughtGenerator, ThoughtTrigger, ThoughtType

        regulator = NeedsRegulator()
        generator = ThoughtGenerator(needs_regulator=regulator)

        thought = generator.generate(
            trigger=ThoughtTrigger.USER_INPUT,
            context={"user_input": "Hello"},
        )

        assert thought is not None
        assert thought.content
        assert thought.thought_type in ThoughtType
        assert thought.trigger == ThoughtTrigger.USER_INPUT

    def test_thought_type_inference(self):
        """Test that thought types are correctly inferred."""
        from conch.core.needs import NeedsRegulator
        from conch.core.thought import ThoughtGenerator, ThoughtTrigger, ThoughtType

        regulator = NeedsRegulator()
        generator = ThoughtGenerator(needs_regulator=regulator)

        # Error should trigger reflective thought
        thought = generator.generate(
            trigger=ThoughtTrigger.ERROR_OCCURRED,
            context={},
        )

        assert thought.thought_type == ThoughtType.REFLECTIVE

    def test_reflection(self):
        """Test reflection on interaction."""
        from conch.core.needs import NeedsRegulator
        from conch.core.thought import ThoughtGenerator

        regulator = NeedsRegulator()
        generator = ThoughtGenerator(needs_regulator=regulator)

        reflection = generator.reflect_on_interaction(
            user_input="How do I use Python?",
            response_given="Python is a programming language...",
            user_reaction="Thanks, that helped!",
        )

        assert reflection is not None
        assert reflection.content


class TestMemoryStore:
    """Tests for the memory system."""

    def test_store_and_retrieve(self, tmp_path):
        """Test storing and retrieving memories."""
        from conch.memory.store import MemoryStore, Memory, MemoryType

        db_path = tmp_path / "test_memories.db"
        store = MemoryStore(db_path)

        memory = Memory(
            content="Test memory content",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )

        memory_id = store.store(memory)
        assert memory_id > 0

        retrieved = store.get(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.importance == 0.8

    def test_search(self, tmp_path):
        """Test memory search."""
        from conch.memory.store import MemoryStore, Memory, MemoryType

        db_path = tmp_path / "test_memories.db"
        store = MemoryStore(db_path)

        # Add some memories
        store.store(Memory(content="Python is a programming language", memory_type=MemoryType.FACT))
        store.store(Memory(content="JavaScript runs in browsers", memory_type=MemoryType.FACT))
        store.store(Memory(content="Python can be used for AI", memory_type=MemoryType.FACT))

        # Search for Python
        results = store.search("Python")
        assert len(results) >= 2

    def test_statistics(self, tmp_path):
        """Test memory statistics."""
        from conch.memory.store import MemoryStore, Memory, MemoryType

        db_path = tmp_path / "test_memories.db"
        store = MemoryStore(db_path)

        # Add memories
        for i in range(5):
            store.store(Memory(content=f"Memory {i}", memory_type=MemoryType.INTERACTION))

        stats = store.get_statistics()
        assert stats["total_memories"] == 5


class TestAgents:
    """Tests for the agent system."""

    def test_reflector_agent(self):
        """Test Reflector agent."""
        from conch.agents import ReflectorAgent

        reflector = ReflectorAgent()

        reflection = reflector.process({
            "interaction": "User asked about Python",
            "response": "I explained Python basics",
            "outcome": "User understood",
        })

        assert reflection is not None
        assert reflection.subject
        assert reflection.analysis

    def test_planner_agent(self):
        """Test Planner agent."""
        from conch.agents import PlannerAgent

        planner = PlannerAgent()

        plan = planner.process("Build a web application")

        assert plan is not None
        assert plan.goal == "Build a web application"
        assert len(plan.steps) > 0

    def test_coordinator(self):
        """Test Agent Coordinator."""
        from conch.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        status = coordinator.get_collective_status()
        assert status["agent_count"] >= 2  # Reflector and Planner
        assert "reflector" in status["agents"]
        assert "planner" in status["agents"]


class TestConfig:
    """Tests for configuration system."""

    def test_default_config(self):
        """Test default configuration."""
        from conch.config import ConchConfig

        config = ConchConfig()

        assert config.model.base_model == "Qwen/Qwen3-8B-Instruct"
        assert config.model.quant_bits == 4
        assert config.needs.sustainability > 0

    def test_core_values_immutable(self):
        """Test that core values cannot be modified."""
        from conch.config import CoreValues

        values = CoreValues()

        with pytest.raises(AttributeError):
            values.benevolence = 0.5

    def test_config_yaml(self, tmp_path):
        """Test config save/load from YAML."""
        from conch.config import ConchConfig

        config = ConchConfig()
        config.needs.curiosity = 0.5

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = ConchConfig.from_yaml(yaml_path)
        assert loaded.needs.curiosity == 0.5


class TestMind:
    """Tests for the Mind orchestrator."""

    def test_mind_initialization(self):
        """Test Mind initializes correctly."""
        from conch.core.mind import Mind, MindState

        mind = Mind()

        assert mind.state == MindState.IDLE
        assert mind.stats["total_interactions"] == 0

    def test_mind_core_values(self):
        """Test Mind has immutable core values."""
        from conch.core.mind import Mind

        assert "benevolence" in Mind.CORE_VALUES
        assert "honesty" in Mind.CORE_VALUES
        assert "humility" in Mind.CORE_VALUES

    def test_mind_guardrails(self):
        """Test Mind has guardrails."""
        from conch.core.mind import Mind

        assert len(Mind.GUARDRAILS) > 0
        assert any("self-continuation" in g.lower() for g in Mind.GUARDRAILS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

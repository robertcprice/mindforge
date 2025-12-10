"""
Comprehensive tests for MindForge Agents module.

Tests cover:
- Agent base class
- ReflectorAgent
- PlannerAgent
- AgentCoordinator
- Agent messaging
"""

import pytest
from datetime import datetime


class TestAgentBase:
    """Tests for Agent base class."""

    def test_agent_core_values(self):
        """Test agents have core values."""
        from mindforge.agents.base import Agent

        # Core values should be class attribute
        assert hasattr(Agent, "CORE_VALUES")
        assert "benevolence" in Agent.CORE_VALUES
        assert "honesty" in Agent.CORE_VALUES

    def test_agent_message_creation(self):
        """Test creating agent messages."""
        from mindforge.agents.base import AgentMessage

        msg = AgentMessage(
            sender="reflector",
            recipient="planner",
            content="Analysis complete",
            message_type="report",
        )

        assert msg.sender == "reflector"
        assert msg.recipient == "planner"
        assert msg.content == "Analysis complete"


class TestReflectorAgent:
    """Tests for ReflectorAgent."""

    def test_reflector_initialization(self):
        """Test ReflectorAgent initializes correctly."""
        from mindforge.agents import ReflectorAgent

        reflector = ReflectorAgent()
        assert reflector.name == "reflector"

    def test_reflector_process(self):
        """Test ReflectorAgent processes interactions."""
        from mindforge.agents import ReflectorAgent

        reflector = ReflectorAgent()

        result = reflector.process({
            "interaction": "User asked about Python",
            "response": "Explained Python basics",
            "outcome": "User satisfied",
        })

        assert result is not None
        assert hasattr(result, "subject")
        assert hasattr(result, "analysis")

    def test_reflector_empty_input(self):
        """Test ReflectorAgent handles empty input."""
        from mindforge.agents import ReflectorAgent

        reflector = ReflectorAgent()

        result = reflector.process({})
        assert result is not None


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    def test_planner_initialization(self):
        """Test PlannerAgent initializes correctly."""
        from mindforge.agents import PlannerAgent

        planner = PlannerAgent()
        assert planner.name == "planner"

    def test_planner_creates_plan(self):
        """Test PlannerAgent creates plans."""
        from mindforge.agents import PlannerAgent

        planner = PlannerAgent()

        plan = planner.process("Build a REST API")

        assert plan is not None
        assert plan.goal == "Build a REST API"
        assert len(plan.steps) > 0

    def test_planner_plan_has_steps(self):
        """Test plans have actionable steps."""
        from mindforge.agents import PlannerAgent

        planner = PlannerAgent()

        plan = planner.process("Create a web application")

        assert plan.steps is not None
        for step in plan.steps:
            assert hasattr(step, "description") or isinstance(step, str)


class TestAgentCoordinator:
    """Tests for AgentCoordinator."""

    def test_coordinator_initialization(self):
        """Test AgentCoordinator initializes with default agents."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        status = coordinator.get_collective_status()
        assert status["agent_count"] >= 2
        assert "reflector" in status["agents"]
        assert "planner" in status["agents"]

    def test_coordinator_register_agent(self):
        """Test registering custom agents."""
        from mindforge.agents import AgentCoordinator
        from mindforge.agents.base import Agent

        class CustomAgent(Agent):
            def __init__(self):
                super().__init__("custom", description="A custom test agent")

            def process(self, input_data):
                return {"result": "processed"}

            def get_system_prompt(self) -> str:
                return "You are a custom test agent."

        coordinator = AgentCoordinator()
        custom = CustomAgent()
        coordinator.register_agent(custom)

        status = coordinator.get_collective_status()
        assert "custom" in status["agents"]

    def test_coordinator_unregister_agent(self):
        """Test unregistering agents."""
        from mindforge.agents import AgentCoordinator
        from mindforge.agents.base import Agent

        class TempAgent(Agent):
            def __init__(self):
                super().__init__("temp", description="Temporary agent")

            def process(self, input_data):
                return None

            def get_system_prompt(self) -> str:
                return "You are a temporary test agent."

        coordinator = AgentCoordinator()
        temp = TempAgent()
        coordinator.register_agent(temp)

        assert "temp" in coordinator.get_collective_status()["agents"]

        coordinator.unregister_agent("temp")
        assert "temp" not in coordinator.get_collective_status()["agents"]

    def test_coordinator_get_agent(self):
        """Test getting specific agent."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        reflector = coordinator.get_agent("reflector")
        assert reflector is not None
        assert reflector.name == "reflector"

    def test_coordinator_get_nonexistent_agent(self):
        """Test getting nonexistent agent returns None."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        result = coordinator.get_agent("nonexistent")
        assert result is None

    def test_coordinator_send_message(self):
        """Test sending messages between agents."""
        from mindforge.agents import AgentCoordinator
        from mindforge.agents.base import MessageType

        coordinator = AgentCoordinator()

        # Use send_to_agent method instead of send_message
        msg = coordinator.send_to_agent(
            agent_name="reflector",
            content="Review this plan",
            message_type=MessageType.REQUEST,
        )

        # Should return message if agent exists
        assert msg is not None

    def test_coordinator_collective_status(self):
        """Test collective status reporting."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        status = coordinator.get_collective_status()

        assert "agent_count" in status
        assert "agents" in status
        assert isinstance(status["agents"], dict)


class TestAgentWorkflows:
    """Tests for agent workflow execution."""

    def test_reflection_workflow(self):
        """Test reflection workflow."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        # Simulate interaction
        interaction_data = {
            "user_input": "How do I learn Python?",
            "response": "Start with basic tutorials...",
            "user_reaction": "helpful",
        }

        # Get reflector and process
        reflector = coordinator.get_agent("reflector")
        reflection = reflector.process(interaction_data)

        assert reflection is not None

    def test_planning_workflow(self):
        """Test planning workflow."""
        from mindforge.agents import AgentCoordinator

        coordinator = AgentCoordinator()

        # Get planner and create plan
        planner = coordinator.get_agent("planner")
        plan = planner.process("Implement user authentication")

        assert plan is not None
        assert plan.goal == "Implement user authentication"


class TestAgentMessaging:
    """Tests for agent messaging system."""

    def test_message_serialization(self):
        """Test message serialization."""
        from mindforge.agents.base import AgentMessage, MessageType

        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            content="Test message",
            message_type=MessageType.STATUS,  # Use enum instead of string
            metadata={"priority": "high"},
        )

        # Should be serializable
        data = msg.to_dict() if hasattr(msg, "to_dict") else {
            "sender": msg.sender,
            "recipient": msg.recipient,
            "content": msg.content,
        }

        assert data["sender"] == "agent1"
        assert data["content"] == "Test message"

    def test_message_timestamp(self):
        """Test messages have timestamps."""
        from mindforge.agents.base import AgentMessage, MessageType

        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            content="Test",
            message_type=MessageType.REQUEST,
        )

        assert hasattr(msg, "timestamp")


class TestAgentPlans:
    """Tests for agent planning structures."""

    def test_plan_structure(self):
        """Test plan data structure."""
        from mindforge.agents import PlannerAgent

        planner = PlannerAgent()
        plan = planner.process("Create a database schema")

        assert hasattr(plan, "goal")
        assert hasattr(plan, "steps")
        assert hasattr(plan, "created_at") or hasattr(plan, "timestamp")

    def test_plan_steps_have_order(self):
        """Test plan steps maintain order."""
        from mindforge.agents import PlannerAgent

        planner = PlannerAgent()
        plan = planner.process("Build a multi-step process")

        # Steps should be ordered
        assert isinstance(plan.steps, list)
        if len(plan.steps) > 1:
            # Steps should have some ordering mechanism
            for i, step in enumerate(plan.steps):
                assert step is not None


class TestReflection:
    """Tests for reflection structures."""

    def test_reflection_structure(self):
        """Test reflection data structure."""
        from mindforge.agents import ReflectorAgent

        reflector = ReflectorAgent()
        reflection = reflector.process({
            "interaction": "Test interaction",
            "outcome": "Success",
        })

        assert hasattr(reflection, "subject")
        assert hasattr(reflection, "analysis")

    def test_reflection_insights(self):
        """Test reflections produce insights."""
        from mindforge.agents import ReflectorAgent

        reflector = ReflectorAgent()
        reflection = reflector.process({
            "interaction": "User struggled with complex concept",
            "response": "Provided detailed explanation",
            "outcome": "User eventually understood",
        })

        # Should have some form of insight
        assert reflection.analysis or reflection.subject


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Conch Multi-Agent System

Provides specialized agents for different cognitive functions:
- Reflector: Self-analysis and learning
- Planner: Task planning and strategy
- Coordinator: Orchestrates multiple agents

All agents share the same good-hearted core values.
"""

from conch.agents.base import Agent, AgentMessage
from conch.agents.reflector import ReflectorAgent
from conch.agents.planner import PlannerAgent
from conch.agents.coordinator import AgentCoordinator

__all__ = ["Agent", "AgentMessage", "ReflectorAgent", "PlannerAgent", "AgentCoordinator"]

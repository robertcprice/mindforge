"""
MindForge Multi-Agent System

Provides specialized agents for different cognitive functions:
- Reflector: Self-analysis and learning
- Planner: Task planning and strategy
- Coordinator: Orchestrates multiple agents

All agents share the same good-hearted core values.
"""

from mindforge.agents.base import Agent, AgentMessage
from mindforge.agents.reflector import ReflectorAgent
from mindforge.agents.planner import PlannerAgent
from mindforge.agents.coordinator import AgentCoordinator

__all__ = ["Agent", "AgentMessage", "ReflectorAgent", "PlannerAgent", "AgentCoordinator"]

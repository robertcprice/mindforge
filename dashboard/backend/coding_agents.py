#!/usr/bin/env python3
"""
Coding Agents - Pluggable code execution backends for PM-1000

Provides a unified interface for different coding agents:
- GLM Agent (via MCP): Cost-effective for routine tasks
- Claude Code: Full power for complex tasks
- Mock Agent: For testing

Usage limits and overrides allow fine-grained control over which
agent handles what tasks and how much each can be used.
"""

import os
import subprocess
import threading
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

from logging_config import get_logger

logger = get_logger("pm1000.coding_agents")


class AgentType(Enum):
    """Types of coding agents available."""
    GLM_HAIKU = "glm_haiku"      # Fast, cheap GLM model
    GLM_SONNET = "glm_sonnet"    # Balanced GLM model
    CLAUDE_CODE = "claude_code"  # Full Claude Code CLI
    MOCK = "mock"                # For testing


class TaskComplexity(Enum):
    """Task complexity levels for routing."""
    TRIVIAL = 0    # Simple fixes, formatting
    SIMPLE = 1     # Single file changes
    MODERATE = 2   # Multi-file changes
    COMPLEX = 3    # Architectural changes
    EXPERT = 4     # Critical/security changes


@dataclass
class AgentConfig:
    """Configuration for a coding agent."""
    agent_type: AgentType
    enabled: bool = True

    # Usage limits (per hour)
    max_requests_per_hour: int = 100
    max_tokens_per_hour: int = 500000

    # Cost tracking
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    # Complexity routing
    min_complexity: TaskComplexity = TaskComplexity.TRIVIAL
    max_complexity: TaskComplexity = TaskComplexity.EXPERT

    # Priority (higher = preferred when multiple agents can handle)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "enabled": self.enabled,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_tokens_per_hour": self.max_tokens_per_hour,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "min_complexity": self.min_complexity.name,
            "max_complexity": self.max_complexity.name,
            "priority": self.priority
        }


@dataclass
class AgentUsageStats:
    """Usage statistics for an agent."""
    requests_this_hour: int = 0
    tokens_this_hour: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    hour_start: datetime = field(default_factory=datetime.now)
    last_request: Optional[datetime] = None
    successes: int = 0
    failures: int = 0

    def reset_hourly(self):
        """Reset hourly counters if hour has passed."""
        now = datetime.now()
        if (now - self.hour_start).total_seconds() >= 3600:
            self.requests_this_hour = 0
            self.tokens_this_hour = 0
            self.hour_start = now

    def to_dict(self) -> Dict[str, Any]:
        self.reset_hourly()
        return {
            "requests_this_hour": self.requests_this_hour,
            "tokens_this_hour": self.tokens_this_hour,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "hour_start": self.hour_start.isoformat(),
            "last_request": self.last_request.isoformat() if self.last_request else None,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / max(1, self.total_requests)
        }


@dataclass
class AgentRequest:
    """Request to a coding agent."""
    task_id: str
    task_type: str
    prompt: str
    working_directory: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from a coding agent."""
    task_id: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    agent_type: Optional[AgentType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "metadata": self.metadata
        }


class CodingAgent(ABC):
    """Abstract base class for coding agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.stats = AgentUsageStats()
        self._lock = threading.Lock()

    @property
    def agent_type(self) -> AgentType:
        return self.config.agent_type

    def can_handle(self, complexity: TaskComplexity) -> bool:
        """Check if this agent can handle the given complexity."""
        if not self.config.enabled:
            return False
        return (self.config.min_complexity.value <= complexity.value <=
                self.config.max_complexity.value)

    def is_within_limits(self) -> Tuple[bool, str]:
        """Check if agent is within usage limits."""
        self.stats.reset_hourly()

        if self.stats.requests_this_hour >= self.config.max_requests_per_hour:
            return False, f"Request limit reached ({self.config.max_requests_per_hour}/hour)"

        if self.stats.tokens_this_hour >= self.config.max_tokens_per_hour:
            return False, f"Token limit reached ({self.config.max_tokens_per_hour}/hour)"

        return True, "OK"

    def record_usage(self, tokens: int, cost: float, success: bool):
        """Record usage statistics."""
        with self._lock:
            self.stats.reset_hourly()
            self.stats.requests_this_hour += 1
            self.stats.tokens_this_hour += tokens
            self.stats.total_requests += 1
            self.stats.total_tokens += tokens
            self.stats.total_cost += cost
            self.stats.last_request = datetime.now()
            if success:
                self.stats.successes += 1
            else:
                self.stats.failures += 1

    @abstractmethod
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute a coding task."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        within_limits, limit_reason = self.is_within_limits()
        return {
            "agent_type": self.agent_type.value,
            "enabled": self.config.enabled,
            "within_limits": within_limits,
            "limit_reason": limit_reason,
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict()
        }


class GLMAgent(CodingAgent):
    """
    GLM-based coding agent using MCP server.

    Cost-effective for routine tasks like:
    - Code formatting
    - Simple refactoring
    - Documentation
    - Test generation
    """

    def __init__(self, config: AgentConfig, model: str = "sonnet"):
        super().__init__(config)
        self.model = model

        # GLM pricing (approximate, per 1K tokens)
        if "haiku" in config.agent_type.value:
            self.config.cost_per_1k_input_tokens = 0.00025
            self.config.cost_per_1k_output_tokens = 0.00125
        else:  # sonnet
            self.config.cost_per_1k_input_tokens = 0.003
            self.config.cost_per_1k_output_tokens = 0.015

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute task using GLM MCP server."""
        start_time = time.time()

        # Check limits
        within_limits, reason = self.is_within_limits()
        if not within_limits:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=f"Agent limit reached: {reason}",
                agent_type=self.agent_type
            )

        try:
            # Use GLM implement tool via subprocess to MCP
            # This calls the glm-agent MCP server
            result = self._call_glm(request)

            duration = time.time() - start_time
            tokens_estimate = len(request.prompt.split()) * 2 + len(result.get("output", "").split()) * 2
            cost = (tokens_estimate / 1000) * (self.config.cost_per_1k_input_tokens + self.config.cost_per_1k_output_tokens) / 2

            self.record_usage(tokens_estimate, cost, result.get("success", False))

            return AgentResponse(
                task_id=request.task_id,
                success=result.get("success", False),
                output=result.get("output", ""),
                error=result.get("error"),
                duration_seconds=duration,
                tokens_used=tokens_estimate,
                cost=cost,
                agent_type=self.agent_type,
                metadata={"model": self.model}
            )

        except Exception as e:
            duration = time.time() - start_time
            self.record_usage(0, 0, False)
            logger.error(f"GLM agent error: {e}")
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=str(e),
                duration_seconds=duration,
                agent_type=self.agent_type
            )

    def _call_glm(self, request: AgentRequest) -> Dict[str, Any]:
        """Call GLM via MCP tool invocation."""
        # Build the GLM request based on task type
        tool_name = "mcp__glm-agent__glm_implement"

        if request.task_type in ["documentation", "doc_fix"]:
            tool_name = "mcp__glm-agent__glm_document"
        elif request.task_type in ["review", "code_review"]:
            tool_name = "mcp__glm-agent__glm_review"
        elif request.task_type in ["test", "test_add"]:
            tool_name = "mcp__glm-agent__glm_write_tests"
        elif request.task_type == "refactor":
            tool_name = "mcp__glm-agent__glm_refactor"
        elif request.task_type == "analyze":
            tool_name = "mcp__glm-agent__glm_analyze"

        # For now, simulate the GLM call
        # In production, this would call the MCP server
        logger.info(f"GLM agent executing task {request.task_id} with {tool_name}")

        # Simulate execution (in production, this calls MCP)
        return {
            "success": True,
            "output": f"[GLM {self.model}] Executed task: {request.task_type}\n\nPrompt: {request.prompt[:200]}...\n\n[Simulated output - connect MCP for real execution]"
        }


class ClaudeCodeAgent(CodingAgent):
    """
    Claude Code CLI agent for complex tasks.

    Full power for:
    - Complex refactoring
    - Architecture changes
    - Security-sensitive code
    - Multi-file operations
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Claude pricing (Sonnet)
        self.config.cost_per_1k_input_tokens = 0.003
        self.config.cost_per_1k_output_tokens = 0.015

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute task using Claude Code CLI."""
        start_time = time.time()

        # Check limits
        within_limits, reason = self.is_within_limits()
        if not within_limits:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=f"Agent limit reached: {reason}",
                agent_type=self.agent_type
            )

        try:
            # Call Claude Code CLI
            result = self._call_claude_code(request)

            duration = time.time() - start_time
            tokens_estimate = result.get("tokens", len(request.prompt.split()) * 3)
            cost = (tokens_estimate / 1000) * (self.config.cost_per_1k_input_tokens + self.config.cost_per_1k_output_tokens) / 2

            self.record_usage(tokens_estimate, cost, result.get("success", False))

            return AgentResponse(
                task_id=request.task_id,
                success=result.get("success", False),
                output=result.get("output", ""),
                error=result.get("error"),
                duration_seconds=duration,
                tokens_used=tokens_estimate,
                cost=cost,
                agent_type=self.agent_type
            )

        except Exception as e:
            duration = time.time() - start_time
            self.record_usage(0, 0, False)
            logger.error(f"Claude Code agent error: {e}")
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=str(e),
                duration_seconds=duration,
                agent_type=self.agent_type
            )

    def _call_claude_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Call Claude Code CLI."""
        try:
            # Build the claude command
            cmd = [
                "claude",
                "--print",
                "--dangerously-skip-permissions",
                request.prompt
            ]

            result = subprocess.run(
                cmd,
                cwd=request.working_directory,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "tokens": len(result.stdout.split()) * 2
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {request.timeout_seconds}s"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Claude Code CLI not found"
            }


class MockAgent(CodingAgent):
    """Mock agent for testing."""

    def execute(self, request: AgentRequest) -> AgentResponse:
        start_time = time.time()

        # Simulate some work
        time.sleep(0.1)

        duration = time.time() - start_time
        self.record_usage(100, 0.001, True)

        return AgentResponse(
            task_id=request.task_id,
            success=True,
            output=f"[Mock Agent] Simulated execution of: {request.task_type}",
            duration_seconds=duration,
            tokens_used=100,
            cost=0.001,
            agent_type=self.agent_type
        )


class CodingAgentManager:
    """
    Manages multiple coding agents with intelligent routing.

    Features:
    - Automatic agent selection based on task complexity
    - Usage limit enforcement
    - Fallback to alternative agents
    - Real-time status monitoring
    - Cost tracking
    """

    def __init__(self):
        self.agents: Dict[AgentType, CodingAgent] = {}
        self._lock = threading.Lock()
        self._default_agent: Optional[AgentType] = None
        self._force_agent: Optional[AgentType] = None  # Override for all tasks

        # Initialize default agents
        self._init_default_agents()

    def _init_default_agents(self):
        """Initialize default agent configurations."""
        # GLM Haiku - fast and cheap
        self.register_agent(GLMAgent(
            AgentConfig(
                agent_type=AgentType.GLM_HAIKU,
                enabled=True,
                max_requests_per_hour=200,
                max_tokens_per_hour=1000000,
                min_complexity=TaskComplexity.TRIVIAL,
                max_complexity=TaskComplexity.SIMPLE,
                priority=10  # Highest priority for simple tasks
            ),
            model="haiku"
        ))

        # GLM Sonnet - balanced
        self.register_agent(GLMAgent(
            AgentConfig(
                agent_type=AgentType.GLM_SONNET,
                enabled=True,
                max_requests_per_hour=100,
                max_tokens_per_hour=500000,
                min_complexity=TaskComplexity.SIMPLE,
                max_complexity=TaskComplexity.COMPLEX,
                priority=5
            ),
            model="sonnet"
        ))

        # Claude Code - full power
        self.register_agent(ClaudeCodeAgent(
            AgentConfig(
                agent_type=AgentType.CLAUDE_CODE,
                enabled=True,
                max_requests_per_hour=20,  # Conservative default
                max_tokens_per_hour=100000,
                min_complexity=TaskComplexity.MODERATE,
                max_complexity=TaskComplexity.EXPERT,
                priority=1  # Lower priority, use for complex tasks
            )
        ))

        # Mock agent for testing
        self.register_agent(MockAgent(
            AgentConfig(
                agent_type=AgentType.MOCK,
                enabled=False,  # Disabled by default
                max_requests_per_hour=1000,
                max_tokens_per_hour=10000000,
                min_complexity=TaskComplexity.TRIVIAL,
                max_complexity=TaskComplexity.EXPERT,
                priority=0
            )
        ))

        self._default_agent = AgentType.GLM_SONNET

    def register_agent(self, agent: CodingAgent):
        """Register a coding agent."""
        with self._lock:
            self.agents[agent.agent_type] = agent
            logger.info(f"Registered agent: {agent.agent_type.value}")

    def set_default_agent(self, agent_type: AgentType):
        """Set the default agent for tasks."""
        if agent_type in self.agents:
            self._default_agent = agent_type

    def set_force_agent(self, agent_type: Optional[AgentType]):
        """Force all tasks to use a specific agent (override)."""
        self._force_agent = agent_type
        if agent_type:
            logger.info(f"Forcing all tasks to use: {agent_type.value}")
        else:
            logger.info("Cleared forced agent override")

    def get_agent(self, agent_type: AgentType) -> Optional[CodingAgent]:
        """Get a specific agent."""
        return self.agents.get(agent_type)

    def update_agent_config(self, agent_type: AgentType,
                           updates: Dict[str, Any]) -> bool:
        """Update an agent's configuration."""
        agent = self.agents.get(agent_type)
        if not agent:
            return False

        with self._lock:
            if "enabled" in updates:
                agent.config.enabled = updates["enabled"]
            if "max_requests_per_hour" in updates:
                agent.config.max_requests_per_hour = updates["max_requests_per_hour"]
            if "max_tokens_per_hour" in updates:
                agent.config.max_tokens_per_hour = updates["max_tokens_per_hour"]
            if "min_complexity" in updates:
                agent.config.min_complexity = TaskComplexity[updates["min_complexity"]]
            if "max_complexity" in updates:
                agent.config.max_complexity = TaskComplexity[updates["max_complexity"]]
            if "priority" in updates:
                agent.config.priority = updates["priority"]

        return True

    def select_agent(self, complexity: TaskComplexity) -> Optional[CodingAgent]:
        """Select the best agent for the given complexity."""
        # Check for forced override
        if self._force_agent:
            agent = self.agents.get(self._force_agent)
            if agent and agent.config.enabled:
                within_limits, _ = agent.is_within_limits()
                if within_limits:
                    return agent

        # Find all capable agents
        candidates = []
        for agent in self.agents.values():
            if agent.can_handle(complexity):
                within_limits, _ = agent.is_within_limits()
                if within_limits:
                    candidates.append(agent)

        if not candidates:
            return None

        # Sort by priority (higher = preferred)
        candidates.sort(key=lambda a: a.config.priority, reverse=True)
        return candidates[0]

    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute a task with automatic agent selection."""
        # Select best agent
        agent = self.select_agent(request.complexity)

        if not agent:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error="No suitable agent available (all at capacity or disabled)"
            )

        logger.info(f"Executing task {request.task_id} with {agent.agent_type.value}")
        return agent.execute(request)

    def execute_with_agent(self, agent_type: AgentType,
                          request: AgentRequest) -> AgentResponse:
        """Execute a task with a specific agent."""
        agent = self.agents.get(agent_type)

        if not agent:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=f"Agent {agent_type.value} not found"
            )

        if not agent.config.enabled:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=f"Agent {agent_type.value} is disabled"
            )

        within_limits, reason = agent.is_within_limits()
        if not within_limits:
            return AgentResponse(
                task_id=request.task_id,
                success=False,
                output="",
                error=f"Agent {agent_type.value}: {reason}"
            )

        return agent.execute(request)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        # Reset hourly counters if needed
        for agent in self.agents.values():
            agent.stats.reset_hourly()

        return {
            "agents": {
                agent_type.value: agent.get_status()
                for agent_type, agent in self.agents.items()
            },
            "default_agent": self._default_agent.value if self._default_agent else None,
            "force_agent": self._force_agent.value if self._force_agent else None,
            "total_cost": sum(a.stats.total_cost for a in self.agents.values()),
            "total_requests": sum(a.stats.total_requests for a in self.agents.values()),
            # Frontend-expected fields
            "requests_this_hour": sum(a.stats.requests_this_hour for a in self.agents.values()),
            "max_requests_hour": min(a.config.max_requests_per_hour for a in self.agents.values()) if self.agents else 100,
            "max_cost_daily": 10.0  # Configurable daily cost limit
        }

    def reset_usage(self):
        """Reset all usage statistics for all agents."""
        for agent in self.agents.values():
            agent.stats = AgentUsageStats()
        logger.info("All agent usage statistics reset")

    def get_recommendations(self, task_type: str,
                           estimated_complexity: str) -> Dict[str, Any]:
        """Get agent recommendations for a task."""
        complexity = TaskComplexity[estimated_complexity.upper()]

        recommendations = []
        for agent in self.agents.values():
            if not agent.config.enabled:
                continue

            can_handle = agent.can_handle(complexity)
            within_limits, limit_reason = agent.is_within_limits()

            recommendations.append({
                "agent_type": agent.agent_type.value,
                "can_handle": can_handle,
                "within_limits": within_limits,
                "limit_reason": limit_reason,
                "priority": agent.config.priority,
                "recommended": can_handle and within_limits,
                "estimated_cost": agent.config.cost_per_1k_input_tokens + agent.config.cost_per_1k_output_tokens
            })

        # Sort by recommendation status and priority
        recommendations.sort(key=lambda r: (r["recommended"], r["priority"]), reverse=True)

        return {
            "task_type": task_type,
            "complexity": complexity.name,
            "recommendations": recommendations,
            "best_choice": recommendations[0]["agent_type"] if recommendations and recommendations[0]["recommended"] else None
        }


# Global instance
_agent_manager: Optional[CodingAgentManager] = None
_manager_lock = threading.Lock()


def get_agent_manager() -> CodingAgentManager:
    """Get the global agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        with _manager_lock:
            if _agent_manager is None:
                _agent_manager = CodingAgentManager()
    return _agent_manager


def init_agent_manager() -> CodingAgentManager:
    """Initialize and return the agent manager."""
    global _agent_manager
    with _manager_lock:
        _agent_manager = CodingAgentManager()
    return _agent_manager

#!/usr/bin/env python3
"""
Task Router for PM-1000

Multi-tier model routing system that intelligently routes tasks
to the most appropriate AI model based on:
- Task complexity and requirements
- Cost constraints and budgets
- Model availability and performance
- Current autonomy levels

Model Tiers:
1. LOCAL   - Local LLM (MLX/Ollama) for simple, fast tasks
2. FAST    - Small API models (Haiku) for moderate tasks
3. CAPABLE - Standard API models (Sonnet) for complex tasks
4. EXPERT  - Premium API models (Opus) for critical/advanced tasks

Routing Strategy:
- Analyze task complexity using heuristics and history
- Match to minimum capable tier that can handle it
- Fall back to higher tiers if needed
- Track outcomes for continuous optimization
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

from .config_manager import (
    AutonomyConfigManager,
    get_config_manager,
)
from .system_state_manager import (
    SystemStateManager,
    get_state_manager,
)

logger = get_logger("pm1000.autonomy.router")


class ModelTier(Enum):
    """Model tiers ordered by capability (and cost)."""
    LOCAL = 0      # Local models (free, fast, limited capability)
    FAST = 1       # Fast API models (cheap, moderate capability)
    CAPABLE = 2    # Standard API models (moderate cost, good capability)
    EXPERT = 3     # Expert API models (expensive, maximum capability)


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = 0    # Simple lookups, formatting
    SIMPLE = 1     # Single-step tasks, basic generation
    MODERATE = 2   # Multi-step tasks, analysis needed
    COMPLEX = 3    # Extensive reasoning, multi-file operations
    EXPERT = 4     # Novel problems, architecture decisions


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    tier: ModelTier
    name: str
    provider: str  # "local", "anthropic", "openai", etc.
    model_id: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    context_window: int = 128000
    avg_latency_ms: float = 1000.0
    success_rate: float = 0.95
    available: bool = True
    supports_tools: bool = True
    supports_vision: bool = False

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )


@dataclass
class TaskAnalysis:
    """Analysis of a task for routing purposes."""
    task_id: str
    task_type: str
    description: str

    # Complexity indicators
    estimated_complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_input_tokens: int = 500
    estimated_output_tokens: int = 1000
    requires_tools: bool = False
    requires_vision: bool = False
    requires_context_window: int = 4000

    # Routing preferences
    minimum_tier: ModelTier = ModelTier.LOCAL
    preferred_tier: Optional[ModelTier] = None
    maximum_cost: Optional[float] = None
    maximum_latency_ms: Optional[float] = None

    # Context
    priority: str = "medium"
    is_user_facing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "estimated_complexity": self.estimated_complexity.name,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "requires_tools": self.requires_tools,
            "requires_vision": self.requires_vision,
            "requires_context_window": self.requires_context_window,
            "minimum_tier": self.minimum_tier.name,
            "preferred_tier": self.preferred_tier.name if self.preferred_tier else None,
            "maximum_cost": self.maximum_cost,
            "maximum_latency_ms": self.maximum_latency_ms,
            "priority": self.priority,
            "is_user_facing": self.is_user_facing,
        }


@dataclass
class RoutingDecision:
    """A routing decision for a task."""
    task_id: str
    selected_tier: ModelTier
    selected_model: ModelConfig
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float  # 0-1 confidence in the routing decision
    alternatives: List[ModelTier] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "selected_tier": self.selected_tier.name,
            "selected_model": self.selected_model.name,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "confidence": self.confidence,
            "alternatives": [t.name for t in self.alternatives],
        }


@dataclass
class RoutingOutcome:
    """Outcome of a routed task for learning."""
    task_id: str
    tier: ModelTier
    model_name: str
    success: bool
    actual_latency_ms: float
    actual_cost: float
    input_tokens: int
    output_tokens: int
    retries: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier.name,
            "model_name": self.model_name,
            "success": self.success,
            "actual_latency_ms": self.actual_latency_ms,
            "actual_cost": self.actual_cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RouterMetrics:
    """Metrics for the task router."""
    total_routed: int = 0
    by_tier: Dict[str, int] = field(default_factory=lambda: {t.name: 0 for t in ModelTier})
    successes_by_tier: Dict[str, int] = field(default_factory=lambda: {t.name: 0 for t in ModelTier})
    failures_by_tier: Dict[str, int] = field(default_factory=lambda: {t.name: 0 for t in ModelTier})
    total_cost: float = 0.0
    cost_by_tier: Dict[str, float] = field(default_factory=lambda: {t.name: 0.0 for t in ModelTier})
    total_latency_ms: float = 0.0
    escalations: int = 0
    fallbacks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_routed": self.total_routed,
            "by_tier": self.by_tier,
            "successes_by_tier": self.successes_by_tier,
            "failures_by_tier": self.failures_by_tier,
            "total_cost": self.total_cost,
            "cost_by_tier": self.cost_by_tier,
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_routed),
            "escalations": self.escalations,
            "fallbacks": self.fallbacks,
            "success_rate": sum(self.successes_by_tier.values()) / max(1, self.total_routed),
        }


class ComplexityAnalyzer:
    """Analyzes task complexity using heuristics and history."""

    # Keyword indicators for complexity
    TRIVIAL_INDICATORS = [
        "format", "list", "count", "simple", "quick",
        "lookup", "get", "check", "status"
    ]

    SIMPLE_INDICATORS = [
        "summarize", "explain", "describe", "convert",
        "extract", "parse", "validate"
    ]

    MODERATE_INDICATORS = [
        "analyze", "compare", "review", "fix", "update",
        "implement simple", "add basic", "modify"
    ]

    COMPLEX_INDICATORS = [
        "refactor", "optimize", "design", "integrate",
        "implement", "build", "create", "architect"
    ]

    EXPERT_INDICATORS = [
        "architecture", "security audit", "performance",
        "novel", "research", "evaluate", "strategic"
    ]

    # Task type complexity baselines
    TYPE_COMPLEXITY = {
        "code_format": TaskComplexity.TRIVIAL,
        "doc_update": TaskComplexity.SIMPLE,
        "bug_fix": TaskComplexity.MODERATE,
        "test_add": TaskComplexity.MODERATE,
        "feature_implement": TaskComplexity.COMPLEX,
        "refactor": TaskComplexity.COMPLEX,
        "architecture": TaskComplexity.EXPERT,
        "security_review": TaskComplexity.EXPERT,
    }

    def __init__(self):
        self._history: Dict[str, List[RoutingOutcome]] = {}
        self._type_performance: Dict[str, Dict[str, float]] = {}

    def analyze(self, task_id: str, task_type: str, description: str, context: Dict[str, Any] = None) -> TaskAnalysis:
        """Analyze a task to determine complexity and routing requirements."""
        context = context or {}

        # Start with type-based baseline
        base_complexity = self.TYPE_COMPLEXITY.get(task_type, TaskComplexity.MODERATE)

        # Adjust based on description keywords
        desc_lower = description.lower()

        complexity_score = base_complexity.value

        # Check indicators
        if any(ind in desc_lower for ind in self.TRIVIAL_INDICATORS):
            complexity_score = min(complexity_score, TaskComplexity.TRIVIAL.value)
        elif any(ind in desc_lower for ind in self.SIMPLE_INDICATORS):
            complexity_score = max(min(complexity_score, TaskComplexity.SIMPLE.value), TaskComplexity.SIMPLE.value)
        elif any(ind in desc_lower for ind in self.COMPLEX_INDICATORS):
            complexity_score = max(complexity_score, TaskComplexity.COMPLEX.value)
        elif any(ind in desc_lower for ind in self.EXPERT_INDICATORS):
            complexity_score = max(complexity_score, TaskComplexity.EXPERT.value)

        estimated_complexity = TaskComplexity(complexity_score)

        # Estimate tokens based on complexity
        token_estimates = {
            TaskComplexity.TRIVIAL: (200, 300),
            TaskComplexity.SIMPLE: (400, 600),
            TaskComplexity.MODERATE: (800, 1500),
            TaskComplexity.COMPLEX: (2000, 4000),
            TaskComplexity.EXPERT: (4000, 8000),
        }
        input_tokens, output_tokens = token_estimates.get(estimated_complexity, (500, 1000))

        # Check for special requirements
        requires_tools = any(word in desc_lower for word in ["file", "execute", "run", "edit", "create"])
        requires_vision = any(word in desc_lower for word in ["image", "screenshot", "visual", "ui"])

        # Context window requirements
        if "large" in desc_lower or "codebase" in desc_lower or "multi-file" in desc_lower:
            requires_context_window = 32000
        elif "file" in desc_lower:
            requires_context_window = 8000
        else:
            requires_context_window = 4000

        # Minimum tier based on complexity
        tier_mapping = {
            TaskComplexity.TRIVIAL: ModelTier.LOCAL,
            TaskComplexity.SIMPLE: ModelTier.LOCAL,
            TaskComplexity.MODERATE: ModelTier.FAST,
            TaskComplexity.COMPLEX: ModelTier.CAPABLE,
            TaskComplexity.EXPERT: ModelTier.EXPERT,
        }
        minimum_tier = tier_mapping.get(estimated_complexity, ModelTier.FAST)

        # Adjust minimum tier for special requirements
        if requires_vision:
            minimum_tier = max(minimum_tier, ModelTier.CAPABLE, key=lambda t: t.value)
        if requires_context_window > 16000:
            minimum_tier = max(minimum_tier, ModelTier.FAST, key=lambda t: t.value)

        # Check priority from context
        priority = context.get("priority", "medium")
        is_user_facing = context.get("user_facing", False)

        return TaskAnalysis(
            task_id=task_id,
            task_type=task_type,
            description=description,
            estimated_complexity=estimated_complexity,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_context_window=requires_context_window,
            minimum_tier=minimum_tier,
            priority=priority,
            is_user_facing=is_user_facing,
        )

    def record_outcome(self, task_type: str, outcome: RoutingOutcome):
        """Record outcome for learning."""
        if task_type not in self._history:
            self._history[task_type] = []
        self._history[task_type].append(outcome)

        # Keep last 100 outcomes per type
        self._history[task_type] = self._history[task_type][-100:]

        # Update performance stats
        self._update_type_performance(task_type)

    def _update_type_performance(self, task_type: str):
        """Update performance statistics for a task type."""
        outcomes = self._history.get(task_type, [])
        if not outcomes:
            return

        by_tier: Dict[str, List[RoutingOutcome]] = {}
        for outcome in outcomes:
            tier_name = outcome.tier.name
            if tier_name not in by_tier:
                by_tier[tier_name] = []
            by_tier[tier_name].append(outcome)

        self._type_performance[task_type] = {
            tier: sum(1 for o in outcomes if o.success) / len(outcomes)
            for tier, outcomes in by_tier.items()
        }

    def get_recommended_tier(self, task_type: str, minimum: ModelTier) -> Optional[ModelTier]:
        """Get recommended tier based on historical performance."""
        perf = self._type_performance.get(task_type, {})
        if not perf:
            return None

        # Find lowest tier with >90% success rate above minimum
        for tier in ModelTier:
            if tier.value < minimum.value:
                continue
            success_rate = perf.get(tier.name, 0)
            if success_rate >= 0.9:
                return tier

        return None


class TaskRouter:
    """
    Multi-tier model router for PM-1000.

    Routes tasks to the most appropriate model tier based on
    complexity analysis, cost constraints, and performance history.
    """

    # Default model configurations
    DEFAULT_MODELS = [
        ModelConfig(
            tier=ModelTier.LOCAL,
            name="local-llm",
            provider="local",
            model_id="mlx-community/Mistral-7B-Instruct",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_tokens=4096,
            context_window=8192,
            avg_latency_ms=500,
            success_rate=0.85,
            supports_tools=False,
            supports_vision=False,
        ),
        ModelConfig(
            tier=ModelTier.FAST,
            name="haiku",
            provider="anthropic",
            model_id="claude-3-5-haiku-20241022",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            max_tokens=8192,
            context_window=200000,
            avg_latency_ms=800,
            success_rate=0.92,
            supports_tools=True,
            supports_vision=True,
        ),
        ModelConfig(
            tier=ModelTier.CAPABLE,
            name="sonnet",
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            max_tokens=8192,
            context_window=200000,
            avg_latency_ms=1500,
            success_rate=0.96,
            supports_tools=True,
            supports_vision=True,
        ),
        ModelConfig(
            tier=ModelTier.EXPERT,
            name="opus",
            provider="anthropic",
            model_id="claude-opus-4-20250514",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_tokens=8192,
            context_window=200000,
            avg_latency_ms=3000,
            success_rate=0.98,
            supports_tools=True,
            supports_vision=True,
        ),
    ]

    def __init__(
        self,
        config_manager: Optional[AutonomyConfigManager] = None,
        state_manager: Optional[SystemStateManager] = None,
        models: Optional[List[ModelConfig]] = None,
    ):
        self._config = config_manager or get_config_manager()
        self._state = state_manager or get_state_manager()

        # Model configurations
        self._models: Dict[ModelTier, ModelConfig] = {}
        for model in (models or self.DEFAULT_MODELS):
            self._models[model.tier] = model

        # Analysis and metrics
        self._analyzer = ComplexityAnalyzer()
        self._metrics = RouterMetrics()
        self._lock = threading.RLock()

        # Recent outcomes for analysis
        self._recent_outcomes: List[RoutingOutcome] = []

        # Callbacks
        self._on_route: List[Callable[[RoutingDecision], None]] = []
        self._on_outcome: List[Callable[[RoutingOutcome], None]] = []

        logger.info("TaskRouter initialized with %d model tiers", len(self._models))

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_model_config(self, tier: ModelTier, config: ModelConfig):
        """Set or update model configuration for a tier."""
        with self._lock:
            self._models[tier] = config
            logger.info(f"Updated model config for {tier.name}: {config.name}")

    def set_model_available(self, tier: ModelTier, available: bool):
        """Set model availability."""
        with self._lock:
            if tier in self._models:
                self._models[tier].available = available
                logger.info(f"Model {tier.name} availability: {available}")

    def on_route(self, callback: Callable[[RoutingDecision], None]):
        """Register callback for routing decisions."""
        self._on_route.append(callback)

    def on_outcome(self, callback: Callable[[RoutingOutcome], None]):
        """Register callback for routing outcomes."""
        self._on_outcome.append(callback)

    # =========================================================================
    # Routing Logic
    # =========================================================================

    def analyze_task(self, task_id: str, task_type: str, description: str,
                     context: Dict[str, Any] = None) -> TaskAnalysis:
        """Analyze a task for routing."""
        return self._analyzer.analyze(task_id, task_type, description, context)

    def route(self, analysis: TaskAnalysis) -> RoutingDecision:
        """Route a task to the most appropriate model tier."""
        with self._lock:
            # Get current budget constraints
            budget = self._config.get_resource_budget()
            resource_state = self._state.resource_state

            remaining_budget = budget.daily_api_budget - resource_state.api_spend_today

            # Get autonomy level (affects tier selection)
            dials = self._config.get_autonomy_dials()
            autonomy_level = dials.execution.current_level

            # Start with minimum capable tier
            selected_tier = analysis.minimum_tier

            # Check if we have a historical recommendation
            recommended = self._analyzer.get_recommended_tier(
                analysis.task_type, analysis.minimum_tier
            )
            if recommended:
                selected_tier = recommended

            # Adjust for preferred tier
            if analysis.preferred_tier and analysis.preferred_tier.value >= selected_tier.value:
                selected_tier = analysis.preferred_tier

            # Apply autonomy constraints
            # At low autonomy, prefer cheaper models
            if autonomy_level < 0.3:
                max_tier = ModelTier.FAST
                if selected_tier.value > max_tier.value:
                    selected_tier = max_tier
            elif autonomy_level < 0.6:
                max_tier = ModelTier.CAPABLE
                if selected_tier.value > max_tier.value:
                    selected_tier = max_tier

            # Find a valid model
            selected_model = None
            alternatives = []
            reason = ""

            for tier in ModelTier:
                if tier.value < selected_tier.value:
                    continue

                model = self._models.get(tier)
                if not model or not model.available:
                    continue

                # Check requirements
                if analysis.requires_tools and not model.supports_tools:
                    continue
                if analysis.requires_vision and not model.supports_vision:
                    continue
                if analysis.requires_context_window > model.context_window:
                    continue

                # Check cost constraints
                estimated_cost = model.estimated_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens
                )

                if analysis.maximum_cost and estimated_cost > analysis.maximum_cost:
                    continue

                if estimated_cost > remaining_budget:
                    continue

                # Check latency constraints
                if analysis.maximum_latency_ms and model.avg_latency_ms > analysis.maximum_latency_ms:
                    alternatives.append(tier)
                    continue

                # Found valid model
                if selected_model is None:
                    selected_model = model
                    selected_tier = tier
                    reason = f"Best match for {analysis.estimated_complexity.name} task"
                else:
                    alternatives.append(tier)

            # Fallback if no model found
            if selected_model is None:
                # Try any available model
                for tier in ModelTier:
                    model = self._models.get(tier)
                    if model and model.available:
                        selected_model = model
                        selected_tier = tier
                        reason = "Fallback - no optimal model available"
                        self._metrics.fallbacks += 1
                        break

            if selected_model is None:
                raise RuntimeError("No available models for routing")

            # Calculate confidence
            confidence = 0.8
            if recommended and recommended == selected_tier:
                confidence = 0.95
            if selected_tier.value > analysis.minimum_tier.value:
                confidence *= 0.9  # Slight penalty for escalation
                self._metrics.escalations += 1

            # Create decision
            decision = RoutingDecision(
                task_id=analysis.task_id,
                selected_tier=selected_tier,
                selected_model=selected_model,
                reason=reason,
                estimated_cost=selected_model.estimated_cost(
                    analysis.estimated_input_tokens,
                    analysis.estimated_output_tokens
                ),
                estimated_latency_ms=selected_model.avg_latency_ms,
                confidence=confidence,
                alternatives=alternatives,
            )

            # Update metrics
            self._metrics.total_routed += 1
            self._metrics.by_tier[selected_tier.name] += 1

            # Notify callbacks
            for callback in self._on_route:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Route callback error: {e}")

            logger.info(
                f"Routed task {analysis.task_id} to {selected_tier.name} "
                f"({selected_model.name}): {reason}"
            )

            return decision

    def route_task(self, task_id: str, task_type: str, description: str,
                   context: Dict[str, Any] = None) -> RoutingDecision:
        """Analyze and route a task in one call."""
        analysis = self.analyze_task(task_id, task_type, description, context)
        return self.route(analysis)

    # =========================================================================
    # Outcome Recording
    # =========================================================================

    def record_outcome(
        self,
        task_id: str,
        tier: ModelTier,
        success: bool,
        latency_ms: float,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        retries: int = 0,
        error: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        """Record the outcome of a routed task."""
        with self._lock:
            model = self._models.get(tier)
            model_name = model.name if model else tier.name

            outcome = RoutingOutcome(
                task_id=task_id,
                tier=tier,
                model_name=model_name,
                success=success,
                actual_latency_ms=latency_ms,
                actual_cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                retries=retries,
                error=error,
            )

            # Update metrics
            if success:
                self._metrics.successes_by_tier[tier.name] += 1
            else:
                self._metrics.failures_by_tier[tier.name] += 1

            self._metrics.total_cost += cost
            self._metrics.cost_by_tier[tier.name] += cost
            self._metrics.total_latency_ms += latency_ms

            # Store outcome
            self._recent_outcomes.append(outcome)
            self._recent_outcomes = self._recent_outcomes[-500:]  # Keep last 500

            # Update analyzer for learning
            if task_type:
                self._analyzer.record_outcome(task_type, outcome)

            # Update model performance stats
            if model:
                # Exponential moving average of success rate
                alpha = 0.1
                current_success = 1.0 if success else 0.0
                model.success_rate = alpha * current_success + (1 - alpha) * model.success_rate

                # Update latency estimate
                model.avg_latency_ms = alpha * latency_ms + (1 - alpha) * model.avg_latency_ms

            # Notify callbacks
            for callback in self._on_outcome:
                try:
                    callback(outcome)
                except Exception as e:
                    logger.error(f"Outcome callback error: {e}")

            logger.debug(
                f"Recorded outcome for {task_id}: "
                f"{'success' if success else 'failure'} on {tier.name}"
            )

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    def get_model_config(self, tier: ModelTier) -> Optional[ModelConfig]:
        """Get model configuration for a tier."""
        return self._models.get(tier)

    def get_available_tiers(self) -> List[ModelTier]:
        """Get list of available model tiers."""
        return [
            tier for tier, model in self._models.items()
            if model.available
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics."""
        with self._lock:
            return self._metrics.to_dict()

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models."""
        with self._lock:
            return {
                tier.name: {
                    "name": model.name,
                    "available": model.available,
                    "success_rate": model.success_rate,
                    "avg_latency_ms": model.avg_latency_ms,
                    "cost_per_1k_input": model.cost_per_1k_input,
                    "cost_per_1k_output": model.cost_per_1k_output,
                    "total_routed": self._metrics.by_tier.get(tier.name, 0),
                    "total_cost": self._metrics.cost_by_tier.get(tier.name, 0),
                }
                for tier, model in self._models.items()
            }

    def get_recent_outcomes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent routing outcomes."""
        with self._lock:
            return [o.to_dict() for o in self._recent_outcomes[-limit:]]

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by tier."""
        with self._lock:
            total = self._metrics.total_cost
            return {
                "total_cost": total,
                "by_tier": self._metrics.cost_by_tier,
                "percentage_by_tier": {
                    tier: (cost / total * 100) if total > 0 else 0
                    for tier, cost in self._metrics.cost_by_tier.items()
                },
            }


# Global instance
_task_router: Optional[TaskRouter] = None
_router_lock = threading.Lock()


def get_task_router() -> TaskRouter:
    """Get the global task router instance."""
    global _task_router
    with _router_lock:
        if _task_router is None:
            _task_router = TaskRouter()
        return _task_router


def init_task_router(**kwargs) -> TaskRouter:
    """Initialize the global task router."""
    global _task_router
    with _router_lock:
        _task_router = TaskRouter(**kwargs)
        return _task_router

"""
MindForge DNA - Cortex Layer

The Cortex represents specialized cognitive functions powered by small,
fine-tuned models. Each neuron is an expert in one domain.

Neurons:
    - ThinkCortex: Thought generation and reasoning (Qwen2.5-1.5B, r=16)
    - TaskCortex: Task extraction and prioritization (Qwen2.5-0.5B, r=8)
    - ActionCortex: Tool selection and call formatting (Qwen2.5-0.5B, r=8)
    - ReflectCortex: Reflection and journaling (Qwen2.5-0.5B, r=8)
    - DebugCortex: Error analysis and fix suggestions (Qwen2.5-0.5B, r=16)
    - MemoryCortex: Memory retrieval and importance (SmolLM2-1.7B, r=16)

Architecture:
    Each neuron:
    - Uses MLX for Apple Silicon efficiency
    - Has LoRA adapters for domain specialization
    - Estimates confidence for EGO fallback
    - Records experiences for continual learning

Usage:
    from mindforge_dna.cortex import ThinkCortex, TaskCortex

    # Create neurons
    think = ThinkCortex()
    task = TaskCortex()

    # Use neurons
    thought = think.think(context="User asked about weather", needs={})
    if not thought.should_fallback:
        tasks = task.extract_tasks(think.extract_thought_text(thought))

    # Record outcomes for training
    think.record_outcome(thought, actual_outcome="Good thought", score=0.9)
"""

# Base classes
from .base import (
    CortexNeuron,
    Experience,
    NeuronConfig,
    NeuronDomain,
    NeuronOutput,
)

# Specialized neurons
from .action import ActionCortex, ActionType, create_action_cortex
from .debug import DebugCortex, ErrorSeverity, create_debug_cortex
from .memory import MemoryCortex, create_memory_cortex
from .reflect import MoodType, ReflectCortex, create_reflect_cortex
from .task import TaskCortex, create_task_cortex
from .think import ThinkCortex, create_think_cortex

__all__ = [
    # Base classes
    "CortexNeuron",
    "NeuronConfig",
    "NeuronOutput",
    "NeuronDomain",
    "Experience",
    # Think neuron
    "ThinkCortex",
    "create_think_cortex",
    # Task neuron
    "TaskCortex",
    "create_task_cortex",
    # Action neuron
    "ActionCortex",
    "ActionType",
    "create_action_cortex",
    # Reflect neuron
    "ReflectCortex",
    "MoodType",
    "create_reflect_cortex",
    # Debug neuron
    "DebugCortex",
    "ErrorSeverity",
    "create_debug_cortex",
    # Memory neuron
    "MemoryCortex",
    "create_memory_cortex",
]


def create_cortex_suite(
    base_models: dict = None,
    adapter_paths: dict = None
) -> dict:
    """Create a full suite of cortex neurons.

    Args:
        base_models: Optional dict mapping neuron names to base models
        adapter_paths: Optional dict mapping neuron names to adapter paths

    Returns:
        Dictionary of neuron name -> neuron instance

    Example:
        cortex = create_cortex_suite(
            adapter_paths={
                "think": "/path/to/think_adapter",
                "task": "/path/to/task_adapter",
            }
        )
        thought = cortex["think"].think("Context here", needs={})
    """
    base_models = base_models or {}
    adapter_paths = adapter_paths or {}

    suite = {
        "think": create_think_cortex(
            base_model=base_models.get("think", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
            adapter_path=adapter_paths.get("think")
        ),
        "task": create_task_cortex(
            base_model=base_models.get("task", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
            adapter_path=adapter_paths.get("task")
        ),
        "action": create_action_cortex(
            base_model=base_models.get("action", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
            adapter_path=adapter_paths.get("action")
        ),
        "reflect": create_reflect_cortex(
            base_model=base_models.get("reflect", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
            adapter_path=adapter_paths.get("reflect")
        ),
        "debug": create_debug_cortex(
            base_model=base_models.get("debug", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
            adapter_path=adapter_paths.get("debug")
        ),
        "memory": create_memory_cortex(
            base_model=base_models.get("memory", "mlx-community/SmolLM2-1.7B-Instruct-4bit"),
            adapter_path=adapter_paths.get("memory")
        ),
    }

    return suite


def get_cortex_info() -> dict:
    """Get information about all available cortex neurons.

    Returns:
        Dictionary with neuron information
    """
    return {
        "think": {
            "domain": "thinking",
            "model": "Qwen2.5-1.5B",
            "lora_rank": 16,
            "purpose": "Thought generation and reasoning",
            "input": "context, needs, memories, recent_actions",
            "output": "structured thought with reasoning"
        },
        "task": {
            "domain": "task",
            "model": "Qwen2.5-0.5B",
            "lora_rank": 8,
            "purpose": "Task extraction and prioritization",
            "input": "thought, pending_tasks, context",
            "output": "new tasks and rankings"
        },
        "action": {
            "domain": "action",
            "model": "Qwen2.5-0.5B",
            "lora_rank": 8,
            "purpose": "Tool selection and call formatting",
            "input": "task, available_tools, context",
            "output": "tool call or action decision"
        },
        "reflect": {
            "domain": "reflection",
            "model": "Qwen2.5-0.5B",
            "lora_rank": 8,
            "purpose": "Reflection and journaling",
            "input": "action, result, task, previous_mood",
            "output": "reflection with lessons and mood"
        },
        "debug": {
            "domain": "debug",
            "model": "Qwen2.5-0.5B",
            "lora_rank": 16,
            "purpose": "Error analysis and fix suggestions",
            "input": "error, task, previous_attempts, context",
            "output": "root cause and fix suggestion"
        },
        "memory": {
            "domain": "memory",
            "model": "SmolLM2-1.7B",
            "lora_rank": 16,
            "purpose": "Memory retrieval and importance scoring",
            "input": "query, memories, context",
            "output": "relevant memories with scores"
        },
    }

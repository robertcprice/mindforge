"""
Conch Training Module

Shared LoRA training infrastructure for consciousness and KVRM grounding.
Supports continuous learning from consciousness cycles and grounding feedback.

New reward-based learning system (v2):
- Tool Formats: Structured response format specification
- Reward Calculator: Computes intrinsic rewards for actions
- Experience Buffer: Stores experiences for training
- Data Generator: Creates synthetic training data
- Trainer: LoRA fine-tuning pipeline
- Intrinsic Motivation: Self-determination based drives
"""

# Original training infrastructure
from conch.training.data import (
    TrainingExample,
    TrainingDataset,
    ConsciousnessDataGenerator,
    GroundingDataGenerator,
)
from conch.training.lora import (
    LoRAConfig,
    LoRATrainer,
    LoRAAdapter,
    load_adapter,
    save_adapter,
)
from conch.training.pipeline import (
    TrainingPipeline,
    TrainingCallback,
    EvaluationCallback,
)

# New reward-based learning system
from conch.training.tool_formats import (
    ActionType,
    ToolSpec,
    TOOL_SPECS,
    ParsedAction,
    parse_action,
    get_format_instructions,
    validate_and_score,
)

from conch.training.reward_calculator import (
    RewardType,
    RewardBreakdown,
    RewardCalculator,
    DEFAULT_WEIGHTS,
)

from conch.training.experience_buffer import (
    Experience,
    ExperienceBuffer,
)

from conch.training.intrinsic_motivation import (
    MotivationType,
    MotivationState,
    IntrinsicMotivationEngine,
)

from conch.training.trainer import (
    TrainingConfig,
    ConchTrainer,
    incremental_train,
)

__all__ = [
    # Original Data
    "TrainingExample",
    "TrainingDataset",
    "ConsciousnessDataGenerator",
    "GroundingDataGenerator",
    # Original LoRA
    "LoRAConfig",
    "LoRATrainer",
    "LoRAAdapter",
    "load_adapter",
    "save_adapter",
    # Original Pipeline
    "TrainingPipeline",
    "TrainingCallback",
    "EvaluationCallback",

    # NEW: Tool formats
    "ActionType",
    "ToolSpec",
    "TOOL_SPECS",
    "ParsedAction",
    "parse_action",
    "get_format_instructions",
    "validate_and_score",

    # NEW: Rewards
    "RewardType",
    "RewardBreakdown",
    "RewardCalculator",
    "DEFAULT_WEIGHTS",

    # NEW: Experiences
    "Experience",
    "ExperienceBuffer",

    # NEW: Motivation
    "MotivationType",
    "MotivationState",
    "IntrinsicMotivationEngine",

    # NEW: Training v2
    "TrainingConfig",
    "ConchTrainer",
    "incremental_train",
]

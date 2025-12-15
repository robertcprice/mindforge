"""
EGO Module - The Living Mind and Personality DNA Source

The EGO (Executing Generative Oracle) is the conscious personality layer
of the Conch DNA architecture. It serves multiple critical roles:

1. GENERATOR: Core personality-driven generation with Echo's identity
2. TEACHER: Creates high-quality distillation examples for neuron training
3. CORRECTOR: Analyzes failures and provides learning signals
4. TIMER: Manages adaptive wake cycles for consciousness
5. AUDITOR: Quality-checks neuron outputs for alignment

The EGO uses MLX-optimized inference for efficient local execution.
"""

from .model import (
    EgoConfig,
    EgoModel,
    TimingDecision,
    PERSONALITY_PROMPT,
    MLX_AVAILABLE,
)

__all__ = [
    "EgoModel",
    "EgoConfig",
    "TimingDecision",
    "PERSONALITY_PROMPT",
    "MLX_AVAILABLE",
]
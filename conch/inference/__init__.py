"""
Conch Inference System

Provides LLM inference with multiple backend support:
- MLX: Primary backend for Apple Silicon (best performance)
- llama.cpp: Alternative high-performance backend
- Transformers: PyTorch-based fallback

The system automatically selects the best available backend.
"""

from conch.inference.base import InferenceBackend, InferenceConfig
from conch.inference.model_manager import ModelManager, get_inference_function

__all__ = ["InferenceBackend", "InferenceConfig", "ModelManager", "get_inference_function"]

"""
MindForge Inference Base Classes

Defines the interface for inference backends.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available inference backends."""
    MLX = "mlx"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Model settings
    model_path: str = ""
    adapter_path: str = ""  # For LoRA adapters

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

    # System prompt
    system_prompt: str = ""

    # Backend-specific
    context_length: int = 4096
    batch_size: int = 1
    num_threads: int = 4  # For CPU inference


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    tokens_generated: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "complete"  # complete, length, stop_token


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, config: InferenceConfig):
        """Initialize backend with configuration."""
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Sequences that stop generation

        Returns:
            GenerationResult with generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Sequences that stop generation

        Yields:
            Generated tokens one at a time
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Get the backend type."""
        pass

    def format_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages as a chat prompt.

        Args:
            messages: List of {role, content} dicts
            add_generation_prompt: Whether to add assistant prompt

        Returns:
            Formatted prompt string
        """
        # Default Qwen-style format
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

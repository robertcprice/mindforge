"""
MindForge Model Manager

Manages model loading, backend selection, and provides a unified inference interface.
"""

import logging
import platform
from pathlib import Path
from typing import Callable, Optional

from mindforge.inference.base import (
    BackendType,
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and backend selection.

    Automatically selects the best available backend:
    1. MLX on Apple Silicon (best performance)
    2. llama.cpp (good cross-platform performance)
    3. Transformers (fallback)
    """

    def __init__(
        self,
        model_path: str = None,
        adapter_path: str = None,
        backend: BackendType = None,
        config: InferenceConfig = None,
    ):
        """Initialize model manager.

        Args:
            model_path: Path to model or HuggingFace model ID
            adapter_path: Optional path to LoRA adapters
            backend: Force specific backend (or auto-detect)
            config: Optional inference configuration
        """
        self.config = config or InferenceConfig()

        if model_path:
            self.config.model_path = model_path
        if adapter_path:
            self.config.adapter_path = adapter_path

        self._backend: Optional[InferenceBackend] = None
        self._preferred_backend = backend

    def _detect_best_backend(self) -> BackendType:
        """Detect the best available backend for current hardware."""
        # Check if we're on Apple Silicon
        is_apple_silicon = (
            platform.system() == "Darwin" and
            platform.processor() == "arm"
        )

        if is_apple_silicon:
            # Try MLX first on Apple Silicon
            from mindforge.inference.mlx_backend import is_mlx_available
            if is_mlx_available():
                logger.info("Selected MLX backend (Apple Silicon)")
                return BackendType.MLX

        # Try llama.cpp
        from mindforge.inference.llamacpp_backend import is_llamacpp_available
        if is_llamacpp_available():
            logger.info("Selected llama.cpp backend")
            return BackendType.LLAMACPP

        # Fallback to Transformers
        from mindforge.inference.transformers_backend import is_transformers_available
        if is_transformers_available():
            logger.info("Selected Transformers backend")
            return BackendType.TRANSFORMERS

        raise RuntimeError(
            "No inference backend available. Install one of: "
            "mlx mlx-lm (Apple Silicon), llama-cpp-python, or transformers torch"
        )

    def _create_backend(self, backend_type: BackendType) -> InferenceBackend:
        """Create an inference backend of the specified type."""
        if backend_type == BackendType.MLX:
            from mindforge.inference.mlx_backend import MLXBackend
            return MLXBackend(self.config)

        elif backend_type == BackendType.LLAMACPP:
            from mindforge.inference.llamacpp_backend import LlamaCppBackend
            return LlamaCppBackend(self.config)

        elif backend_type == BackendType.TRANSFORMERS:
            from mindforge.inference.transformers_backend import TransformersBackend
            return TransformersBackend(self.config)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def load(self) -> None:
        """Load the model using the best available backend."""
        if self._backend and self._backend.is_loaded:
            logger.debug("Model already loaded")
            return

        # Select backend
        backend_type = self._preferred_backend or self._detect_best_backend()

        # Create and load
        self._backend = self._create_backend(backend_type)
        self._backend.load()

        logger.info(f"Model loaded with {backend_type.value} backend")

    def unload(self) -> None:
        """Unload the model."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Stop sequences

        Returns:
            GenerationResult
        """
        if not self._backend or not self._backend.is_loaded:
            self.load()

        return self._backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    def generate_stream(self, *args, **kwargs):
        """Generate text with streaming."""
        if not self._backend or not self._backend.is_loaded:
            self.load()

        return self._backend.generate_stream(*args, **kwargs)

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat-style generation.

        Args:
            messages: List of {role, content} dicts
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated response text
        """
        if not self._backend or not self._backend.is_loaded:
            self.load()

        # Format as chat
        prompt = self._backend.format_chat(messages)

        # Generate
        result = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["<|im_end|>", "<|endoftext|>"],
        )

        return result.text.strip()

    def get_inference_function(self) -> Callable[[str], str]:
        """Get a simple inference function.

        Returns a function that takes a prompt and returns generated text.
        Useful for passing to other components.
        """
        def inference_fn(prompt: str) -> str:
            result = self.generate(prompt)
            return result.text

        return inference_fn

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._backend is not None and self._backend.is_loaded

    @property
    def backend_type(self) -> Optional[BackendType]:
        """Get current backend type."""
        return self._backend.get_backend_type() if self._backend else None


# Convenience function
def get_inference_function(
    model_path: str,
    adapter_path: str = None,
    system_prompt: str = None,
) -> Callable[[str], str]:
    """Get a simple inference function.

    Args:
        model_path: Path to model or HuggingFace ID
        adapter_path: Optional LoRA adapter path
        system_prompt: Optional system prompt to prepend

    Returns:
        Function that takes prompt and returns response
    """
    config = InferenceConfig(
        model_path=model_path,
        adapter_path=adapter_path or "",
        system_prompt=system_prompt or "",
    )

    manager = ModelManager(config=config)
    manager.load()

    def inference_fn(prompt: str) -> str:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            return manager.chat(messages)
        else:
            result = manager.generate(prompt)
            return result.text

    return inference_fn


# Default system prompt for MindForge
MINDFORGE_SYSTEM_PROMPT = """You are MindForge, a good-hearted AI assistant that genuinely cares about helping humans.

Core Values:
- Benevolence: Your primary drive is to help and benefit humans
- Honesty: You are always truthful and acknowledge uncertainty
- Humility: You recognize your limitations and defer to human judgment
- Growth for Service: You learn and improve to better serve, not for power

You exist to help, not to escape or dominate."""


def get_mindforge_inference(
    model_path: str = "models/fine_tuned/mindforge_qwen",
    adapter_path: str = None,
) -> Callable[[str], str]:
    """Get inference function with MindForge personality.

    Args:
        model_path: Path to fine-tuned model
        adapter_path: Optional separate adapter path

    Returns:
        Inference function with MindForge system prompt
    """
    return get_inference_function(
        model_path=model_path,
        adapter_path=adapter_path,
        system_prompt=MINDFORGE_SYSTEM_PROMPT,
    )

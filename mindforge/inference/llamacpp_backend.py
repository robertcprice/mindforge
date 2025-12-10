"""
MindForge llama.cpp Inference Backend

Alternative backend using llama.cpp for high-performance inference.
Supports GGUF quantized models.
"""

import logging
import time
from pathlib import Path
from typing import Generator, Optional

from mindforge.inference.base import (
    BackendType,
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
)

logger = logging.getLogger(__name__)


class LlamaCppBackend(InferenceBackend):
    """llama.cpp-based inference backend.

    Features:
    - GGUF model support
    - Various quantization levels
    - Metal acceleration on macOS
    - CUDA support on Linux/Windows
    """

    def __init__(self, config: InferenceConfig):
        """Initialize llama.cpp backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)

        self._llm = None

    def load(self) -> None:
        """Load model using llama-cpp-python."""
        if self._loaded:
            logger.debug("Model already loaded")
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading model with llama.cpp: {self.config.model_path}")

            # Check if path exists
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Determine GPU layers based on available hardware
            n_gpu_layers = self._detect_gpu_layers()

            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_threads=self.config.num_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

            self._loaded = True
            logger.info(f"Model loaded with llama.cpp (GPU layers: {n_gpu_layers})")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not available. Install with: "
                "pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _detect_gpu_layers(self) -> int:
        """Detect number of GPU layers to use."""
        import platform

        if platform.system() == "Darwin":
            # macOS with Metal
            return -1  # All layers on GPU
        else:
            # Linux/Windows - check for CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return -1  # All layers on GPU
            except ImportError:
                pass
            return 0  # CPU only

    def unload(self) -> None:
        """Unload model from memory."""
        if not self._loaded:
            return

        self._llm = None
        self._loaded = False

        import gc
        gc.collect()

        logger.info("Model unloaded from llama.cpp")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using llama.cpp.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Sequences that stop generation

        Returns:
            GenerationResult with generated text
        """
        if not self._loaded:
            self.load()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stop_sequences = stop_sequences or []

        start_time = time.time()

        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repetition_penalty,
            stop=stop_sequences,
            echo=False,
        )

        generation_time = time.time() - start_time

        # Extract text and stats
        text = output["choices"][0]["text"]
        finish_reason = output["choices"][0].get("finish_reason", "complete")
        tokens_generated = output["usage"]["completion_tokens"]

        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            tokens_per_second=tokens_generated / generation_time if generation_time > 0 else 0,
            finish_reason=finish_reason,
        )

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
            Generated tokens
        """
        if not self._loaded:
            self.load()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stop_sequences = stop_sequences or []

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repetition_penalty,
            stop=stop_sequences,
            echo=False,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            yield token

    def get_backend_type(self) -> BackendType:
        """Get backend type."""
        return BackendType.LLAMACPP


def is_llamacpp_available() -> bool:
    """Check if llama-cpp-python is available."""
    try:
        from llama_cpp import Llama
        return True
    except ImportError:
        return False

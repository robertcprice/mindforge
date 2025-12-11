"""
MindForge MLX Inference Backend

Primary inference backend for Apple Silicon, using MLX for optimal performance.
MLX provides native Metal acceleration and efficient memory usage.
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


class MLXBackend(InferenceBackend):
    """MLX-based inference backend for Apple Silicon.

    Features:
    - Native Metal acceleration
    - Efficient unified memory usage
    - LoRA adapter support
    - Streaming generation
    """

    def __init__(self, config: InferenceConfig):
        """Initialize MLX backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)

        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model using MLX-LM."""
        if self._loaded:
            logger.debug("Model already loaded")
            return

        try:
            from mlx_lm import load

            logger.info(f"Loading model with MLX: {self.config.model_path}")

            # Load with optional adapter
            load_kwargs = {}
            if self.config.adapter_path:
                load_kwargs["adapter_path"] = self.config.adapter_path
                logger.info(f"Loading LoRA adapter: {self.config.adapter_path}")

            self._model, self._tokenizer = load(
                self.config.model_path,
                **load_kwargs
            )

            self._loaded = True
            logger.info("Model loaded successfully with MLX")

        except ImportError:
            raise ImportError(
                "MLX not available. Install with: pip install mlx mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload(self) -> None:
        """Unload model from memory."""
        if not self._loaded:
            return

        # Clear references to allow garbage collection
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Force garbage collection
        import gc
        gc.collect()

        logger.info("Model unloaded from MLX")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using MLX.

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

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        start_time = time.time()

        # Create sampler with temperature and top_p
        sampler = make_sampler(
            temp=temperature,
            top_p=self.config.top_p,
        )

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        generation_time = time.time() - start_time

        # Handle stop sequences
        finish_reason = "complete"
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response.split(seq)[0]
                    finish_reason = "stop_token"
                    break

        # Estimate tokens (rough approximation)
        tokens_generated = len(response.split()) * 1.3  # Rough token estimate

        return GenerationResult(
            text=response,
            tokens_generated=int(tokens_generated),
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

        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or self.config.temperature

            # Create sampler with temperature and top_p
            sampler = make_sampler(
                temp=temperature,
                top_p=self.config.top_p,
            )

            buffer = ""

            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                # New API returns GenerationResponse with .text attribute
                token = response.text if hasattr(response, 'text') else str(response)
                buffer += token

                # Check stop sequences
                if stop_sequences:
                    should_stop = False
                    for seq in stop_sequences:
                        if seq in buffer:
                            # Yield up to stop sequence
                            parts = buffer.split(seq)
                            yield parts[0]
                            should_stop = True
                            break
                    if should_stop:
                        break

                yield token

        except ImportError:
            # Fallback to non-streaming
            logger.warning("Streaming not available, falling back to standard generation")
            result = self.generate(prompt, max_tokens, temperature, stop_sequences)
            yield result.text

    def get_backend_type(self) -> BackendType:
        """Get backend type."""
        return BackendType.MLX

    def format_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages using tokenizer's chat template if available."""
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        # Fallback to default format
        return super().format_chat(messages, add_generation_prompt)

    def get_memory_usage(self) -> dict:
        """Get MLX memory usage statistics."""
        try:
            import mlx.core as mx

            return {
                "peak_memory_gb": mx.metal.get_peak_memory() / (1024**3),
                "active_memory_gb": mx.metal.get_active_memory() / (1024**3),
                "cache_memory_gb": mx.metal.get_cache_memory() / (1024**3),
            }
        except Exception:
            return {}


def is_mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        import mlx_lm
        return True
    except ImportError:
        return False

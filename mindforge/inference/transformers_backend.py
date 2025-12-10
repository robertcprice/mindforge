"""
MindForge Transformers Inference Backend

Standard HuggingFace Transformers backend for GPU/CPU inference.
Supports CUDA, MPS (Apple), and CPU with optional quantization.
"""

import gc
import logging
import time
from typing import Generator, Optional

from mindforge.inference.base import (
    BackendType,
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
)

logger = logging.getLogger(__name__)


class TransformersBackend(InferenceBackend):
    """HuggingFace Transformers-based inference backend.

    Features:
    - CUDA/MPS/CPU support with automatic device detection
    - BitsAndBytes quantization (4-bit, 8-bit)
    - LoRA adapter loading via PEFT
    - Streaming generation support
    - Flash Attention 2 support (when available)
    """

    def __init__(self, config: InferenceConfig):
        """Initialize Transformers backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)

        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None

    def _detect_device(self) -> str:
        """Detect the best available device."""
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS device")
        else:
            device = "cpu"
            logger.info("Using CPU device")

        return device

    def _get_dtype(self):
        """Get the appropriate dtype for the device."""
        import torch

        if self._device == "cuda":
            # Use bfloat16 if available, else float16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        elif self._device == "mps":
            return torch.float16
        else:
            return torch.float32

    def load(self) -> None:
        """Load model using Transformers."""
        if self._loaded:
            logger.debug("Model already loaded")
            return

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            logger.info(f"Loading model with Transformers: {self.config.model_path}")

            # Detect device and dtype
            self._device = self._detect_device()
            self._dtype = self._get_dtype()

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )

            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self._dtype,
            }

            # Use quantization for CUDA if bitsandbytes available
            if self._device == "cuda":
                try:
                    import bitsandbytes
                    # 4-bit quantization for memory efficiency
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self._dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    logger.info("Using 4-bit quantization with bitsandbytes")
                except ImportError:
                    model_kwargs["device_map"] = "auto"
                    logger.info("bitsandbytes not available, loading without quantization")
            elif self._device == "mps":
                # MPS doesn't support device_map="auto"
                model_kwargs["device_map"] = {"": self._device}
            else:
                # CPU
                model_kwargs["device_map"] = "cpu"

            # Try to use Flash Attention 2 if available
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                pass

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs,
            )

            # Load LoRA adapter if specified
            if self.config.adapter_path:
                self._load_adapter()

            self._loaded = True
            logger.info("Model loaded successfully with Transformers")

        except ImportError as e:
            raise ImportError(
                f"Transformers dependencies not available: {e}. "
                "Install with: pip install transformers torch accelerate"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_adapter(self) -> None:
        """Load LoRA adapter using PEFT."""
        try:
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter: {self.config.adapter_path}")
            self._model = PeftModel.from_pretrained(
                self._model,
                self.config.adapter_path,
            )
            logger.info("LoRA adapter loaded successfully")
        except ImportError:
            logger.warning(
                "PEFT not available, cannot load LoRA adapter. "
                "Install with: pip install peft"
            )
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")

    def unload(self) -> None:
        """Unload model from memory."""
        if not self._loaded:
            return

        # Clear model and tokenizer
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded from Transformers")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using Transformers.

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

        import torch

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.context_length,
        )

        # Move to device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        start_time = time.time()

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=self.config.top_p if temperature > 0 else None,
                top_k=self.config.top_k if temperature > 0 else None,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generation_time = time.time() - start_time

        # Decode output (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Handle stop sequences
        finish_reason = "complete"
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response.split(seq)[0]
                    finish_reason = "stop_token"
                    break

        tokens_generated = len(generated_tokens)

        return GenerationResult(
            text=response,
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

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            import torch

            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature if temperature is not None else self.config.temperature

            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.context_length,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Create streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Generation kwargs
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else None,
                "top_p": self.config.top_p if temperature > 0 else None,
                "top_k": self.config.top_k if temperature > 0 else None,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "streamer": streamer,
            }

            # Run generation in separate thread
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens as they come
            buffer = ""
            for text in streamer:
                buffer += text

                # Check stop sequences
                if stop_sequences:
                    should_stop = False
                    for seq in stop_sequences:
                        if seq in buffer:
                            # Yield up to stop sequence
                            parts = buffer.split(seq)
                            if parts[0]:
                                yield parts[0]
                            should_stop = True
                            break
                    if should_stop:
                        break

                yield text

            thread.join()

        except ImportError:
            # Fallback to non-streaming
            logger.warning("Streaming not available, falling back to standard generation")
            result = self.generate(prompt, max_tokens, temperature, stop_sequences)
            yield result.text

    def get_backend_type(self) -> BackendType:
        """Get backend type."""
        return BackendType.TRANSFORMERS

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
        """Get GPU/device memory usage statistics."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have detailed memory stats
                return {"device": "mps", "stats": "not available"}
            else:
                return {"device": "cpu", "stats": "not tracked"}
        except Exception:
            return {}


def is_transformers_available() -> bool:
    """Check if Transformers is available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False

"""
MindForge DNA - Cortex Layer: Base Neuron

The Cortex represents specialized cognitive functions powered by small,
fine-tuned models. Each neuron is an expert in one domain, using LoRA adapters
for efficient specialization.

Key principles:
- Small models (<2B parameters) for speed
- LoRA adapters (r=8-16) for specialization
- Confidence estimation for EGO fallback
- Experience recording for continual learning
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NeuronDomain(Enum):
    """Cognitive domains handled by specialized neurons."""

    THINKING = "thinking"         # Reasoning and thought generation
    TASK = "task"                 # Task extraction and prioritization
    ACTION = "action"             # Tool selection and execution
    REFLECTION = "reflection"     # Self-reflection and learning
    DEBUG = "debug"               # Error analysis and recovery
    MEMORY = "memory"             # Memory retrieval and importance


@dataclass
class NeuronConfig:
    """Configuration for a cortex neuron.

    Attributes:
        name: Neuron identifier (e.g., "think_cortex")
        domain: Cognitive domain this neuron handles
        base_model: Base model identifier (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        lora_rank: LoRA rank for adapter (8-16 typical)
        confidence_threshold: Minimum confidence to avoid EGO fallback (0.0-1.0)
        max_tokens: Maximum generation length
        temperature: Sampling temperature
        system_prompt: Domain-specific system prompt
    """

    name: str
    domain: NeuronDomain
    base_model: str
    lora_rank: int = 16
    confidence_threshold: float = 0.7
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: str = ""

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 < self.lora_rank <= 64:
            raise ValueError(f"lora_rank must be 1-64, got {self.lora_rank}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class NeuronOutput:
    """Output from a neuron inference.

    Attributes:
        content: Generated text output
        confidence: Estimated confidence (0.0-1.0)
        should_fallback: Whether to fallback to EGO
        inference_time: Time taken in seconds
        metadata: Additional domain-specific data
    """

    content: str
    confidence: float
    should_fallback: bool = False
    inference_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set fallback flag."""
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(f"Invalid confidence {self.confidence}, clamping to [0, 1]")
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Experience:
    """Training experience for continual learning.

    Captures input-output pairs with outcomes for future fine-tuning.

    Attributes:
        timestamp: When this experience occurred
        input_data: Input provided to neuron
        output: What the neuron generated
        expected: What should have been generated (if known)
        user_feedback: Optional user feedback (positive/negative/neutral)
        outcome_score: Numeric outcome quality (0.0-1.0)
        domain: Which neuron domain this belongs to
        metadata: Additional context
    """

    timestamp: datetime
    input_data: Dict[str, Any]
    output: str
    expected: Optional[str] = None
    user_feedback: Optional[str] = None
    outcome_score: float = 0.5
    domain: NeuronDomain = NeuronDomain.THINKING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_training_sample(self) -> Dict[str, str]:
        """Convert to training sample format for fine-tuning.

        Returns:
            Dictionary with 'input' and 'output' keys for training.
        """
        # Use expected if available, otherwise use actual output
        target = self.expected if self.expected else self.output

        return {
            "input": str(self.input_data),
            "output": target,
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "outcome_score": self.outcome_score,
                "user_feedback": self.user_feedback,
                "domain": self.domain.value,
            }
        }


class CortexNeuron(ABC):
    """Base class for all cortex neurons.

    Each neuron is a specialized cognitive function backed by a small,
    fine-tuned language model. Neurons estimate their own confidence and
    trigger fallback to the EGO (larger model) when uncertain.

    Architecture:
        Base model: Small LLM (0.5B-2B params)
        Adapter: LoRA fine-tuned on domain data
        Backend: MLX for Apple Silicon efficiency

    Usage:
        neuron = ThinkCortex(config)
        output = neuron.infer({"context": "...", "needs": {...}})
        if output.should_fallback:
            # Use EGO instead
            pass
        neuron.record_outcome(output, actual_outcome, score=0.8)
    """

    def __init__(self, config: NeuronConfig):
        """Initialize neuron with configuration.

        Args:
            config: Neuron configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.adapter_path: Optional[Path] = None
        self.experiences: List[Experience] = []
        self._stub_mode: bool = False

        logger.info(
            f"Initializing {config.name} for domain {config.domain.value} "
            f"(base: {config.base_model}, rank: {config.lora_rank})"
        )

    def load_adapter(self, adapter_path: Path) -> None:
        """Load LoRA adapter weights.

        Args:
            adapter_path: Path to adapter weights directory

        Raises:
            FileNotFoundError: If adapter path doesn't exist
        """
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        self.adapter_path = adapter_path
        logger.info(f"{self.config.name}: Loaded adapter from {adapter_path}")

        # Actual model loading happens lazily on first inference
        # to save memory until needed

    def _load_model(self) -> None:
        """Lazy load the model and adapter.

        This is called automatically on first inference to avoid loading
        all models at initialization time.
        """
        if self.model is not None:
            return  # Already loaded

        try:
            import mlx.core as mx
            from mlx_lm import load

            logger.info(f"{self.config.name}: Loading {self.config.base_model}")

            # Load base model with adapter if available
            if self.adapter_path:
                self.model, self.tokenizer = load(
                    self.config.base_model,
                    adapter_path=str(self.adapter_path)
                )
            else:
                self.model, self.tokenizer = load(self.config.base_model)

            logger.info(f"{self.config.name}: Model loaded successfully")

        except ImportError:
            logger.warning(
                f"{self.config.name}: MLX not available - using stub outputs. "
                "Install mlx and mlx-lm for real inference."
            )
            self._stub_mode = True
            self.model = None
            self.tokenizer = None
        except Exception as e:
            logger.error(f"{self.config.name}: Failed to load model: {e}")
            raise
        except Exception as e:
            logger.error(f"{self.config.name}: Failed to load model: {e}")
            raise

    @abstractmethod
    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare input prompt from data.

        Each neuron implements domain-specific prompt formatting.

        Args:
            input_data: Raw input data

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse model output into structured data.

        Each neuron extracts domain-specific structure from text.

        Args:
            raw_output: Raw text from model

        Returns:
            Structured output dictionary
        """
        pass

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in the output.

        Default implementation uses heuristics:
        - Output length (too short or too long is suspicious)
        - Presence of expected structure
        - Repetition detection
        - Coherence checks

        Subclasses can override for domain-specific confidence estimation.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed structured output

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 1.0

        # Length check: penalize very short or very long outputs
        output_length = len(raw_output)
        if output_length < 20:
            confidence *= 0.5
        elif output_length > self.config.max_tokens * 0.95:
            confidence *= 0.8  # Might be truncated

        # Structure check: penalize missing expected fields
        if not parsed_output:
            confidence *= 0.3

        # Repetition check: detect repeated phrases
        words = raw_output.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                confidence *= 0.6  # High repetition

        # Coherence check: detect incomplete sentences
        if not raw_output.rstrip().endswith(('.', '!', '?', '"')):
            confidence *= 0.9

        return confidence

    def infer(
        self,
        input_data: Dict[str, Any],
        force_inference: bool = False
    ) -> NeuronOutput:
        """Run inference on input data.

        Args:
            input_data: Domain-specific input dictionary
            force_inference: If True, ignore confidence threshold

        Returns:
            NeuronOutput with result and confidence
        """
        start_time = time.time()

        try:
            # Lazy load model
            self._load_model()

            # Prepare prompt
            prompt = self._prepare_prompt(input_data)

            # Stub path when MLX unavailable
            if self._stub_mode or self.model is None:
                raw_output = f"STUB: {prompt[:200]}"
                parsed = self._parse_output(raw_output)
                confidence = 0.2
                return NeuronOutput(
                    content=raw_output,
                    confidence=confidence,
                    should_fallback=True,
                    inference_time=time.time() - start_time,
                    metadata=parsed
                )

            # Generate using modern MLX-LM API with sampler
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=self.config.temperature)
            raw_output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                sampler=sampler,
                verbose=False
            )

            # Parse output
            parsed = self._parse_output(raw_output)

            # Estimate confidence
            confidence = self._estimate_confidence(input_data, raw_output, parsed)

            # Check fallback threshold
            should_fallback = (
                not force_inference and
                confidence < self.config.confidence_threshold
            )

            inference_time = time.time() - start_time

            if should_fallback:
                logger.warning(
                    f"{self.config.name}: Low confidence {confidence:.2f} < "
                    f"{self.config.confidence_threshold:.2f}, recommending EGO fallback"
                )

            return NeuronOutput(
                content=raw_output,
                confidence=confidence,
                should_fallback=should_fallback,
                inference_time=inference_time,
                metadata=parsed
            )

        except Exception as e:
            logger.error(f"{self.config.name}: Inference failed: {e}")
            inference_time = time.time() - start_time

            return NeuronOutput(
                content=f"ERROR: {str(e)}",
                confidence=0.0,
                should_fallback=True,
                inference_time=inference_time,
                metadata={"error": str(e)}
            )

    def record_outcome(
        self,
        output: NeuronOutput,
        actual_outcome: Optional[str] = None,
        score: float = 0.5,
        user_feedback: Optional[str] = None
    ) -> None:
        """Record an experience for future training.

        Args:
            output: The neuron's output
            actual_outcome: What should have been generated (ground truth)
            score: Quality score (0.0=bad, 1.0=excellent)
            user_feedback: Optional user feedback
        """
        experience = Experience(
            timestamp=datetime.now(),
            input_data=output.metadata.get("input_data", {}),
            output=output.content,
            expected=actual_outcome,
            outcome_score=score,
            user_feedback=user_feedback,
            domain=self.config.domain,
            metadata=output.metadata
        )

        self.experiences.append(experience)

        logger.debug(
            f"{self.config.name}: Recorded experience "
            f"(score={score:.2f}, feedback={user_feedback})"
        )

    def get_training_data(self, min_score: float = 0.6) -> List[Dict[str, str]]:
        """Export experiences as training data.

        Args:
            min_score: Minimum outcome score to include

        Returns:
            List of training samples
        """
        samples = [
            exp.to_training_sample()
            for exp in self.experiences
            if exp.outcome_score >= min_score
        ]

        logger.info(
            f"{self.config.name}: Exported {len(samples)} training samples "
            f"from {len(self.experiences)} experiences (min_score={min_score})"
        )

        return samples

    def save_experiences(self, path: Path) -> None:
        """Save recorded experiences to file.

        Args:
            path: Output file path (.jsonl format)
        """
        import json

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for exp in self.experiences:
                sample = exp.to_training_sample()
                f.write(json.dumps(sample) + '\n')

        logger.info(f"{self.config.name}: Saved {len(self.experiences)} experiences to {path}")

    def clear_experiences(self) -> None:
        """Clear recorded experiences (after saving to disk)."""
        count = len(self.experiences)
        self.experiences.clear()
        logger.info(f"{self.config.name}: Cleared {count} experiences")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"domain={self.config.domain.value}, "
            f"confidence_threshold={self.config.confidence_threshold}, "
            f"experiences={len(self.experiences)})"
        )

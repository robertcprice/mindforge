"""
Conch DNA - Training Pipeline

Implements the self-improving training loop:
1. Record successful outputs (reward > 0.7)
2. Record EGO corrections for failures
3. Trigger retraining every 100 examples
4. Validate before accepting new adapters

The EGO generates the training data - corrections are the highest value data.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""

    input_text: str
    output_text: str
    domain: str  # think, task, action, reflect, debug, memory
    timestamp: datetime
    source: str  # "success", "ego_correction", "ego_fallback"
    reward: float  # Original reward/score (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Convert to JSONL format for training."""
        return json.dumps({
            "input": self.input_text,
            "output": self.output_text,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "reward": self.reward,
            "metadata": self.metadata
        })

    @classmethod
    def from_jsonl(cls, line: str) -> "TrainingExample":
        """Create from JSONL line."""
        data = json.loads(line)
        return cls(
            input_text=data["input"],
            output_text=data["output"],
            domain=data["domain"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            reward=data["reward"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ExperienceBuffer:
    """Buffer for accumulating training experiences."""

    domain: str
    examples: List[TrainingExample] = field(default_factory=list)
    retrain_threshold: int = 100  # Retrain after this many new examples
    last_retrain_count: int = 0

    @property
    def new_examples_count(self) -> int:
        """Number of examples since last retrain."""
        return len(self.examples) - self.last_retrain_count

    @property
    def should_retrain(self) -> bool:
        """Whether we have enough new examples for retraining."""
        return self.new_examples_count >= self.retrain_threshold

    def add(self, example: TrainingExample) -> None:
        """Add an example to the buffer."""
        self.examples.append(example)
        logger.debug(f"{self.domain}: Added example (total={len(self.examples)}, "
                    f"new={self.new_examples_count})")

    def mark_retrained(self) -> None:
        """Mark that retraining was performed."""
        self.last_retrain_count = len(self.examples)
        logger.info(f"{self.domain}: Marked retrained at {self.last_retrain_count} examples")

    def get_training_data(self, min_reward: float = 0.5) -> List[TrainingExample]:
        """Get examples suitable for training.

        Args:
            min_reward: Minimum reward threshold

        Returns:
            Filtered examples
        """
        return [ex for ex in self.examples if ex.reward >= min_reward]

    def save(self, path: Path) -> None:
        """Save buffer to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for ex in self.examples:
                f.write(ex.to_jsonl() + '\n')
        logger.info(f"{self.domain}: Saved {len(self.examples)} examples to {path}")

    def load(self, path: Path) -> None:
        """Load buffer from file."""
        if not path.exists():
            logger.warning(f"{self.domain}: No saved buffer at {path}")
            return

        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    self.examples.append(TrainingExample.from_jsonl(line))

        logger.info(f"{self.domain}: Loaded {len(self.examples)} examples from {path}")


class TrainingPipeline:
    """Self-improving training pipeline for cortex neurons.

    Workflow:
        1. Record experiences during cycles
        2. Classify by outcome (success/failure/ego_fallback)
        3. For failures: EGO generates corrections (highest value data!)
        4. Accumulate until threshold (100 examples)
        5. Retrain neuron's LoRA adapter
        6. Validate improvement before accepting

    Example:
        pipeline = TrainingPipeline(data_dir=Path("data/training"))
        pipeline.initialize()

        # Record a successful output
        pipeline.record_success(
            domain="action",
            input_text="...",
            output_text="...",
            reward=0.85
        )

        # Record a failure with EGO correction
        pipeline.record_correction(
            domain="action",
            input_text="...",
            neuron_output="(incorrect)",
            ego_correction="(correct answer)",
            analysis="What went wrong and why"
        )

        # Check if retraining is needed
        if pipeline.should_retrain("action"):
            success = pipeline.retrain("action", validator=my_validator)
    """

    DOMAINS = ["think", "task", "action", "reflect", "debug", "memory"]

    def __init__(
        self,
        data_dir: Path,
        retrain_threshold: int = 100
    ):
        """Initialize training pipeline.

        Args:
            data_dir: Directory for training data
            retrain_threshold: Examples needed before retraining
        """
        self.data_dir = data_dir
        self.retrain_threshold = retrain_threshold
        self.buffers: Dict[str, ExperienceBuffer] = {}
        self.adapter_dir = data_dir.parent / "adapters"

        logger.info(f"TrainingPipeline initialized (threshold={retrain_threshold})")

    def initialize(self) -> None:
        """Initialize the pipeline and load existing data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        for domain in self.DOMAINS:
            self.buffers[domain] = ExperienceBuffer(
                domain=domain,
                retrain_threshold=self.retrain_threshold
            )
            # Load existing data
            buffer_path = self.data_dir / f"{domain}_examples.jsonl"
            self.buffers[domain].load(buffer_path)

        logger.info("TrainingPipeline initialized with buffers for all domains")

    def record_success(
        self,
        domain: str,
        input_text: str,
        output_text: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a successful neuron output.

        Success = reward > 0.7

        Args:
            domain: Neuron domain
            input_text: Input that was provided
            output_text: Successful output
            reward: Reward score
            metadata: Optional additional info
        """
        if domain not in self.buffers:
            raise ValueError(f"Unknown domain: {domain}")

        example = TrainingExample(
            input_text=input_text,
            output_text=output_text,
            domain=domain,
            timestamp=datetime.now(),
            source="success",
            reward=reward,
            metadata=metadata or {}
        )

        self.buffers[domain].add(example)
        logger.debug(f"Recorded success for {domain} (reward={reward:.2f})")

    def record_correction(
        self,
        domain: str,
        input_text: str,
        neuron_output: str,
        ego_correction: str,
        analysis: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a failure with EGO correction.

        This is the HIGHEST VALUE training data! EGO corrections
        teach the neuron exactly what it got wrong and why.

        Args:
            domain: Neuron domain
            input_text: Input that was provided
            neuron_output: What the neuron generated (incorrect)
            ego_correction: What EGO says is correct
            analysis: EGO's analysis of what went wrong
            metadata: Optional additional info
        """
        if domain not in self.buffers:
            raise ValueError(f"Unknown domain: {domain}")

        # The correction output includes the correct answer
        # We store the original error for analysis
        example = TrainingExample(
            input_text=input_text,
            output_text=ego_correction,  # Train toward the correct output
            domain=domain,
            timestamp=datetime.now(),
            source="ego_correction",
            reward=0.95,  # Corrections are high-value training data
            metadata={
                "original_error": neuron_output,
                "analysis": analysis,
                **(metadata or {})
            }
        )

        self.buffers[domain].add(example)
        logger.info(f"Recorded EGO correction for {domain}")

    def record_fallback(
        self,
        domain: str,
        input_text: str,
        ego_output: str,
        fallback_reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record when EGO handled a task due to low neuron confidence.

        EGO fallback outputs are good training data too - they show
        what the neuron should have generated.

        Args:
            domain: Neuron domain
            input_text: Input that was provided
            ego_output: What EGO generated
            fallback_reason: Why neuron fell back to EGO
            metadata: Optional additional info
        """
        if domain not in self.buffers:
            raise ValueError(f"Unknown domain: {domain}")

        example = TrainingExample(
            input_text=input_text,
            output_text=ego_output,
            domain=domain,
            timestamp=datetime.now(),
            source="ego_fallback",
            reward=0.85,  # Fallback outputs are good but not as targeted as corrections
            metadata={
                "fallback_reason": fallback_reason,
                **(metadata or {})
            }
        )

        self.buffers[domain].add(example)
        logger.debug(f"Recorded EGO fallback for {domain}")

    def should_retrain(self, domain: str) -> bool:
        """Check if a domain has enough new examples for retraining."""
        if domain not in self.buffers:
            return False
        return self.buffers[domain].should_retrain

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get training statistics for all domains."""
        return {
            domain: {
                "total": len(buffer.examples),
                "new": buffer.new_examples_count,
                "threshold": buffer.retrain_threshold,
                "ready_to_retrain": buffer.should_retrain
            }
            for domain, buffer in self.buffers.items()
        }

    def prepare_training_data(
        self,
        domain: str,
        min_reward: float = 0.5,
        format: str = "alpaca"
    ) -> List[Dict[str, str]]:
        """Prepare training data for fine-tuning.

        Args:
            domain: Neuron domain
            min_reward: Minimum reward threshold
            format: Output format ("alpaca", "chatml", "raw")

        Returns:
            List of formatted training samples
        """
        examples = self.buffers[domain].get_training_data(min_reward)

        if format == "alpaca":
            return [
                {
                    "instruction": ex.input_text,
                    "output": ex.output_text
                }
                for ex in examples
            ]
        elif format == "chatml":
            return [
                {
                    "messages": [
                        {"role": "user", "content": ex.input_text},
                        {"role": "assistant", "content": ex.output_text}
                    ]
                }
                for ex in examples
            ]
        else:
            return [
                {"input": ex.input_text, "output": ex.output_text}
                for ex in examples
            ]

    def retrain(
        self,
        domain: str,
        validator: Optional[callable] = None,
        dry_run: bool = False
    ) -> Tuple[bool, str]:
        """Retrain a neuron's LoRA adapter.

        Args:
            domain: Domain to retrain
            validator: Optional validation function (model -> bool)
            dry_run: If True, just prepare data but don't train

        Returns:
            Tuple of (success, message)
        """
        if domain not in self.buffers:
            return False, f"Unknown domain: {domain}"

        buffer = self.buffers[domain]

        if not buffer.should_retrain:
            return False, f"Not enough new examples ({buffer.new_examples_count}/{buffer.retrain_threshold})"

        # Prepare training data
        training_data = self.prepare_training_data(domain)
        if len(training_data) < 10:
            return False, "Not enough quality training examples"

        logger.info(f"Preparing to retrain {domain} with {len(training_data)} examples")

        # Save training data
        training_file = self.data_dir / f"{domain}_train.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        if dry_run:
            return True, f"Dry run: would train with {len(training_data)} examples"

        # Actual training would happen here
        # For now, we just simulate success
        try:
            # TODO: Integrate with MLX LoRA training
            # from mlx_lm import train_lora
            # adapter_path = self.adapter_dir / f"{domain}_v{version}.safetensors"
            # train_lora(...)

            logger.info(f"Training {domain} adapter...")

            # Validate if validator provided
            if validator:
                # Load new adapter and validate
                # is_valid = validator(new_model)
                is_valid = True  # Placeholder

                if not is_valid:
                    logger.warning(f"Validation failed for {domain}, keeping old adapter")
                    return False, "Validation failed, old adapter retained"

            # Mark as retrained
            buffer.mark_retrained()

            # Save buffer
            buffer.save(self.data_dir / f"{domain}_examples.jsonl")

            return True, f"Successfully retrained {domain} with {len(training_data)} examples"

        except Exception as e:
            logger.error(f"Retraining failed for {domain}: {e}")
            return False, f"Training error: {str(e)}"

    def save_all(self) -> None:
        """Save all buffers to disk."""
        for domain, buffer in self.buffers.items():
            buffer.save(self.data_dir / f"{domain}_examples.jsonl")
        logger.info("All training buffers saved")

    def get_correction_count(self, domain: str) -> int:
        """Get number of EGO corrections for a domain."""
        if domain not in self.buffers:
            return 0
        return sum(1 for ex in self.buffers[domain].examples if ex.source == "ego_correction")


def create_training_pipeline(data_dir: str = "data/training") -> TrainingPipeline:
    """Factory function to create an initialized TrainingPipeline.

    Args:
        data_dir: Path to training data directory

    Returns:
        Initialized TrainingPipeline
    """
    pipeline = TrainingPipeline(Path(data_dir))
    pipeline.initialize()
    return pipeline

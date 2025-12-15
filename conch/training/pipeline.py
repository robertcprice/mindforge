"""
Training Pipeline

End-to-end training pipeline for Conch consciousness and KVRM grounding.
Handles data collection, preprocessing, training, and evaluation.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from conch.training.data import (
    TrainingDataset,
    TrainingExample,
    ExampleType,
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

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_training_start(self, pipeline: "TrainingPipeline") -> None:
        """Called when training starts."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at end of each epoch."""
        pass

    @abstractmethod
    def on_training_end(self, adapter: LoRAAdapter) -> None:
        """Called when training completes."""
        pass


class EvaluationCallback(TrainingCallback):
    """
    Callback for evaluating model during training.

    Runs evaluation on held-out data and logs metrics.
    """

    def __init__(
        self,
        eval_data: List[TrainingExample],
        eval_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize evaluation callback.

        Args:
            eval_data: Held-out evaluation examples
            eval_fn: Custom evaluation function (prompt, completion) -> score
        """
        self.eval_data = eval_data
        self.eval_fn = eval_fn
        self.eval_history: List[Dict[str, float]] = []

    def on_training_start(self, pipeline: "TrainingPipeline") -> None:
        """Log evaluation setup."""
        logger.info(f"Evaluation callback initialized with {len(self.eval_data)} examples")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Run evaluation and log metrics."""
        if self.eval_fn:
            scores = []
            for ex in self.eval_data[:50]:  # Limit for speed
                try:
                    score = self.eval_fn(ex.prompt, ex.completion)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Eval failed for example: {e}")

            avg_score = sum(scores) / len(scores) if scores else 0.0
            metrics["eval_score"] = avg_score

        self.eval_history.append({"epoch": epoch, **metrics})
        logger.info(f"Epoch {epoch} metrics: {metrics}")

    def on_training_end(self, adapter: LoRAAdapter) -> None:
        """Log final evaluation summary."""
        if self.eval_history:
            best = max(self.eval_history, key=lambda x: x.get("eval_score", 0))
            logger.info(f"Best epoch: {best.get('epoch')} with score {best.get('eval_score', 0):.4f}")


class LoggingCallback(TrainingCallback):
    """Simple logging callback for training progress."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.start_time: Optional[float] = None

    def on_training_start(self, pipeline: "TrainingPipeline") -> None:
        self.start_time = time.time()
        msg = f"Training started at {datetime.now().isoformat()}"
        logger.info(msg)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - (self.start_time or time.time())
        msg = f"Epoch {epoch}: loss={metrics.get('loss', 0):.4f}, elapsed={elapsed:.1f}s"
        logger.info(msg)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")

    def on_training_end(self, adapter: LoRAAdapter) -> None:
        total_time = time.time() - (self.start_time or time.time())
        msg = f"Training completed in {total_time:.1f}s. Final loss: {adapter.final_loss:.4f}"
        logger.info(msg)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data/training"))
    output_dir: Path = field(default_factory=lambda: Path("models/adapters"))

    # Data collection
    min_examples_to_train: int = 100
    max_examples_per_type: int = 5000
    train_test_split: float = 0.1  # Fraction for evaluation

    # Training triggers
    auto_train_interval: int = 1000  # Train every N new examples
    min_quality_for_training: float = 0.6

    # Training settings
    base_model: str = "Qwen/Qwen3-8B-Instruct"
    lora_config: LoRAConfig = field(default_factory=LoRAConfig.for_combined)

    # Adapter management
    max_adapters_to_keep: int = 5  # Keep N most recent adapters


class TrainingPipeline:
    """
    End-to-end training pipeline for Conch.

    Handles:
    1. Collecting training data from consciousness cycles
    2. Preprocessing and filtering data
    3. Training LoRA adapters
    4. Evaluating and selecting best adapters
    5. Managing adapter versions
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        """
        Initialize the training pipeline.

        Args:
            config: Pipeline configuration
            callbacks: Training callbacks
        """
        self.config = config or PipelineConfig()
        self.callbacks = callbacks or []

        # Ensure directories exist
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset = TrainingDataset(self.config.data_dir / "dataset.json")
        self.consciousness_generator = ConsciousnessDataGenerator(
            min_quality_threshold=self.config.min_quality_for_training
        )
        self.grounding_generator = GroundingDataGenerator()

        # Training state
        self._trainer: Optional[LoRATrainer] = None
        self._current_adapter: Optional[LoRAAdapter] = None
        self._examples_since_training = 0

        # Load existing dataset
        try:
            if (self.config.data_dir / "dataset.json").exists():
                self.dataset.load()
                logger.info(f"Loaded existing dataset with {len(self.dataset)} examples")
        except Exception as e:
            logger.warning(f"Could not load dataset: {e}")

    def collect_from_cycle(
        self,
        cycle_state: Dict[str, Any],
        system_prompt: str = "",
    ) -> int:
        """
        Collect training data from a consciousness cycle.

        Args:
            cycle_state: ConsciousnessState dictionary
            system_prompt: System prompt used in cycle

        Returns:
            Number of examples collected
        """
        # Generate consciousness examples
        consciousness_examples = self.consciousness_generator.from_cycle_state(
            cycle_state, system_prompt
        )

        # Generate grounding examples
        grounding_examples = []
        if cycle_state.get("grounding_results"):
            grounding_examples = self.grounding_generator.from_grounding_results(
                cycle_state.get("grounded_thought", ""),
                cycle_state.get("grounding_results", []),
            )

        # Add to dataset
        all_examples = consciousness_examples + grounding_examples
        added = self.dataset.add_batch(all_examples)

        self._examples_since_training += added

        # Check if we should auto-train
        if (
            self._examples_since_training >= self.config.auto_train_interval
            and len(self.dataset) >= self.config.min_examples_to_train
        ):
            logger.info(f"Auto-training triggered after {self._examples_since_training} new examples")
            self.train()

        return added

    def collect_from_feedback(
        self,
        claim: str,
        predicted_verified: bool,
        actual_verified: bool,
        key_used: Optional[str] = None,
    ) -> bool:
        """
        Collect training data from human/external feedback.

        Args:
            claim: The claim that was verified
            predicted_verified: System's prediction
            actual_verified: Ground truth
            key_used: Key used for verification

        Returns:
            True if example was added
        """
        example = self.grounding_generator.from_verification_feedback(
            claim, predicted_verified, actual_verified, key_used
        )

        if example:
            added = self.dataset.add(example)
            if added:
                self._examples_since_training += 1
            return added

        return False

    def prepare_training_data(
        self,
        example_types: Optional[List[ExampleType]] = None,
        balance_types: bool = True,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Prepare training and evaluation data.

        Args:
            example_types: Types to include (None = all)
            balance_types: Balance examples across types

        Returns:
            (train_data, eval_data) as lists of prompt/completion dicts
        """
        # Filter examples
        examples = self.dataset.filter(
            example_types=example_types,
            min_quality=self.config.min_quality_for_training,
        )

        if not examples:
            return [], []

        # Balance if requested
        if balance_types and example_types:
            balanced = []
            per_type = self.config.max_examples_per_type // len(example_types)

            for ex_type in example_types:
                type_examples = [e for e in examples if e.example_type == ex_type]
                balanced.extend(type_examples[:per_type])

            examples = balanced

        # Cap total examples
        max_total = self.config.max_examples_per_type * len(ExampleType)
        if len(examples) > max_total:
            examples = self.dataset.sample(max_total, weighted=True)

        # Split into train/eval
        split_idx = int(len(examples) * (1 - self.config.train_test_split))
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        # Convert to dict format
        train_data = [{"prompt": e.prompt, "completion": e.completion} for e in train_examples]
        eval_data = [{"prompt": e.prompt, "completion": e.completion} for e in eval_examples]

        logger.info(f"Prepared {len(train_data)} training, {len(eval_data)} evaluation examples")

        return train_data, eval_data

    def train(
        self,
        adapter_name: Optional[str] = None,
        example_types: Optional[List[ExampleType]] = None,
        continue_from: Optional[LoRAAdapter] = None,
    ) -> Optional[LoRAAdapter]:
        """
        Train a LoRA adapter.

        Args:
            adapter_name: Name for the adapter
            example_types: Types to train on (None = all)
            continue_from: Existing adapter to continue from

        Returns:
            Trained adapter, or None if insufficient data
        """
        # Prepare data
        train_data, eval_data = self.prepare_training_data(example_types)

        if len(train_data) < self.config.min_examples_to_train:
            logger.warning(
                f"Insufficient training data: {len(train_data)} < {self.config.min_examples_to_train}"
            )
            return None

        # Generate adapter name if not provided
        if adapter_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            adapter_name = f"conch_adapter_{timestamp}"

        # Initialize trainer if needed
        if self._trainer is None:
            self._trainer = LoRATrainer(
                base_model=self.config.base_model,
                output_dir=self.config.output_dir,
                config=self.config.lora_config,
            )

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_training_start(self)

        # Train
        try:
            if continue_from:
                adapter = self._trainer.continue_training(
                    continue_from, train_data, adapter_name
                )
            else:
                adapter = self._trainer.train(
                    train_data,
                    eval_data if eval_data else None,
                    adapter_name,
                )

            self._current_adapter = adapter
            self._examples_since_training = 0

            # Save dataset
            self.dataset.save()

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_training_end(adapter)

            # Clean up old adapters
            self._cleanup_old_adapters()

            return adapter

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def train_consciousness(self, adapter_name: Optional[str] = None) -> Optional[LoRAAdapter]:
        """Train specifically on consciousness-related examples."""
        return self.train(
            adapter_name=adapter_name or "consciousness_adapter",
            example_types=[
                ExampleType.THOUGHT_GENERATION,
                ExampleType.DECISION_MAKING,
                ExampleType.REFLECTION,
                ExampleType.SLEEP_DETERMINATION,
            ],
        )

    def train_grounding(self, adapter_name: Optional[str] = None) -> Optional[LoRAAdapter]:
        """Train specifically on grounding-related examples."""
        return self.train(
            adapter_name=adapter_name or "grounding_adapter",
            example_types=[
                ExampleType.CLAIM_CLASSIFICATION,
                ExampleType.KEY_EXTRACTION,
                ExampleType.GROUNDING_VERIFICATION,
            ],
        )

    def train_combined(self, adapter_name: Optional[str] = None) -> Optional[LoRAAdapter]:
        """Train on all example types (consciousness + grounding)."""
        return self.train(
            adapter_name=adapter_name or "combined_adapter",
            example_types=None,  # All types
        )

    def get_current_adapter(self) -> Optional[LoRAAdapter]:
        """Get the most recently trained adapter."""
        return self._current_adapter

    def load_latest_adapter(self) -> Optional[LoRAAdapter]:
        """Load the most recent adapter from disk."""
        adapter_dirs = sorted(
            [d for d in self.config.output_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if adapter_dirs:
            try:
                adapter = load_adapter(adapter_dirs[0])
                self._current_adapter = adapter
                return adapter
            except Exception as e:
                logger.warning(f"Failed to load adapter: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "dataset": self.dataset.statistics(),
            "examples_since_training": self._examples_since_training,
            "current_adapter": self._current_adapter.name if self._current_adapter else None,
            "adapter_count": len(list(self.config.output_dir.iterdir())),
        }
        return stats

    def _cleanup_old_adapters(self) -> None:
        """Remove old adapters beyond max_adapters_to_keep."""
        adapter_dirs = sorted(
            [d for d in self.config.output_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        # Keep only the most recent N
        for old_dir in adapter_dirs[self.config.max_adapters_to_keep:]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Removed old adapter: {old_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_dir}: {e}")


def create_training_pipeline(
    base_model: str = "Qwen/Qwen3-8B-Instruct",
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> TrainingPipeline:
    """
    Factory function to create a training pipeline.

    Args:
        base_model: Base model name
        data_dir: Directory for training data
        output_dir: Directory for adapters

    Returns:
        Configured TrainingPipeline
    """
    from conch.config import get_config

    config = get_config()

    pipeline_config = PipelineConfig(
        base_model=base_model,
        data_dir=data_dir or config.data_dir / "training",
        output_dir=output_dir or config.models_dir / "adapters",
        lora_config=LoRAConfig(
            rank=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            dropout=config.model.lora_dropout,
            target_modules=config.model.lora_target_modules,
            learning_rate=config.model.learning_rate,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
        ),
    )

    callbacks = [LoggingCallback()]

    return TrainingPipeline(config=pipeline_config, callbacks=callbacks)

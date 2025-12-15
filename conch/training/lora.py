"""
LoRA Training Infrastructure

Handles LoRA adapter creation, training, and management for Conch.
Supports both MLX (Apple Silicon) and transformers backends.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapter training.

    Designed for Apple Silicon optimization while supporting other backends.
    """

    # LoRA hyperparameters
    rank: int = 16  # LoRA rank (r)
    alpha: int = 32  # LoRA alpha (scaling factor)
    dropout: float = 0.05  # Dropout probability
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 2048
    weight_decay: float = 0.01

    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd, adam
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Precision
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "use_bf16": self.use_bf16,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_consciousness(cls) -> "LoRAConfig":
        """Preset for consciousness training (creative, reflective)."""
        return cls(
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            epochs=3,
        )

    @classmethod
    def for_grounding(cls) -> "LoRAConfig":
        """Preset for grounding training (precise, factual)."""
        return cls(
            rank=8,  # Lower rank for more precise adaptation
            alpha=16,
            learning_rate=1e-4,  # Lower LR for stability
            epochs=5,  # More epochs for grounding precision
        )

    @classmethod
    def for_combined(cls) -> "LoRAConfig":
        """Preset for combined consciousness + grounding training."""
        return cls(
            rank=16,
            alpha=32,
            learning_rate=1.5e-4,
            epochs=4,
        )


@dataclass
class LoRAAdapter:
    """
    Represents a trained LoRA adapter.

    Includes metadata about training and performance.
    """

    # Identity
    name: str
    version: str = "1.0"

    # Paths
    weights_path: Optional[Path] = None
    base_model: str = ""

    # Configuration
    config: LoRAConfig = field(default_factory=LoRAConfig)

    # Training metadata
    trained_at: Optional[datetime] = None
    training_examples: int = 0
    training_epochs: int = 0
    final_loss: float = 0.0

    # Performance metrics
    eval_loss: float = 0.0
    eval_accuracy: float = 0.0

    # Provenance
    parent_adapter: Optional[str] = None  # If this is a continuation
    training_data_hash: str = ""

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_metadata(self, path: Path) -> None:
        """Save adapter metadata to JSON."""
        data = {
            "name": self.name,
            "version": self.version,
            "base_model": self.base_model,
            "config": self.config.to_dict(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_examples": self.training_examples,
            "training_epochs": self.training_epochs,
            "final_loss": self.final_loss,
            "eval_loss": self.eval_loss,
            "eval_accuracy": self.eval_accuracy,
            "parent_adapter": self.parent_adapter,
            "training_data_hash": self.training_data_hash,
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_metadata(cls, path: Path) -> "LoRAAdapter":
        """Load adapter metadata from JSON."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            base_model=data.get("base_model", ""),
            config=LoRAConfig.from_dict(data.get("config", {})),
            trained_at=datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None,
            training_examples=data.get("training_examples", 0),
            training_epochs=data.get("training_epochs", 0),
            final_loss=data.get("final_loss", 0.0),
            eval_loss=data.get("eval_loss", 0.0),
            eval_accuracy=data.get("eval_accuracy", 0.0),
            parent_adapter=data.get("parent_adapter"),
            training_data_hash=data.get("training_data_hash", ""),
            metadata=data.get("metadata", {}),
        )


class LoRATrainer:
    """
    Trains LoRA adapters for Conch models.

    Supports multiple backends:
    - MLX (Apple Silicon optimized)
    - Transformers + PEFT (cross-platform)
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Path,
        config: Optional[LoRAConfig] = None,
        backend: str = "auto",
    ):
        """
        Initialize LoRA trainer.

        Args:
            base_model: Base model name or path
            output_dir: Directory for saving adapters
            config: LoRA configuration (uses defaults if None)
            backend: Training backend ("mlx", "transformers", "auto")
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config or LoRAConfig()
        self.backend = self._detect_backend(backend)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self._model = None
        self._tokenizer = None
        self._trainer = None

        logger.info(f"LoRATrainer initialized with backend: {self.backend}")

    def _detect_backend(self, backend: str) -> str:
        """Detect the best available backend."""
        if backend != "auto":
            return backend

        # Try MLX first (Apple Silicon)
        try:
            import mlx.core
            import platform
            if platform.processor() == "arm":
                logger.info("Detected Apple Silicon, using MLX backend")
                return "mlx"
        except ImportError:
            pass

        # Fall back to transformers
        try:
            import transformers
            import peft
            logger.info("Using transformers + PEFT backend")
            return "transformers"
        except ImportError:
            pass

        raise RuntimeError("No suitable backend found. Install mlx or transformers+peft")

    def prepare(self) -> None:
        """
        Prepare model and tokenizer for training.

        Loads base model, applies LoRA configuration.
        """
        if self.backend == "mlx":
            self._prepare_mlx()
        else:
            self._prepare_transformers()

    def _prepare_mlx(self) -> None:
        """Prepare MLX backend."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            logger.info(f"Loading base model with MLX: {self.base_model}")

            # MLX-specific loading
            self._model, self._tokenizer = load(self.base_model)

            logger.info("MLX model loaded successfully")

        except ImportError as e:
            raise RuntimeError(f"MLX not available: {e}")

    def _prepare_transformers(self) -> None:
        """Prepare transformers + PEFT backend."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            logger.info(f"Loading base model with transformers: {self.base_model}")

            # Quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Prepare for training
            self._model = prepare_model_for_kbit_training(self._model)

            # Apply LoRA
            lora_config = LoraConfig(
                r=self.config.rank,
                lora_alpha=self.config.alpha,
                lora_dropout=self.config.dropout,
                target_modules=list(self.config.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )

            self._model = get_peft_model(self._model, lora_config)

            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        except ImportError as e:
            raise RuntimeError(f"Transformers/PEFT not available: {e}")

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
        adapter_name: str = "consciousness_adapter",
        callbacks: Optional[List[Callable]] = None,
    ) -> LoRAAdapter:
        """
        Train a LoRA adapter.

        Args:
            train_data: List of {"prompt": ..., "completion": ...} dicts
            eval_data: Optional evaluation data
            adapter_name: Name for the adapter
            callbacks: Optional training callbacks

        Returns:
            Trained LoRAAdapter with metadata
        """
        if self._model is None:
            self.prepare()

        logger.info(f"Starting training with {len(train_data)} examples")

        if self.backend == "mlx":
            return self._train_mlx(train_data, eval_data, adapter_name, callbacks)
        else:
            return self._train_transformers(train_data, eval_data, adapter_name, callbacks)

    def _train_mlx(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        adapter_name: str,
        callbacks: Optional[List[Callable]],
    ) -> LoRAAdapter:
        """Train using MLX backend."""
        try:
            import mlx.core as mx
            import mlx.optimizers as optim
            from mlx_lm import tuner

            # Prepare data in MLX format
            def data_generator():
                for item in train_data:
                    text = f"{item['prompt']}\n{item['completion']}"
                    yield {"text": text}

            # Configure tuning
            adapter_path = self.output_dir / adapter_name

            # Run MLX fine-tuning
            tuner.train(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=list(data_generator()),
                val_dataset=list(data_generator())[:min(100, len(train_data))] if not eval_data else [{"text": f"{e['prompt']}\n{e['completion']}"} for e in eval_data],
                lora_layers=self.config.rank,
                batch_size=self.config.batch_size,
                iters=len(train_data) * self.config.epochs // self.config.batch_size,
                learning_rate=self.config.learning_rate,
                adapter_file=str(adapter_path / "adapters.npz"),
            )

            # Create adapter metadata
            adapter = LoRAAdapter(
                name=adapter_name,
                weights_path=adapter_path,
                base_model=self.base_model,
                config=self.config,
                trained_at=datetime.now(),
                training_examples=len(train_data),
                training_epochs=self.config.epochs,
            )

            adapter.save_metadata(adapter_path / "adapter_config.json")

            logger.info(f"MLX training complete. Adapter saved to {adapter_path}")
            return adapter

        except Exception as e:
            logger.error(f"MLX training failed: {e}")
            raise

    def _train_transformers(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        adapter_name: str,
        callbacks: Optional[List[Callable]],
    ) -> LoRAAdapter:
        """Train using transformers + PEFT backend."""
        try:
            import torch
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from datasets import Dataset

            # Prepare datasets
            def prepare_examples(data):
                texts = []
                for item in data:
                    text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
                    texts.append(text)
                return texts

            def tokenize(examples):
                return self._tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                )

            train_texts = prepare_examples(train_data)
            train_dataset = Dataset.from_dict({"text": train_texts})
            train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

            eval_dataset = None
            if eval_data:
                eval_texts = prepare_examples(eval_data)
                eval_dataset = Dataset.from_dict({"text": eval_texts})
                eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

            # Training arguments
            adapter_path = self.output_dir / adapter_name

            training_args = TrainingArguments(
                output_dir=str(adapter_path),
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps if eval_dataset else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                bf16=self.config.use_bf16,
                gradient_checkpointing=self.config.use_gradient_checkpointing,
                optim="adamw_torch",
                lr_scheduler_type=self.config.lr_scheduler,
                weight_decay=self.config.weight_decay,
                save_total_limit=2,
                load_best_model_at_end=True if eval_dataset else False,
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self._tokenizer,
                mlm=False,
            )

            # Create trainer
            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            # Train
            train_result = trainer.train()

            # Save adapter
            self._model.save_pretrained(adapter_path)
            self._tokenizer.save_pretrained(adapter_path)

            # Get metrics
            final_loss = train_result.training_loss
            eval_loss = 0.0
            if eval_dataset:
                eval_result = trainer.evaluate()
                eval_loss = eval_result.get("eval_loss", 0.0)

            # Create adapter metadata
            adapter = LoRAAdapter(
                name=adapter_name,
                weights_path=adapter_path,
                base_model=self.base_model,
                config=self.config,
                trained_at=datetime.now(),
                training_examples=len(train_data),
                training_epochs=self.config.epochs,
                final_loss=final_loss,
                eval_loss=eval_loss,
            )

            adapter.save_metadata(adapter_path / "adapter_config.json")

            logger.info(f"Training complete. Final loss: {final_loss:.4f}, Adapter saved to {adapter_path}")
            return adapter

        except Exception as e:
            logger.error(f"Transformers training failed: {e}")
            raise

    def continue_training(
        self,
        adapter: LoRAAdapter,
        new_data: List[Dict[str, str]],
        adapter_name: Optional[str] = None,
    ) -> LoRAAdapter:
        """
        Continue training an existing adapter with new data.

        Args:
            adapter: Existing adapter to continue from
            new_data: New training data
            adapter_name: New name (defaults to adapter.name + "_continued")

        Returns:
            New trained adapter
        """
        if adapter.weights_path and adapter.weights_path.exists():
            # Load the adapter weights
            if self.backend == "transformers":
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model,
                    str(adapter.weights_path),
                )

        new_name = adapter_name or f"{adapter.name}_continued"
        new_adapter = self.train(new_data, adapter_name=new_name)
        new_adapter.parent_adapter = adapter.name

        return new_adapter


def load_adapter(path: Path) -> LoRAAdapter:
    """
    Load a LoRA adapter from disk.

    Args:
        path: Path to adapter directory

    Returns:
        LoRAAdapter with loaded metadata
    """
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise ValueError(f"No adapter config found at {config_path}")

    adapter = LoRAAdapter.load_metadata(config_path)
    adapter.weights_path = path

    return adapter


def save_adapter(adapter: LoRAAdapter, path: Path) -> None:
    """
    Save a LoRA adapter to disk.

    Args:
        adapter: Adapter to save
        path: Destination path
    """
    path.mkdir(parents=True, exist_ok=True)

    # Copy weights if they exist elsewhere
    if adapter.weights_path and adapter.weights_path != path:
        for file in adapter.weights_path.iterdir():
            if file.is_file():
                shutil.copy(file, path / file.name)

    adapter.weights_path = path
    adapter.save_metadata(path / "adapter_config.json")

    logger.info(f"Adapter saved to {path}")

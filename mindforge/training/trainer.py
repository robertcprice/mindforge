"""
Fine-Tuning Trainer for MindForge

Implements LoRA-based fine-tuning to teach the model:
1. Correct tool response formats
2. When to use tools vs inaction
3. Appropriate tool selection

Supports both:
- Supervised Fine-Tuning (SFT) from labeled examples
- Direct Preference Optimization (DPO) from preference pairs
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model
    base_model: str = "Qwen/Qwen3-8B-Instruct"
    output_dir: str = "./models/fine_tuned"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 2048

    # Data
    train_data_path: str = "./data/training/sft_tool_format.jsonl"
    eval_data_path: Optional[str] = None

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class MindForgeTrainer:
    """
    Trainer for MindForge model fine-tuning.

    Uses LoRA for efficient adaptation of large language models.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None

        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.training_history: List[Dict] = []

    def setup(self):
        """
        Initialize model, tokenizer, and LoRA configuration.

        This requires transformers and peft libraries.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            logger.info("Install with: pip install transformers peft accelerate")
            raise

        logger.info(f"Loading base model: {self.config.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        # Check if we should use CPU or GPU
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            model_kwargs["device_map"] = "mps"
        else:
            logger.warning("No GPU available, training will be slow")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

        logger.info("Model setup complete")

    def load_dataset(self, path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} examples from {path}")
        return data

    def prepare_batch(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for training.

        Converts chat messages to model inputs.
        """
        texts = []

        for example in examples:
            # Format as chat
            messages = example.get("messages", [])

            # Simple formatting for instruction following
            if len(messages) >= 2:
                prompt = messages[0]["content"]
                response = messages[1]["content"]
                text = f"{prompt}\n{response}"
            else:
                text = str(example)

            texts.append(text)

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        # For causal LM, labels are same as input_ids
        encoded["labels"] = encoded["input_ids"].clone()

        return encoded

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one training step."""
        self.peft_model.train()

        # Move to device
        device = next(self.peft_model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = self.peft_model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        return loss.item()

    def train(
        self,
        train_data: Optional[List[Dict]] = None,
        eval_data: Optional[List[Dict]] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Run the full training loop.

        Args:
            train_data: Training examples (or load from config path)
            eval_data: Evaluation examples (optional)
            callback: Function called after each epoch with metrics
        """
        if self.peft_model is None:
            self.setup()

        # Load data
        if train_data is None:
            train_data = self.load_dataset(self.config.train_data_path)

        if eval_data is None and self.config.eval_data_path:
            eval_data = self.load_dataset(self.config.eval_data_path)

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        from torch.optim import AdamW
        optimizer = AdamW(
            self.peft_model.parameters(),
            lr=self.config.learning_rate,
        )

        # Training loop
        logger.info(f"Starting training: {len(train_data)} examples, {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle data
            import random
            random.shuffle(train_data)

            # Process in batches
            for i in range(0, len(train_data), self.config.batch_size):
                batch_examples = train_data[i:i + self.config.batch_size]
                batch = self.prepare_batch(batch_examples)

                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Gradient accumulation
                if (num_batches % self.config.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {self.global_step}, Loss: {loss:.4f}")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

            # End of epoch
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Avg Loss: {avg_loss:.4f}")

            # Evaluation
            if eval_data:
                eval_loss = self.evaluate(eval_data)
                logger.info(f"Eval Loss: {eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint(output_dir / "best")

            # Record history
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "eval_loss": eval_loss if eval_data else None,
                "timestamp": datetime.now().isoformat(),
            }
            self.training_history.append(metrics)

            if callback:
                callback(metrics)

        # Save final model
        self.save_checkpoint(output_dir / "final")
        logger.info(f"Training complete. Model saved to {output_dir}")

    def evaluate(self, eval_data: List[Dict]) -> float:
        """Evaluate on held-out data."""
        self.peft_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(eval_data), self.config.batch_size):
                batch_examples = eval_data[i:i + self.config.batch_size]
                batch = self.prepare_batch(batch_examples)

                device = next(self.peft_model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = self.peft_model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.peft_model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__,
            "history": self.training_history,
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        from peft import PeftModel

        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, path)

        # Load training state
        state_path = path / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.best_eval_loss = state.get("best_eval_loss", float('inf'))
            self.training_history = state.get("history", [])

        logger.info(f"Checkpoint loaded from {path}")


def incremental_train(
    experience_buffer,
    trainer: MindForgeTrainer,
    min_experiences: int = 50,
    epochs_per_session: int = 1,
) -> Dict[str, Any]:
    """
    Incrementally train from collected experiences.

    Called periodically during consciousness loop when enough
    experiences have been collected.

    Args:
        experience_buffer: ExperienceBuffer with collected experiences
        trainer: MindForgeTrainer instance
        min_experiences: Minimum experiences before training
        epochs_per_session: Epochs for this training session

    Returns:
        Training metrics
    """
    stats = experience_buffer.get_stats()

    if stats["size"] < min_experiences:
        logger.info(f"Not enough experiences ({stats['size']}/{min_experiences})")
        return {"skipped": True, "reason": "insufficient_data"}

    # Generate training data from experiences
    train_data = experience_buffer.generate_training_data(
        min_positive_reward=0.3,
        include_negatives=True,
    )

    if len(train_data) < 10:
        logger.info("Not enough high-quality training examples")
        return {"skipped": True, "reason": "low_quality_data"}

    # Update config for incremental training
    original_epochs = trainer.config.num_epochs
    trainer.config.num_epochs = epochs_per_session

    # Train
    trainer.train(train_data=train_data)

    # Restore config
    trainer.config.num_epochs = original_epochs

    return {
        "skipped": False,
        "examples": len(train_data),
        "epochs": epochs_per_session,
        "final_loss": trainer.training_history[-1]["train_loss"] if trainer.training_history else None,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(
        base_model="Qwen/Qwen3-8B-Instruct",
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-5,
    )

    trainer = MindForgeTrainer(config)
    print("Trainer initialized. Run trainer.train() to start fine-tuning.")

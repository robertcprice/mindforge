"""
Conch DNA - Neuron Training Script

Trains student neurons (1.5B) on EGO-generated training data using LoRA.
This script uses the correct mlx_lm API.

Usage:
    python -m conch_dna.training.train_neurons --domain thinking
    python -m conch_dna.training.train_neurons --all
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def train_neuron(
    domain: str,
    training_data_path: Path,
    adapter_dir: Path,
    student_base: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    lora_rank: int = 16,
    learning_rate: float = 1e-4,
    iters: int = 100,
    batch_size: int = 4,
) -> bool:
    """Train a single neuron using LoRA.

    Args:
        domain: The cognitive domain (thinking, task, action, etc.)
        training_data_path: Path to the JSONL training data
        adapter_dir: Directory to save the LoRA adapter
        student_base: Base model to fine-tune
        lora_rank: LoRA rank (higher = more capacity but slower)
        learning_rate: Training learning rate
        iters: Number of training iterations
        batch_size: Training batch size

    Returns:
        True if training succeeded, False otherwise
    """
    try:
        import json
        from mlx_lm import load
        from mlx_lm.tuner import TrainingArgs, train, linear_to_lora_layers
        from mlx_lm.tuner.datasets import TextDataset
        import mlx.optimizers as optim

        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Load student base model
        logger.info(f"Loading student base: {student_base}")
        model, tokenizer = load(student_base)

        # Convert to LoRA
        logger.info(f"Converting to LoRA with rank {lora_rank}")
        # Get number of layers in the model
        num_layers = len(model.model.layers) if hasattr(model, 'model') else 12
        lora_config = {
            "rank": lora_rank,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]
        }
        linear_to_lora_layers(model, num_layers, lora_config)

        # Load training data from JSONL
        logger.info(f"Loading training data from {training_data_path}")
        with open(training_data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(data)} training samples")

        # Create train/val split (90/10)
        split_idx = max(1, int(len(data) * 0.9))
        train_data = data[:split_idx]
        val_data = data[split_idx:] if len(data) > split_idx else data[:1]

        train_set = TextDataset(train_data, tokenizer, text_key="text")
        val_set = TextDataset(val_data, tokenizer, text_key="text")

        # Create optimizer
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Training args - using correct API
        adapter_file = adapter_dir / "adapters.safetensors"
        train_args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            val_batches=5,
            steps_per_report=10,
            steps_per_eval=50,
            steps_per_save=50,
            adapter_file=str(adapter_file)
        )

        logger.info(f"Training {domain} neuron...")
        logger.info(f"  Iterations: {iters}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")

        # Run training
        train(model, optimizer, train_set, val_set, train_args)

        logger.info(f"Training complete! Adapter saved to {adapter_file}")
        return True

    except Exception as e:
        logger.error(f"Training failed for {domain}: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_all_neurons(
    training_data_dir: Path,
    output_dir: Path,
    domains: Optional[list] = None,
    **kwargs
) -> dict:
    """Train all neurons with available training data.

    Args:
        training_data_dir: Directory containing *_training.jsonl files
        output_dir: Directory to save adapters
        domains: List of domains to train (None = all available)
        **kwargs: Additional training arguments

    Returns:
        Dict mapping domain -> success status
    """
    # Find all training data files
    available_domains = []
    for f in training_data_dir.glob("*_training.jsonl"):
        domain = f.stem.replace("_training", "")
        available_domains.append(domain)

    if domains:
        domains = [d for d in domains if d in available_domains]
    else:
        domains = available_domains

    logger.info(f"Training neurons for domains: {domains}")

    results = {}
    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING {domain.upper()} NEURON")
        logger.info(f"{'='*60}")

        training_data_path = training_data_dir / f"{domain}_training.jsonl"
        adapter_dir = output_dir / "adapters" / domain

        success = train_neuron(
            domain=domain,
            training_data_path=training_data_path,
            adapter_dir=adapter_dir,
            **kwargs
        )
        results[domain] = success

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for domain, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {domain}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Conch DNA neurons")
    parser.add_argument("--domain", type=str, help="Specific domain to train")
    parser.add_argument("--all", action="store_true", help="Train all available domains")
    parser.add_argument("--training-data-dir", type=str,
                        default="models/distilled_neurons/training_data",
                        help="Directory containing training data")
    parser.add_argument("--output-dir", type=str,
                        default="models/distilled_neurons",
                        help="Directory to save adapters")
    parser.add_argument("--student-base", type=str,
                        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                        help="Base model for students")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Training learning rate")
    parser.add_argument("--iters", type=int, default=100,
                        help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    training_data_dir = Path(args.training_data_dir)
    output_dir = Path(args.output_dir)

    if not training_data_dir.exists():
        logger.error(f"Training data directory not found: {training_data_dir}")
        return 1

    training_kwargs = {
        "student_base": args.student_base,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "iters": args.iters,
        "batch_size": args.batch_size,
    }

    if args.domain:
        training_data_path = training_data_dir / f"{args.domain}_training.jsonl"
        if not training_data_path.exists():
            logger.error(f"Training data not found: {training_data_path}")
            return 1

        adapter_dir = output_dir / "adapters" / args.domain
        success = train_neuron(
            domain=args.domain,
            training_data_path=training_data_path,
            adapter_dir=adapter_dir,
            **training_kwargs
        )
        return 0 if success else 1

    elif args.all:
        results = train_all_neurons(
            training_data_dir=training_data_dir,
            output_dir=output_dir,
            **training_kwargs
        )
        failed = sum(1 for s in results.values() if not s)
        return failed

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())

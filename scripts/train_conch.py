#!/usr/bin/env python3
"""
Conch Local LoRA Training Script

Train Conch personality and values using MLX on Apple Silicon.
Optimized for M4 Pro with 24GB RAM.

Usage:
    python scripts/train_conch.py [--model MODEL] [--epochs N] [--lr LR]

Examples:
    # Quick test run (4-bit, fast)
    python scripts/train_conch.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --epochs 1

    # Full training (8-bit, better quality)
    python scripts/train_conch.py --model mlx-community/Qwen2.5-7B-Instruct-8bit --epochs 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train Conch LoRA adapter")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--output", default="./models/conch_trained", help="Output directory"
    )
    parser.add_argument(
        "--data", default="./data/training", help="Training data directory"
    )
    args = parser.parse_args()

    # Ensure data exists
    train_file = Path(args.data) / "train.jsonl"
    valid_file = Path(args.data) / "valid.jsonl"

    if not train_file.exists():
        print(f"Error: Training data not found at {train_file}")
        print("Run the data generation script first.")
        sys.exit(1)

    # Count training examples
    with open(train_file) as f:
        num_examples = sum(1 for _ in f)

    # Calculate iterations
    iters_per_epoch = max(1, num_examples // args.batch_size)
    total_iters = iters_per_epoch * args.epochs

    print(f"=" * 60)
    print(f"Conch LoRA Training")
    print(f"=" * 60)
    print(f"Model:          {args.model}")
    print(f"Training data:  {train_file} ({num_examples} examples)")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Batch size:     {args.batch_size}")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"Total iters:    {total_iters}")
    print(f"Output:         {args.output}")
    print(f"=" * 60)

    # Build mlx_lm.lora command
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--train",
        "--data",
        args.data,
        "--iters",
        str(total_iters),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.lr),
        "--num-layers",
        str(args.lora_rank),
        "--adapter-path",
        args.output,
        "--val-batches",
        "1",
        "--steps-per-eval",
        str(max(1, total_iters // 5)),
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Adapter saved to: {args.output}")
        print(f"{'=' * 60}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()

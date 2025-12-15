#!/usr/bin/env python3
"""
Train Distilled Neurons with Recommended Base Models

Uses EGO-generated training data to fine-tune specialized neurons:
- thinking, task, reflection, debug: Ministral-3B (reasoning-optimized)
- action: Qwen3-4B (precision)
- memory: SmolLM2-1.7B (retrieval)

All with LoRA r=32 for more trainable parameters.
"""

import json
import subprocess
import sys
from pathlib import Path

# Configuration
TRAINING_DATA_DIR = Path("models/distilled_neurons/training_data")
ADAPTERS_DIR = Path("models/distilled_neurons/adapters")
CONFIGS_DIR = Path("models/distilled_neurons/configs")

# Model assignments - Llama-3.2-3B for reasoning, Qwen3-4B for action precision
# Llama 3.2 3B: Excellent reasoning, well-supported MLX tokenizer
# r=32 LoRA gives ~8-12M trainable params - 3x more than 1.5B setup
NEURON_MODELS = {
    "thinking": "mlx-community/Llama-3.2-3B-Instruct-4bit",    # reasoning-heavy
    "task": "mlx-community/Llama-3.2-3B-Instruct-4bit",        # decomposition
    "reflection": "mlx-community/Llama-3.2-3B-Instruct-4bit",  # self-analysis
    "debug": "mlx-community/Llama-3.2-3B-Instruct-4bit",       # error analysis
    "action": "mlx-community/Qwen3-4B-4bit",                   # precision/tool-use
    "memory": "mlx-community/Llama-3.2-3B-Instruct-4bit",      # unified for consistency
}

# Training parameters
LORA_RANK = 32
LORA_LAYERS = 8
LEARNING_RATE = 1e-4
ITERS = 100
BATCH_SIZE = 4


def create_lora_config(domain: str) -> Path:
    """Create a LoRA config file for the neuron."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{domain}_lora_config.yaml"

    # Write YAML config for LoRA parameters
    config_content = f"""# LoRA config for {domain} neuron
lora_parameters:
  rank: {LORA_RANK}
  alpha: {LORA_RANK * 2}
  dropout: 0.05
  scale: 1.0
"""
    config_path.write_text(config_content)
    return config_path


def prepare_data_dir(domain: str) -> Path:
    """Prepare data directory with train.jsonl for mlx_lm."""
    src_path = TRAINING_DATA_DIR / f"{domain}_training.jsonl"
    data_dir = TRAINING_DATA_DIR / domain
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    # Copy/link source data to train.jsonl and valid.jsonl
    if src_path.exists():
        import shutil
        shutil.copy(src_path, train_path)
        shutil.copy(src_path, valid_path)  # Use same for validation

    return data_dir


def train_neuron(domain: str) -> bool:
    """Train a single neuron using mlx_lm lora command."""
    model = NEURON_MODELS[domain]
    src_data_path = TRAINING_DATA_DIR / f"{domain}_training.jsonl"
    adapter_dir = ADAPTERS_DIR / domain

    if not src_data_path.exists():
        print(f"[SKIP] No training data for {domain}: {src_data_path}")
        return False

    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data directory with train.jsonl format
    data_dir = prepare_data_dir(domain)

    print(f"\n{'='*60}")
    print(f"TRAINING {domain.upper()} NEURON")
    print(f"  Base Model: {model}")
    print(f"  Data: {data_dir}")
    print(f"  LoRA Rank: {LORA_RANK} (via config)")
    print(f"  Layers: {LORA_LAYERS}")
    print(f"{'='*60}\n")

    # Create config file for LoRA parameters
    config_path = create_lora_config(domain)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--data", str(data_dir),
        "--train",
        "--adapter-path", str(adapter_dir),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(ITERS),
        "--learning-rate", str(LEARNING_RATE),
        "--num-layers", str(LORA_LAYERS),
        "-c", str(config_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n[SUCCESS] {domain} neuron trained -> {adapter_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training {domain} failed: {e}")
        return False


def main():
    """Train all neurons."""
    print("=" * 60)
    print("NEURON DISTILLATION TRAINING")
    print("Using EGO-generated training data with LoRA r=32")
    print("=" * 60)

    # Check training data
    print("\nTraining data status:")
    for domain in NEURON_MODELS:
        data_path = TRAINING_DATA_DIR / f"{domain}_training.jsonl"
        if data_path.exists():
            with open(data_path) as f:
                lines = sum(1 for _ in f)
            print(f"  {domain}: {lines} samples")
        else:
            print(f"  {domain}: MISSING")

    # Train each neuron
    results = {}
    for domain in NEURON_MODELS:
        success = train_neuron(domain)
        results[domain] = "SUCCESS" if success else "FAILED"

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for domain, status in results.items():
        emoji = "[OK]" if status == "SUCCESS" else "[!!]"
        print(f"  {emoji} {domain}: {status}")


if __name__ == "__main__":
    main()

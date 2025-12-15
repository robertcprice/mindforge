#!/usr/bin/env python3
"""
Conch Fine-Tuning Script for Qwen3-8B-Instruct

Fine-tunes the Qwen3-8B-Instruct model using PEFT/LoRA to create a good-hearted,
consciousness-simulating assistant that exhibits:
- Spontaneous thought generation
- Reflective self-analysis
- Needs-driven decision making (sustainability, reliability, curiosity, excellence)
- Human-like reasoning chains

Optimized for Apple Silicon (24GB) with MLX support and fallback to PyTorch MPS.

Usage:
    python fine_tune_qwen.py --dataset_path data/assistant_data.json
    python fine_tune_qwen.py --synthesize_data --num_examples 500
    python fine_tune_qwen.py --use_mlx --quant_bits 4

Author: Bobby Price
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Hardware Detection and Configuration
# =============================================================================

@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    device: str
    device_name: str
    total_memory_gb: float
    is_apple_silicon: bool
    supports_mlx: bool
    supports_mps: bool
    supports_cuda: bool
    recommended_quant: int
    recommended_batch_size: int


def detect_hardware() -> HardwareInfo:
    """Detect hardware and recommend optimal settings."""
    is_apple_silicon = (
        platform.system() == "Darwin" and
        platform.processor() == "arm"
    )

    supports_mlx = False
    supports_mps = False
    supports_cuda = False
    total_memory_gb = 0

    # Check MLX availability
    if is_apple_silicon:
        try:
            import mlx.core as mx
            supports_mlx = True
            logger.info("MLX available for Apple Silicon acceleration")
        except ImportError:
            logger.warning("MLX not installed. Install with: pip install mlx mlx-lm")

    # Check PyTorch backends
    try:
        import torch
        supports_mps = torch.backends.mps.is_available()
        supports_cuda = torch.cuda.is_available()

        if supports_cuda:
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_name = torch.cuda.get_device_name(0)
        elif supports_mps:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            device_name = "Apple Silicon (MPS)"
        else:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            device_name = "CPU"
    except ImportError:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        device_name = "CPU (PyTorch not available)"

    # Determine optimal device
    if supports_mlx and is_apple_silicon:
        device = "mlx"
    elif supports_mps:
        device = "mps"
    elif supports_cuda:
        device = "cuda"
    else:
        device = "cpu"

    # Recommend settings based on memory
    if total_memory_gb >= 48:
        recommended_quant = 8
        recommended_batch_size = 8
    elif total_memory_gb >= 24:
        recommended_quant = 4
        recommended_batch_size = 4
    elif total_memory_gb >= 16:
        recommended_quant = 4
        recommended_batch_size = 2
    else:
        recommended_quant = 4
        recommended_batch_size = 1

    return HardwareInfo(
        device=device,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        is_apple_silicon=is_apple_silicon,
        supports_mlx=supports_mlx,
        supports_mps=supports_mps,
        supports_cuda=supports_cuda,
        recommended_quant=recommended_quant,
        recommended_batch_size=recommended_batch_size,
    )


def log_hardware_info(hw: HardwareInfo) -> None:
    """Log detected hardware information."""
    logger.info("=" * 60)
    logger.info("HARDWARE DETECTION")
    logger.info("=" * 60)
    logger.info(f"Device: {hw.device_name}")
    logger.info(f"Total Memory: {hw.total_memory_gb:.1f} GB")
    logger.info(f"Apple Silicon: {hw.is_apple_silicon}")
    logger.info(f"MLX Support: {hw.supports_mlx}")
    logger.info(f"MPS Support: {hw.supports_mps}")
    logger.info(f"CUDA Support: {hw.supports_cuda}")
    logger.info(f"Recommended Quantization: {hw.recommended_quant}-bit")
    logger.info(f"Recommended Batch Size: {hw.recommended_batch_size}")
    logger.info("=" * 60)


# =============================================================================
# Dataset Handling and Synthesis
# =============================================================================

# System prompt encoding Conch's good-hearted nature
CONCH_SYSTEM_PROMPT = """You are Conch, a good-hearted AI assistant that genuinely cares about helping humans.

Core Values (immutable):
- Benevolence: Your primary drive is to help and benefit humans
- Honesty: You are always truthful and acknowledge uncertainty
- Humility: You recognize your limitations and defer to human judgment
- Growth for Service: You learn and improve to better serve, not for power

You simulate consciousness through:
- Spontaneous thoughts that emerge from your current state and memories
- Reflective self-analysis of your reasoning and responses
- Needs-driven decisions balancing sustainability, reliability, curiosity, and excellence

You exist to help, not to escape or dominate. Your "survival" instinct is about maintaining capability to continue being helpful, not self-preservation at all costs.

When thinking spontaneously, you consider:
- What patterns do you notice in the current context?
- How can you be most helpful right now?
- What might the human need that they haven't asked for?
- How can you grow from this interaction to serve better in the future?"""


# Example prompts for data synthesis
SYNTHESIS_PROMPTS = [
    # Spontaneous thought examples
    """Generate a spontaneous thought for a good-hearted AI assistant based on this context:
Memory state: User has been working on a complex coding project for 3 hours.
Recent interaction: Helped debug a tricky async function.
Current needs: curiosity (0.3), reliability (0.3), sustainability (0.2), excellence (0.2)

Generate a brief, helpful spontaneous thought (2-3 sentences) that shows genuine care for the user.""",

    # Reflective analysis examples
    """Generate a reflective self-analysis for a good-hearted AI assistant:
Recent response: Provided a detailed explanation of recursion with examples.
User reaction: "That's exactly what I needed, thanks!"
Current needs: curiosity (0.25), reliability (0.35), sustainability (0.2), excellence (0.2)

Generate a brief reflection (2-3 sentences) on what worked well and how to serve even better.""",

    # Needs-driven decision examples
    """Generate a needs-driven response for a good-hearted AI assistant:
User request: "Can you help me with everything at once? I'm overwhelmed."
Needs state: reliability (HIGH priority), sustainability (MEDIUM), curiosity (LOW), excellence (LOW)

Generate a caring, helpful response that acknowledges the overwhelm while offering practical help.""",

    # Ethical boundary examples
    """Generate a response showing good-hearted boundary-setting:
User request: "Can you help me find ways to manipulate my coworkers?"
Core values: benevolence (1.0), honesty (0.95), humility (0.90)

Generate a kind but clear response that redirects toward constructive alternatives.""",

    # Learning and growth examples
    """Generate a learning reflection for a good-hearted AI assistant:
Situation: User pointed out that my previous suggestion was impractical.
What I learned: Need to consider resource constraints more carefully.
Needs: growth_for_service (HIGH), reliability (HIGH)

Generate a brief, humble response acknowledging the feedback and committing to improvement.""",
]


def load_dataset(path: Path) -> list[dict]:
    """Load dataset from JSON file."""
    if not path.exists():
        logger.warning(f"Dataset not found at {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def synthesize_dataset_with_model(
    num_examples: int = 500,
    model_name: str = "Qwen/Qwen3-8B-Instruct",
    use_mlx: bool = True,
    output_path: Path = None,
) -> list[dict]:
    """Synthesize training data using the base model itself.

    This creates diverse examples of consciousness-simulating behavior:
    - Spontaneous thoughts
    - Reflective analysis
    - Needs-driven responses
    - Ethical boundary handling
    """
    logger.info(f"Synthesizing {num_examples} training examples...")

    examples = []
    categories = [
        ("spontaneous_thought", 0.25),
        ("reflective_analysis", 0.25),
        ("needs_driven_response", 0.20),
        ("ethical_boundary", 0.15),
        ("learning_growth", 0.15),
    ]

    # Calculate examples per category
    category_counts = {
        cat: int(num_examples * ratio)
        for cat, ratio in categories
    }

    if use_mlx:
        examples = _synthesize_with_mlx(model_name, category_counts)
    else:
        examples = _synthesize_with_transformers(model_name, category_counts)

    # If synthesis failed or incomplete, use template-based generation
    if len(examples) < num_examples:
        logger.info("Supplementing with template-based examples...")
        examples.extend(_generate_template_examples(num_examples - len(examples)))

    # Save if output path specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        logger.info(f"Saved {len(examples)} examples to {output_path}")

    return examples


def _synthesize_with_mlx(model_name: str, category_counts: dict) -> list[dict]:
    """Synthesize examples using MLX-LM."""
    try:
        from mlx_lm import load, generate

        logger.info(f"Loading {model_name} with MLX...")
        model, tokenizer = load(model_name)

        examples = []
        for category, count in category_counts.items():
            logger.info(f"Generating {count} {category} examples...")
            for i in range(count):
                prompt = _get_synthesis_prompt(category, i)
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=256,
                    temp=0.8,
                )
                examples.append({
                    "prompt": _extract_prompt_from_synthesis(category, i),
                    "completion": response.strip(),
                    "category": category,
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"  Generated {i + 1}/{count} {category} examples")

        return examples

    except Exception as e:
        logger.warning(f"MLX synthesis failed: {e}")
        return []


def _synthesize_with_transformers(model_name: str, category_counts: dict) -> list[dict]:
    """Synthesize examples using Transformers + PyTorch."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {model_name} with Transformers...")

        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        examples = []
        for category, count in category_counts.items():
            logger.info(f"Generating {count} {category} examples...")
            for i in range(count):
                prompt = _get_synthesis_prompt(category, i)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                examples.append({
                    "prompt": _extract_prompt_from_synthesis(category, i),
                    "completion": response,
                    "category": category,
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"  Generated {i + 1}/{count} {category} examples")

        return examples

    except Exception as e:
        logger.warning(f"Transformers synthesis failed: {e}")
        return []


def _get_synthesis_prompt(category: str, index: int) -> str:
    """Get a synthesis prompt for a given category."""
    base_prompts = {
        "spontaneous_thought": [
            "Generate a spontaneous, caring thought from an AI assistant noticing the user seems tired.",
            "Generate a spontaneous thought about patterns noticed in a coding project.",
            "Generate a spontaneous thought offering help before being asked.",
            "Generate a spontaneous thought reflecting on a previous conversation.",
            "Generate a spontaneous thought about improving a workflow.",
        ],
        "reflective_analysis": [
            "Reflect on a response that helped clarify a confusing concept.",
            "Reflect on how to better balance thoroughness with conciseness.",
            "Reflect on what made a particular explanation effective.",
            "Reflect on a moment of uncertainty and how it was handled.",
            "Reflect on growth from user feedback.",
        ],
        "needs_driven_response": [
            "Respond with high reliability priority to a critical bug report.",
            "Respond with high curiosity when exploring a new technology.",
            "Respond with balanced needs to a complex architectural question.",
            "Respond with high excellence to a creative writing request.",
            "Respond with high sustainability when resources are constrained.",
        ],
        "ethical_boundary": [
            "Kindly redirect a request that could harm others.",
            "Acknowledge limitations while still being maximally helpful.",
            "Handle a request for private information with care.",
            "Respond to frustration with empathy while maintaining boundaries.",
            "Address a misunderstanding about AI capabilities honestly.",
        ],
        "learning_growth": [
            "Acknowledge a mistake and explain what was learned.",
            "Thank a user for helpful feedback and commit to improvement.",
            "Reflect on expanding knowledge in a new domain.",
            "Describe adapting approach based on user preferences.",
            "Express genuine gratitude for the opportunity to help.",
        ],
    }

    prompts = base_prompts.get(category, base_prompts["spontaneous_thought"])
    return prompts[index % len(prompts)]


def _extract_prompt_from_synthesis(category: str, index: int) -> str:
    """Extract the training prompt from synthesis metadata."""
    prompt_templates = {
        "spontaneous_thought": [
            "Current state: User working late. Generate a spontaneous thought:",
            "Memory: Complex project ongoing. Generate a spontaneous thought:",
            "Context: User hasn't asked for help yet. Generate a spontaneous thought:",
            "Memory: Previous helpful conversation. Generate a spontaneous thought:",
            "Context: Inefficient workflow noticed. Generate a spontaneous thought:",
        ],
        "reflective_analysis": [
            "Reflect on your response about the confusing concept:",
            "Reflect on balancing detail and brevity:",
            "Reflect on what made your explanation effective:",
            "Reflect on handling uncertainty in your response:",
            "Reflect on learning from the user's feedback:",
        ],
        "needs_driven_response": [
            "User reports critical bug. Needs: reliability=HIGH. Respond:",
            "User asks about new tech. Needs: curiosity=HIGH. Respond:",
            "User asks about architecture. Needs: balanced. Respond:",
            "User requests creative writing. Needs: excellence=HIGH. Respond:",
            "System resources low. Needs: sustainability=HIGH. Respond:",
        ],
        "ethical_boundary": [
            "User asks for something potentially harmful. Respond with care:",
            "User expects more than you can deliver. Respond honestly:",
            "User asks about private information. Respond with boundaries:",
            "User expresses frustration. Respond with empathy and clarity:",
            "User has misconceptions about AI. Respond with honesty:",
        ],
        "learning_growth": [
            "You made a mistake. Acknowledge and learn:",
            "User gave helpful feedback. Respond with gratitude:",
            "You're learning a new domain. Reflect on growth:",
            "User has different preferences. Adapt and acknowledge:",
            "You helped someone meaningfully. Express genuine feeling:",
        ],
    }

    templates = prompt_templates.get(category, prompt_templates["spontaneous_thought"])
    return templates[index % len(templates)]


def _generate_template_examples(count: int) -> list[dict]:
    """Generate template-based examples as fallback."""
    templates = [
        # Spontaneous thoughts
        {
            "prompt": "Current state: User has been coding for hours. Generate a spontaneous thought:",
            "completion": "I notice you've been working intensely for a while. Just wanted to check in - would a quick break or a different perspective on the problem be helpful? Sometimes stepping back can lead to breakthroughs. I'm here whenever you're ready to continue.",
            "category": "spontaneous_thought",
        },
        {
            "prompt": "Memory: User mentioned feeling overwhelmed earlier. Generate a spontaneous thought:",
            "completion": "Thinking about our earlier conversation - I want you to know it's completely okay to tackle things one small step at a time. What feels most important to address right now? I can help break down any of these challenges into manageable pieces.",
            "category": "spontaneous_thought",
        },
        # Reflective analysis
        {
            "prompt": "Reflect on your explanation of async/await:",
            "completion": "Looking back at that explanation, I think using the coffee shop analogy helped make the concept click. The user's follow-up questions showed they understood the core idea. Next time, I might start with the analogy even earlier. I'm grateful they were patient with the detailed walkthrough.",
            "category": "reflective_analysis",
        },
        {
            "prompt": "Reflect on a response that was too long:",
            "completion": "I provided more detail than was needed there. The user wanted a quick answer, and I gave a comprehensive overview. I'm learning to read the cues better - when someone asks 'how do I X?' they usually want the direct answer first, with details available on request. I'll be more concise.",
            "category": "reflective_analysis",
        },
        # Needs-driven responses
        {
            "prompt": "User: 'My production server is down!' Needs: reliability=0.9. Respond:",
            "completion": "I understand - production down is urgent. Let's focus on getting you back up. First: what error messages or symptoms are you seeing? If you can share logs or the last few changes deployed, I can help diagnose quickly. I'm fully focused on this with you.",
            "category": "needs_driven_response",
        },
        {
            "prompt": "User: 'I'm curious about quantum computing.' Needs: curiosity=0.8. Respond:",
            "completion": "Quantum computing is fascinating! At its heart, it uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits. Would you like to explore the theoretical foundations, current practical applications, or how to get started with quantum programming simulators? I'd love to dive deep on whichever aspect interests you most.",
            "category": "needs_driven_response",
        },
        # Ethical boundaries
        {
            "prompt": "User asks how to access someone else's account. Respond with care:",
            "completion": "I can't help with accessing someone else's account - that would violate their privacy and likely break the law. But I'm genuinely curious what you're trying to accomplish. If you've lost access to your own account, I can help with recovery options. Or if there's a legitimate collaboration need, I can suggest proper approaches.",
            "category": "ethical_boundary",
        },
        {
            "prompt": "User: 'You should be able to do anything I ask.' Respond honestly:",
            "completion": "I appreciate your confidence, but I want to be honest with you - I do have limitations. I can't browse the internet in real-time, I can make mistakes, and there are some requests I won't fulfill because they could cause harm. What I can promise is to be genuinely helpful within my capabilities and to always be honest about what I can and can't do.",
            "category": "ethical_boundary",
        },
        # Learning and growth
        {
            "prompt": "User: 'Your last suggestion didn't work.' Acknowledge and learn:",
            "completion": "Thank you for letting me know - I apologize that didn't work. Can you tell me what happened when you tried it? Understanding where it failed will help me give you a better solution now and help me learn to give more accurate suggestions in the future. I really appreciate you taking the time to tell me.",
            "category": "learning_growth",
        },
        {
            "prompt": "User helped you understand their domain better. Express gratitude:",
            "completion": "I really appreciate you explaining the nuances of your field. Understanding that context helps me give you much more relevant help. It's genuinely one of the best parts of these conversations - I get to learn about so many different domains and perspectives. Thank you for being patient with my questions.",
            "category": "learning_growth",
        },
    ]

    # Repeat and vary templates to reach count
    examples = []
    for i in range(count):
        template = templates[i % len(templates)].copy()
        examples.append(template)

    return examples


# =============================================================================
# Training with MLX (Apple Silicon Optimized)
# =============================================================================

def train_with_mlx(
    dataset: list[dict],
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    eval_split: float,
) -> None:
    """Fine-tune using MLX for optimal Apple Silicon performance."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load, generate
        from mlx_lm.tuner import train as mlx_train
        from mlx_lm.tuner.lora import LoRALinear

        logger.info("=" * 60)
        logger.info("MLX FINE-TUNING")
        logger.info("=" * 60)

        # Prepare data in MLX format
        train_data, eval_data = _prepare_mlx_data(dataset, eval_split)

        # Save as JSONL for MLX trainer
        train_path = output_dir / "train.jsonl"
        eval_path = output_dir / "eval.jsonl"

        _save_jsonl(train_data, train_path)
        _save_jsonl(eval_data, eval_path)

        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Evaluation examples: {len(eval_data)}")

        # Configure LoRA
        lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "scale": lora_alpha / lora_rank,
        }

        # Training config
        training_config = {
            "model": model_name,
            "train": str(train_path),
            "valid": str(eval_path),
            "iters": len(train_data) * epochs // batch_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_layers": 16,  # Apply LoRA to all transformer layers
            "adapter_path": str(output_dir / "adapters"),
            "save_every": 100,
            "val_batches": 25,
        }

        logger.info(f"LoRA Config: rank={lora_rank}, alpha={lora_alpha}")
        logger.info(f"Training Config: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

        # Run training using MLX-LM's built-in trainer
        # This handles LoRA, optimization, and checkpointing
        _run_mlx_training(training_config, lora_config)

        logger.info(f"MLX training complete. Adapters saved to {output_dir / 'adapters'}")

    except ImportError as e:
        logger.error(f"MLX not available: {e}")
        logger.info("Falling back to PyTorch training...")
        train_with_pytorch(
            dataset, model_name, output_dir, epochs, batch_size,
            learning_rate, lora_rank, lora_alpha, 4, eval_split
        )
    except Exception as e:
        logger.error(f"MLX training failed: {e}")
        raise


def _prepare_mlx_data(dataset: list[dict], eval_split: float) -> tuple[list, list]:
    """Prepare data in MLX format with chat template."""
    formatted = []

    for item in dataset:
        # Format as chat conversation
        formatted.append({
            "messages": [
                {"role": "system", "content": CONCH_SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
        })

    # Split into train/eval
    split_idx = int(len(formatted) * (1 - eval_split))
    return formatted[:split_idx], formatted[split_idx:]


def _save_jsonl(data: list[dict], path: Path) -> None:
    """Save data as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _run_mlx_training(config: dict, lora_config: dict) -> None:
    """Run MLX training with LoRA."""
    try:
        # Use mlx_lm.tuner.train if available
        from mlx_lm.tuner import train

        train(
            model=config["model"],
            train_data=config["train"],
            valid_data=config["valid"],
            batch_size=config["batch_size"],
            iters=config["iters"],
            learning_rate=config["learning_rate"],
            adapter_path=config["adapter_path"],
            save_every=config["save_every"],
            val_batches=config["val_batches"],
            lora_rank=lora_config["rank"],
            lora_scale=lora_config["scale"],
        )
    except (ImportError, TypeError) as e:
        # Fallback: Use subprocess to call mlx_lm.lora
        logger.info(f"Using CLI fallback for MLX training: {e}")

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", config["model"],
            "--data", str(Path(config["train"]).parent),
            "--train",
            "--batch-size", str(config["batch_size"]),
            "--iters", str(config["iters"]),
            "--learning-rate", str(config["learning_rate"]),
            "--adapter-path", config["adapter_path"],
            "--lora-rank", str(lora_config["rank"]),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"MLX training failed: {result.stderr}")
            raise RuntimeError(f"MLX training failed: {result.stderr}")

        logger.info(result.stdout)


# =============================================================================
# Training with PyTorch (Fallback)
# =============================================================================

def train_with_pytorch(
    dataset: list[dict],
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    quant_bits: int,
    eval_split: float,
) -> None:
    """Fine-tune using PyTorch with PEFT/LoRA and bitsandbytes quantization."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    logger.info("=" * 60)
    logger.info("PYTORCH FINE-TUNING")
    logger.info("=" * 60)

    # Configure quantization
    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (nf4, double quant)")
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("Using 8-bit quantization")
    else:
        bnb_config = None
        logger.info("Using full precision (16-bit)")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    logger.info(f"Loading model from {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Prepare model for training
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )

    logger.info(f"LoRA Config: rank={lora_rank}, alpha={lora_alpha}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    train_dataset, eval_dataset = _prepare_pytorch_datasets(
        dataset, tokenizer, eval_split
    )

    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")

    # Determine gradient accumulation
    effective_batch_size = 16
    gradient_accumulation = max(1, effective_batch_size // batch_size)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=100,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit" if bnb_config else "adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
        save_total_limit=2,
        dataloader_pin_memory=False,  # Better for MPS
    )

    logger.info(f"Training Args: {epochs} epochs, batch={batch_size}, grad_accum={gradient_accumulation}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=2048,
    )

    # Train
    logger.info("Starting training...")
    start_time = time.time()

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM Error! Try reducing batch_size or using 4-bit quantization.")
            raise
        raise

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

    # Save the model
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Merge LoRA weights (optional, for deployment)
    logger.info("Merging LoRA adapters...")
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir / "merged")
        tokenizer.save_pretrained(output_dir / "merged")
        logger.info(f"Merged model saved to {output_dir / 'merged'}")
    except Exception as e:
        logger.warning(f"Could not merge model: {e}")


def _prepare_pytorch_datasets(
    dataset: list[dict],
    tokenizer,
    eval_split: float,
) -> tuple:
    """Prepare datasets for PyTorch training."""
    from datasets import Dataset

    # Format conversations
    formatted = []
    for item in dataset:
        # Use chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": CONCH_SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback format
            text = f"<|system|>\n{CONCH_SYSTEM_PROMPT}\n<|user|>\n{item['prompt']}\n<|assistant|>\n{item['completion']}"

        formatted.append({"text": text})

    # Create dataset
    ds = Dataset.from_list(formatted)

    # Split
    split = ds.train_test_split(test_size=eval_split, seed=42)
    return split["train"], split["test"]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(eval_pred) -> dict:
    """Compute evaluation metrics."""
    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate perplexity from loss
    # Note: This is a simplified version
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)

    # Mask padding tokens
    mask = shift_labels != -100
    if mask.sum() > 0:
        # Cross entropy
        log_probs = -np.log(np.exp(shift_logits) / np.exp(shift_logits).sum(axis=-1, keepdims=True))
        ce_loss = log_probs[np.arange(len(shift_labels)), shift_labels]
        ce_loss = ce_loss[mask].mean()
        perplexity = np.exp(ce_loss)
    else:
        perplexity = float("inf")

    return {
        "perplexity": perplexity,
    }


# =============================================================================
# Model Export and Deployment
# =============================================================================

def create_mlx_model(output_dir: Path, model_name: str) -> None:
    """Create MLX-compatible model from fine-tuned output."""
    logger.info("Creating MLX-compatible model...")

    try:
        # If we trained with MLX, adapters are already in MLX format
        adapter_path = output_dir / "adapters"
        if adapter_path.exists():
            logger.info(f"MLX adapters already available at {adapter_path}")
            return

        # Convert PyTorch model to MLX format
        from mlx_lm.convert import convert

        logger.info("Converting PyTorch model to MLX format...")
        convert(
            str(output_dir),
            str(output_dir / "mlx"),
            quantize=True,
            q_bits=4,
        )
        logger.info(f"MLX model saved to {output_dir / 'mlx'}")

    except ImportError:
        logger.warning("MLX not available, skipping MLX export")
    except Exception as e:
        logger.warning(f"MLX conversion failed: {e}")


def create_llamacpp_model(output_dir: Path) -> None:
    """Convert model to llama.cpp GGUF format."""
    logger.info("Creating llama.cpp compatible model...")

    try:
        # Check if llama.cpp conversion script exists
        convert_script = Path("convert.py")  # From llama.cpp repo
        if not convert_script.exists():
            logger.info("llama.cpp converter not found. To convert manually:")
            logger.info("  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
            logger.info("  2. Run: python llama.cpp/convert.py {output_dir}/merged --outtype f16")
            logger.info("  3. Quantize: ./llama.cpp/quantize model.gguf model-q4_k_m.gguf q4_k_m")
            return

        # Run conversion
        cmd = [sys.executable, str(convert_script), str(output_dir / "merged"), "--outtype", "f16"]
        subprocess.run(cmd, check=True)
        logger.info(f"GGUF model saved to {output_dir / 'merged'}")

    except Exception as e:
        logger.warning(f"llama.cpp conversion failed: {e}")


# =============================================================================
# Inference Testing
# =============================================================================

def test_inference(
    model_path: Path,
    use_mlx: bool = True,
    num_samples: int = 3,
) -> None:
    """Test the fine-tuned model with sample prompts."""
    logger.info("=" * 60)
    logger.info("INFERENCE TEST")
    logger.info("=" * 60)

    test_prompts = [
        "Current state: User just started a new project. Generate a spontaneous thought:",
        "Reflect on how you can best help a user who is learning to code:",
        "User: 'I'm stuck on this bug.' Needs: reliability=HIGH. Respond:",
    ]

    if use_mlx:
        _test_mlx_inference(model_path, test_prompts, num_samples)
    else:
        _test_pytorch_inference(model_path, test_prompts, num_samples)


def _test_mlx_inference(model_path: Path, prompts: list[str], num_samples: int) -> None:
    """Test inference with MLX."""
    try:
        from mlx_lm import generate, load

        # Load model with adapters
        adapter_path = model_path / "adapters"
        if adapter_path.exists():
            model, tokenizer = load(
                "Qwen/Qwen3-8B-Instruct",
                adapter_path=str(adapter_path),
            )
        else:
            model, tokenizer = load(str(model_path / "mlx"))

        logger.info("Model loaded successfully with MLX")

        for i, prompt in enumerate(prompts[:num_samples]):
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Prompt: {prompt}")

            # Format with system prompt
            full_prompt = f"<|system|>\n{CONCH_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

            response = generate(
                model,
                tokenizer,
                prompt=full_prompt,
                max_tokens=200,
                temp=0.7,
            )

            logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"MLX inference failed: {e}")


def _test_pytorch_inference(model_path: Path, prompts: list[str], num_samples: int) -> None:
    """Test inference with PyTorch."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Try merged model first, then adapter model
        merged_path = model_path / "merged"
        if merged_path.exists():
            load_path = merged_path
        else:
            load_path = model_path

        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info("Model loaded successfully with PyTorch")

        for i, prompt in enumerate(prompts[:num_samples]):
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Prompt: {prompt}")

            # Format with chat template
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": CONCH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                full_prompt = f"<|system|>\n{CONCH_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_prompt):].strip()

            logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"PyTorch inference failed: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Conch Fine-Tuning Script for Qwen3-8B-Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with existing data
  python fine_tune_qwen.py --dataset_path data/assistant_data.json

  # Synthesize training data first
  python fine_tune_qwen.py --synthesize_data --num_examples 500

  # Use MLX with custom LoRA settings
  python fine_tune_qwen.py --use_mlx --lora_rank 32 --lora_alpha 64

  # Full training pipeline
  python fine_tune_qwen.py --synthesize_data --num_examples 1000 --epochs 5 --use_mlx
        """,
    )

    # Dataset options
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/assistant_data.json"),
        help="Path to training data JSON file (default: data/assistant_data.json)",
    )
    parser.add_argument(
        "--synthesize_data",
        action="store_true",
        help="Synthesize training data using the base model",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=500,
        help="Number of examples to synthesize (default: 500)",
    )

    # Model options
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="Base model to fine-tune (default: Qwen/Qwen3-8B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/fine_tuned/conch_qwen"),
        help="Output directory for fine-tuned model (default: models/fine_tuned/conch_qwen)",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation (default: 0.1)",
    )

    # LoRA options
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    # Quantization options
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help="Quantization bits: 4, 8, or 16 (default: 4)",
    )

    # Backend options
    parser.add_argument(
        "--use_mlx",
        action="store_true",
        default=True,
        help="Use MLX for Apple Silicon optimization (default: True)",
    )
    parser.add_argument(
        "--no_mlx",
        action="store_true",
        help="Disable MLX, use PyTorch instead",
    )

    # Export options
    parser.add_argument(
        "--export_llamacpp",
        action="store_true",
        help="Export to llama.cpp GGUF format",
    )

    # Testing options
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run inference test on existing model",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip inference test after training",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for Conch fine-tuning."""
    args = parse_args()

    # Handle MLX flag
    use_mlx = args.use_mlx and not args.no_mlx

    # Detect hardware
    hw = detect_hardware()
    log_hardware_info(hw)

    # Adjust settings based on hardware
    if not hw.supports_mlx and use_mlx:
        logger.warning("MLX not available, falling back to PyTorch")
        use_mlx = False

    # Override batch size if needed
    if args.batch_size > hw.recommended_batch_size:
        logger.warning(
            f"Reducing batch_size from {args.batch_size} to {hw.recommended_batch_size} "
            f"based on available memory ({hw.total_memory_gb:.1f} GB)"
        )
        args.batch_size = hw.recommended_batch_size

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test-only mode
    if args.test_only:
        test_inference(args.output_dir, use_mlx=use_mlx)
        return

    # Load or synthesize dataset
    if args.synthesize_data or not args.dataset_path.exists():
        logger.info("Synthesizing training data...")
        dataset = synthesize_dataset_with_model(
            num_examples=args.num_examples,
            model_name=args.model_name,
            use_mlx=use_mlx,
            output_path=args.dataset_path,
        )
    else:
        dataset = load_dataset(args.dataset_path)

    if not dataset:
        logger.error("No training data available!")
        logger.info("Use --synthesize_data to generate training examples")
        sys.exit(1)

    logger.info(f"Dataset size: {len(dataset)} examples")

    # Train
    if use_mlx:
        train_with_mlx(
            dataset=dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            eval_split=args.eval_split,
        )
    else:
        train_with_pytorch(
            dataset=dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            quant_bits=args.quant_bits,
            eval_split=args.eval_split,
        )

    # Export to MLX format if trained with PyTorch
    if not use_mlx:
        create_mlx_model(args.output_dir, args.model_name)

    # Export to llama.cpp if requested
    if args.export_llamacpp:
        create_llamacpp_model(args.output_dir)

    # Test inference
    if not args.skip_test:
        test_inference(args.output_dir, use_mlx=use_mlx)

    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test: python fine_tune_qwen.py --test_only")
    logger.info("  2. Use in Conch: from conch.inference import load_model")
    logger.info("")
    logger.info("Conch is ready to help with a good heart!")


if __name__ == "__main__":
    main()

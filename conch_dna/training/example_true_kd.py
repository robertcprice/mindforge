#!/usr/bin/env python3
"""
Example usage of TRUE Knowledge Distillation for Conch DNA

This script demonstrates how to use the TrueKnowledgeDistiller to create
specialized student neurons that learn from EGO's probability distributions.

Run with:
    python conch_dna/training/example_true_kd.py
"""

import logging
from pathlib import Path

from conch_dna.training.true_kd import (
    TrueKnowledgeDistiller,
    KDConfig,
    get_kd_adapter_path
)
from conch_dna.training.distillation import DOMAIN_PROMPTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def example_single_domain():
    """Example: Distill a single domain with TRUE KD."""
    
    logger.info("=" * 80)
    logger.info("Example 1: Single Domain Distillation")
    logger.info("=" * 80)
    
    # Create configuration
    config = KDConfig(
        temperature=2.0,      # Soften distributions for more knowledge transfer
        alpha=0.7,            # 70% KD loss, 30% CE loss
        lora_rank=16,         # Standard LoRA rank
        num_iters=100,        # Shorter for demo (use 200+ in production)
        steps_per_report=10   # Log every 10 iterations
    )
    
    # Initialize distiller
    distiller = TrueKnowledgeDistiller(
        teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        config=config
    )
    
    # Get training prompts for thinking domain
    domain = "thinking"
    prompts = DOMAIN_PROMPTS[domain]["examples"]
    
    logger.info(f"Distilling {domain} domain with {len(prompts)} prompts")
    
    # Run distillation
    result = distiller.distill_domain(domain, prompts)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Distillation Complete!")
    logger.info("=" * 80)
    logger.info(f"Domain: {result.domain}")
    logger.info(f"Training Samples: {result.num_samples}")
    logger.info(f"Final KD Loss: {result.final_kd_loss:.4f}")
    logger.info(f"Final CE Loss: {result.final_ce_loss:.4f}")
    logger.info(f"Final Combined Loss: {result.final_combined_loss:.4f}")
    logger.info(f"Training Time: {result.training_time_seconds:.2f}s")
    logger.info(f"Adapter Path: {result.adapter_path}")
    logger.info("=" * 80)
    
    return result


def example_all_domains():
    """Example: Distill all CORTEX domains with TRUE KD."""
    
    logger.info("=" * 80)
    logger.info("Example 2: All Domains Distillation")
    logger.info("=" * 80)
    
    # Create configuration with production settings
    config = KDConfig(
        temperature=2.0,
        alpha=0.7,
        lora_rank=16,
        num_iters=200,        # Full training
        learning_rate=1e-4,
        batch_size=4,
        steps_per_save=100    # Save checkpoints
    )
    
    # Initialize distiller
    distiller = TrueKnowledgeDistiller(
        teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        config=config
    )
    
    # Distill all domains
    logger.info("Starting distillation of all CORTEX neurons...")
    results = distiller.distill_all()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL DOMAINS DISTILLATION COMPLETE")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"\n{result.domain.upper()}:")
        logger.info(f"  Samples: {result.num_samples}")
        logger.info(f"  KD Loss: {result.final_kd_loss:.4f}")
        logger.info(f"  CE Loss: {result.final_ce_loss:.4f}")
        logger.info(f"  Combined: {result.final_combined_loss:.4f}")
        logger.info(f"  Time: {result.training_time_seconds:.2f}s")
        logger.info(f"  Path: {result.adapter_path}")
    
    logger.info("=" * 80)
    
    return results


def example_custom_prompts():
    """Example: Distill with custom training prompts."""
    
    logger.info("=" * 80)
    logger.info("Example 3: Custom Prompts")
    logger.info("=" * 80)
    
    # Custom prompts for a specialized domain
    custom_prompts = [
        "Analyze the performance bottleneck in this code",
        "Debug this authentication flow",
        "Optimize this database query",
        "Review this API design for security",
        "Refactor this code to reduce complexity",
        "Explain this algorithm's time complexity",
        "Design a caching strategy for this endpoint",
        "Identify the memory leak in this function"
    ]
    
    config = KDConfig(
        temperature=2.0,
        alpha=0.7,
        lora_rank=16,
        num_iters=50  # Short demo
    )
    
    distiller = TrueKnowledgeDistiller(
        teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        config=config
    )
    
    # Distill with custom domain name
    result = distiller.distill_domain("code_review", custom_prompts)
    
    logger.info(f"\nCustom domain 'code_review' distilled successfully!")
    logger.info(f"Adapter saved to: {result.adapter_path}")
    
    return result


def example_hyperparameter_tuning():
    """Example: Compare different hyperparameter settings."""
    
    logger.info("=" * 80)
    logger.info("Example 4: Hyperparameter Tuning")
    logger.info("=" * 80)
    
    # Get test prompts
    prompts = DOMAIN_PROMPTS["thinking"]["examples"][:10]  # Small subset for speed
    
    # Test different configurations
    configs = [
        ("Low Temperature", KDConfig(temperature=1.0, alpha=0.7, num_iters=50)),
        ("High Temperature", KDConfig(temperature=5.0, alpha=0.7, num_iters=50)),
        ("High KD Weight", KDConfig(temperature=2.0, alpha=0.9, num_iters=50)),
        ("Balanced", KDConfig(temperature=2.0, alpha=0.5, num_iters=50)),
    ]
    
    results = []
    
    for name, config in configs:
        logger.info(f"\nTesting configuration: {name}")
        logger.info(f"  Temperature: {config.temperature}")
        logger.info(f"  Alpha: {config.alpha}")
        
        distiller = TrueKnowledgeDistiller(
            teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            config=config
        )
        
        result = distiller.distill_domain(f"test_{name.lower().replace(' ', '_')}", prompts)
        results.append((name, result))
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("HYPERPARAMETER COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Configuration':<20} {'KD Loss':<12} {'CE Loss':<12} {'Combined':<12}")
    logger.info("-" * 80)
    
    for name, result in results:
        logger.info(
            f"{name:<20} "
            f"{result.final_kd_loss:<12.4f} "
            f"{result.final_ce_loss:<12.4f} "
            f"{result.final_combined_loss:<12.4f}"
        )
    
    logger.info("=" * 80)
    
    return results


def example_load_and_use_adapter():
    """Example: Load a trained adapter and use it."""
    
    logger.info("=" * 80)
    logger.info("Example 5: Load and Use Trained Adapter")
    logger.info("=" * 80)
    
    # Check if adapter exists
    domain = "thinking"
    adapter_path = get_kd_adapter_path(domain)
    
    if adapter_path is None:
        logger.warning(f"No adapter found for {domain}. Run distillation first!")
        return None
    
    logger.info(f"Found adapter at: {adapter_path}")
    
    # Load adapter config
    import json
    config_file = adapter_path / "adapter_config.json"
    
    if config_file.exists():
        with open(config_file) as f:
            adapter_config = json.load(f)
        
        logger.info("\nAdapter Configuration:")
        logger.info(f"  LoRA Rank: {adapter_config['lora_rank']}")
        logger.info(f"  Temperature: {adapter_config['temperature']}")
        logger.info(f"  Alpha: {adapter_config['alpha']}")
        logger.info(f"  Teacher: {adapter_config['teacher_model']}")
        logger.info(f"  Student: {adapter_config['student_model']}")
        logger.info(f"  Trained: {adapter_config['timestamp']}")
    
    # Load and use the student model with adapter
    from mlx_lm import load, generate
    
    logger.info("\nLoading student model with adapter...")
    model, tokenizer = load(
        adapter_config['student_model'],
        adapter_path=str(adapter_path / "adapters.safetensors")
    )
    
    # Test generation
    test_prompt = "Analyze the trade-offs between microservices and monolithic architecture"
    
    logger.info(f"\nTest Prompt: {test_prompt}")
    logger.info("\nGenerating response...")
    
    response = generate(
        model,
        tokenizer,
        prompt=test_prompt,
        max_tokens=256,
        temp=0.7,
        verbose=False
    )
    
    logger.info(f"\nResponse:\n{response}")
    logger.info("=" * 80)
    
    return response


def main():
    """Run all examples."""
    
    logger.info("\n" + "=" * 80)
    logger.info("TRUE KNOWLEDGE DISTILLATION EXAMPLES")
    logger.info("Conch DNA - Conscious AI Architecture")
    logger.info("=" * 80 + "\n")
    
    # Uncomment the examples you want to run:
    
    # Example 1: Single domain (quick test)
    # example_single_domain()
    
    # Example 2: All domains (production run)
    # example_all_domains()
    
    # Example 3: Custom prompts
    # example_custom_prompts()
    
    # Example 4: Hyperparameter tuning
    # example_hyperparameter_tuning()
    
    # Example 5: Load and use adapter
    # example_load_and_use_adapter()
    
    logger.info("\nExamples completed!")
    logger.info("Uncomment the examples you want to run in the main() function.")


if __name__ == "__main__":
    main()

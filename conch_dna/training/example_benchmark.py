#!/usr/bin/env python3
"""
Example usage of the Conch DNA benchmark suite.

This script demonstrates how to:
1. Run benchmarks programmatically
2. Access results
3. Generate custom reports
"""

import logging
from pathlib import Path

from conch_dna.training.benchmark import (
    BenchmarkConfig,
    NeuronBenchmark,
    run_benchmark,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def example_single_domain():
    """Example: Benchmark a single domain."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Benchmark Single Domain (thinking)")
    print("=" * 70 + "\n")

    config = BenchmarkConfig(
        ego_model="mlx-community/Qwen2.5-7B-Instruct-8bit",
        student_base="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter_dir=Path("models/distilled_neurons/adapters"),
        output_dir=Path("models/distilled_neurons/benchmarks")
    )

    benchmark = NeuronBenchmark(config)
    summary, results = benchmark.benchmark_domain("thinking")

    print(f"\nResults for 'thinking' domain:")
    print(f"  Pass rate: {summary.pass_rate:.1%}")
    print(f"  Recommendation: {summary.recommendation}")

    # Access individual test results
    for i, result in enumerate(results, 1):
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  Test {i}: {status}")
        print(f"    Prompt: {result.prompt}")
        print(f"    JSON Valid: {result.neuron_json_valid}")
        print(f"    Complete: {result.neuron_has_all_fields}")
        print(f"    Quality: {result.neuron_quality_score:.1%}")


def example_all_domains():
    """Example: Benchmark all domains."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Benchmark All Domains")
    print("=" * 70 + "\n")

    # Use convenience function
    results = run_benchmark()

    # Access results programmatically
    for domain, (summary, test_results) in results.items():
        print(f"\n{domain}: {summary.recommendation}")
        print(f"  Pass rate: {summary.pass_rate:.1%}")
        print(f"  Quality gap: {summary.avg_quality_gap:+.1%}")


def example_custom_analysis():
    """Example: Custom analysis of results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Analysis")
    print("=" * 70 + "\n")

    benchmark = NeuronBenchmark()
    all_results = benchmark.benchmark_all()

    # Find domains that need improvement
    needs_work = []
    for domain, (summary, _) in all_results.items():
        if summary.recommendation == "requires_retraining":
            needs_work.append(domain)

    print("Domains requiring retraining:")
    for domain in needs_work:
        print(f"  - {domain}")

    # Calculate overall statistics
    total_tests = sum(s.total_tests for s, _ in all_results.values())
    total_passed = sum(s.passed_tests for s, _ in all_results.values())

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")


def example_result_export():
    """Example: Export results for further analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Export Results")
    print("=" * 70 + "\n")

    benchmark = NeuronBenchmark()

    # Benchmark a single domain
    summary, results = benchmark.benchmark_domain("thinking")

    # Save to custom location
    output_path = benchmark.save_results(
        {"thinking": (summary, results)},
        filename="thinking_benchmark.json"
    )

    print(f"Results exported to: {output_path}")

    # You can then load these results for analysis:
    # import json
    # with open(output_path) as f:
    #     data = json.load(f)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONCH DNA BENCHMARK - USAGE EXAMPLES")
    print("=" * 70)
    print("\nNote: These examples require MLX and trained adapters to run.")
    print("They demonstrate the API, not executable code.\n")

    # Uncomment to run (requires MLX and adapters):
    # example_single_domain()
    # example_all_domains()
    # example_custom_analysis()
    # example_result_export()

    print("\nTo run benchmarks from CLI:")
    print("  python -m conch_dna.training.benchmark --all")
    print("  python -m conch_dna.training.benchmark --domain thinking")
    print("  python -m conch_dna.training.benchmark --verbose")
    print()

#!/usr/bin/env python3
"""
Example: Using the Neuron Benchmark Suite

Demonstrates different ways to run benchmarks and interpret results.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conch_dna.training.benchmark import (
    NeuronBenchmark,
    TEST_PROMPTS,
)


def example_quick_test():
    """Run a quick test on one domain."""
    print("=" * 60)
    print("EXAMPLE 1: Quick Domain Test")
    print("=" * 60)

    benchmark = NeuronBenchmark()

    # Test just 3 prompts from thinking domain
    original_prompts = TEST_PROMPTS["thinking"]
    TEST_PROMPTS["thinking"] = original_prompts[:3]

    domain_result, metrics = benchmark.test_domain("thinking")

    print(f"\nResults Summary:")
    print(f"  Tests: {domain_result.test_count}")
    print(f"  Passed: {domain_result.passed_count}")
    print(f"  Pass Rate: {domain_result.passed_count/domain_result.test_count:.1%}")
    print(f"  Recommendation: {domain_result.recommendation}")

    # Restore original prompts
    TEST_PROMPTS["thinking"] = original_prompts


def example_custom_metrics():
    """Show detailed metrics for each test."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Detailed Metrics Analysis")
    print("=" * 60)

    benchmark = NeuronBenchmark()

    # Run just one test
    prompt = "What are the benefits of using Python over JavaScript?"
    metrics = benchmark.test_single_prompt(prompt, "thinking")

    print(f"\nTest Prompt: {prompt}")
    print(f"\nEGO Output Preview:")
    print(f"  {metrics.ego_output[:200]}...")
    print(f"\nNeuron Output Preview:")
    print(f"  {metrics.neuron_output[:200]}...")

    print(f"\nQuality Metrics:")
    print(f"  JSON Valid (EGO):    {metrics.json_valid_ego}")
    print(f"  JSON Valid (Neuron): {metrics.json_valid_neuron}")
    print(f"  Structure Match:     {metrics.structure_match}")
    print(f"  Key Coverage:        {metrics.key_coverage:.1%}")
    print(f"  Semantic Similarity: {metrics.semantic_similarity:.1%}")

    print(f"\nPerformance Metrics:")
    print(f"  EGO Latency:         {metrics.ego_latency:.3f}s")
    print(f"  Neuron Latency:      {metrics.neuron_latency:.3f}s")
    print(f"  Speedup:             {metrics.speedup:.2f}x")
    print(f"  EGO Memory:          {metrics.ego_memory_mb:.1f} MB")
    print(f"  Neuron Memory:       {metrics.neuron_memory_mb:.1f} MB")
    print(f"  Memory Reduction:    {metrics.memory_reduction:.1%}")

    print(f"\nOverall: {'PASSED' if metrics.passed else 'FAILED'}")


def example_compare_domains():
    """Compare performance across domains."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Domain Comparison")
    print("=" * 60)

    benchmark = NeuronBenchmark()

    # Test 5 prompts from each domain
    for domain in ["thinking", "task", "action"]:
        original_prompts = TEST_PROMPTS[domain]
        TEST_PROMPTS[domain] = original_prompts[:5]

        domain_result, _ = benchmark.test_domain(domain)

        print(f"\n{domain.upper():12s}  "
              f"Pass: {domain_result.passed_count/domain_result.test_count:4.0%}  "
              f"Speed: {domain_result.avg_speedup:4.1f}x  "
              f"Rec: {domain_result.recommendation}")

        # Restore
        TEST_PROMPTS[domain] = original_prompts


def example_quality_gate():
    """Use benchmark as a quality gate for deployment."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Quality Gate for Deployment")
    print("=" * 60)

    benchmark = NeuronBenchmark()

    # Quick test
    original_prompts = TEST_PROMPTS["thinking"]
    TEST_PROMPTS["thinking"] = original_prompts[:10]

    domain_result, _ = benchmark.test_domain("thinking")

    # Restore
    TEST_PROMPTS["thinking"] = original_prompts

    # Decision logic
    pass_rate = domain_result.passed_count / domain_result.test_count

    print(f"\nQuality Gate Results:")
    print(f"  Pass Rate:     {pass_rate:.1%}")
    print(f"  Threshold:     85%")
    print(f"  Speedup:       {domain_result.avg_speedup:.2f}x")
    print(f"  Min Speedup:   1.5x")

    if pass_rate >= 0.85 and domain_result.avg_speedup >= 1.5:
        print(f"\n✓ QUALITY GATE PASSED")
        print(f"  Deploy thinking neuron to production")
        print(f"  Confidence: {domain_result.confidence:.1%}")
    elif pass_rate >= 0.60:
        print(f"\n⚠ QUALITY GATE: NEEDS IMPROVEMENT")
        print(f"  Recommendation: Retrain with more data")
        print(f"  Current samples: 50")
        print(f"  Suggested: 100-200 samples")
    else:
        print(f"\n✗ QUALITY GATE FAILED")
        print(f"  Do not deploy neuron")
        print(f"  Use EGO directly for this domain")
        print(f"  Review training data and model architecture")


def example_identify_weak_areas():
    """Identify specific areas where neuron struggles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Identify Weak Areas")
    print("=" * 60)

    benchmark = NeuronBenchmark()

    # Test with categorized prompts
    simple_prompts = TEST_PROMPTS["thinking"][:10]
    complex_prompts = TEST_PROMPTS["thinking"][10:30]
    edge_prompts = TEST_PROMPTS["thinking"][30:40]

    def test_category(prompts, category_name):
        passed = 0
        total = len(prompts)
        for prompt in prompts:
            metrics = benchmark.test_single_prompt(prompt, "thinking")
            if metrics.passed:
                passed += 1
        return passed, total

    print("\nTesting different complexity levels:")

    simple_passed, simple_total = test_category(simple_prompts[:3], "Simple")
    print(f"  Simple Cases:   {simple_passed}/{simple_total} ({simple_passed/simple_total:.0%})")

    complex_passed, complex_total = test_category(complex_prompts[:3], "Complex")
    print(f"  Complex Cases:  {complex_passed}/{complex_total} ({complex_passed/complex_total:.0%})")

    edge_passed, edge_total = test_category(edge_prompts[:3], "Edge")
    print(f"  Edge Cases:     {edge_passed}/{edge_total} ({edge_passed/edge_total:.0%})")

    print("\nAnalysis:")
    if edge_passed / edge_total < 0.5:
        print("  ⚠ Weak on edge cases - needs more error handling training")
    if complex_passed / complex_total < 0.7:
        print("  ⚠ Weak on complex cases - needs deeper reasoning training")
    if simple_passed / simple_total < 0.9:
        print("  ⚠ Weak on simple cases - fundamental issues with base model")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CONCH DNA BENCHMARK EXAMPLES")
    print("=" * 60)

    examples = [
        ("Quick Test", example_quick_test),
        ("Custom Metrics", example_custom_metrics),
        ("Compare Domains", example_compare_domains),
        ("Quality Gate", example_quality_gate),
        ("Identify Weak Areas", example_identify_weak_areas),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning Example 1 (Quick Test)...")
    print("For other examples, uncomment in main()")

    # Run example 1 by default
    example_quick_test()

    # Uncomment to run others:
    # example_custom_metrics()
    # example_compare_domains()
    # example_quality_gate()
    # example_identify_weak_areas()


if __name__ == "__main__":
    main()

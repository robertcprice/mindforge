#!/usr/bin/env python3
"""
Benchmark Runner for Conch DNA Neurons

Quick script to run neuron benchmarks with various options.

Usage:
    # Benchmark all neurons
    python run_benchmark.py

    # Benchmark specific domain
    python run_benchmark.py --domain thinking

    # Benchmark with custom sample size
    python run_benchmark.py --samples 10

    # Quick test mode (3 samples per domain)
    python run_benchmark.py --quick

    # Verbose output
    python run_benchmark.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add conch_dna to path
sys.path.insert(0, str(Path(__file__).parent))

from conch_dna.training.benchmark import NeuronBenchmark, TEST_PROMPTS


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def limit_test_prompts(max_samples: int):
    """Limit test prompts for quick testing."""
    for domain in TEST_PROMPTS:
        TEST_PROMPTS[domain] = TEST_PROMPTS[domain][:max_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Conch DNA distilled neurons vs EGO"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(TEST_PROMPTS.keys()),
        help="Benchmark specific domain only"
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of test samples per domain (default: all 50)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (3 samples per domain)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/distilled_neurons"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Configure sample size
    if args.quick:
        limit_test_prompts(3)
        print("Quick test mode: 3 samples per domain")
    elif args.samples:
        limit_test_prompts(args.samples)
        print(f"Testing with {args.samples} samples per domain")

    # Create benchmark
    benchmark = NeuronBenchmark(output_dir=args.output_dir)

    # Run benchmark
    if args.domain:
        print(f"\nBenchmarking {args.domain} neuron...\n")
        domain_result, metrics = benchmark.test_domain(args.domain)

        print("\n" + "=" * 60)
        print(f"{args.domain.upper()} BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Tests Run:      {domain_result.test_count}")
        print(f"Passed:         {domain_result.passed_count} ({domain_result.passed_count/domain_result.test_count:.1%})")
        print(f"Failed:         {domain_result.failed_count}")
        print(f"\nQuality Metrics:")
        print(f"  JSON Validity:        {domain_result.avg_json_validity:.1%}")
        print(f"  Structure Match:      {domain_result.avg_structure_match:.1%}")
        print(f"  Key Coverage:         {domain_result.avg_key_coverage:.1%}")
        print(f"  Semantic Similarity:  {domain_result.avg_semantic_similarity:.1%}")
        print(f"\nPerformance Metrics:")
        print(f"  Speedup:              {domain_result.avg_speedup:.2f}x")
        print(f"  Memory Reduction:     {domain_result.avg_memory_reduction:.1%}")
        print(f"\nRecommendation:       {domain_result.recommendation.upper()}")
        print(f"Confidence:           {domain_result.confidence:.1%}")

        # Save single-domain results
        import json
        result_file = args.output_dir / f"benchmark_{args.domain}.json"
        result_data = {
            "domain": domain_result.domain,
            "timestamp": benchmark.output_dir,
            "results": domain_result.__dict__,
            "test_details": [
                {
                    "prompt": m.prompt,
                    "passed": m.passed,
                    "key_coverage": m.key_coverage,
                    "semantic_similarity": m.semantic_similarity,
                    "speedup": m.speedup,
                }
                for m in metrics
            ]
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    else:
        print("\nBenchmarking all neurons...\n")
        report = benchmark.benchmark_all()

        print("\n" + "=" * 60)
        print("OVERALL BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Tests:    {report.total_tests}")
        print(f"Passed:         {report.total_passed} ({report.overall_pass_rate:.1%})")
        print(f"Failed:         {report.total_tests - report.total_passed}")
        print(f"\nPer-Domain Recommendations:")

        for domain_result in report.domains:
            symbol = "✓" if domain_result.recommendation == "use_neuron" else \
                     "⚠" if domain_result.recommendation == "needs_training" else "✗"
            pass_rate = domain_result.passed_count / domain_result.test_count
            print(f"  [{symbol}] {domain_result.domain:12s}  {pass_rate:5.1%}  "
                  f"{domain_result.avg_speedup:4.1f}x  -> {domain_result.recommendation}")

        print(f"\nResults saved to: {args.output_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()

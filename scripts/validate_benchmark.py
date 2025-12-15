#!/usr/bin/env python3
"""
Validate Benchmark Suite Setup

Checks that all components are properly installed and configured.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Check required imports."""
    print("Checking imports...")
    errors = []

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(("numpy", str(e)))
        print("  ✗ numpy - MISSING")

    try:
        import mlx.core as mx
        print("  ✓ mlx.core")
    except ImportError as e:
        errors.append(("mlx.core", str(e)))
        print("  ✗ mlx.core - MISSING")

    try:
        from mlx_lm import load, generate
        print("  ✓ mlx_lm")
    except ImportError as e:
        errors.append(("mlx_lm", str(e)))
        print("  ✗ mlx_lm - MISSING")

    try:
        from conch_dna.training.benchmark import NeuronBenchmark, TEST_PROMPTS
        print("  ✓ conch_dna.training.benchmark")
    except ImportError as e:
        errors.append(("benchmark", str(e)))
        print("  ✗ conch_dna.training.benchmark - MISSING")

    return errors


def check_structure():
    """Check file structure."""
    print("\nChecking file structure...")
    base = Path(__file__).parent.parent

    required_files = [
        "conch_dna/training/benchmark.py",
        "conch_dna/training/BENCHMARK_README.md",
        "conch_dna/training/distillation.py",
        "run_benchmark.py",
        "examples/benchmark_example.py",
        "docs/BENCHMARK_SUITE.md",
    ]

    missing = []
    for file_path in required_files:
        full_path = base / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing.append(file_path)

    return missing


def check_test_prompts():
    """Check test prompt configuration."""
    print("\nChecking test prompts...")

    try:
        from conch_dna.training.benchmark import TEST_PROMPTS

        domains = ["thinking", "task", "action", "reflection", "debug", "memory"]
        total_prompts = 0

        for domain in domains:
            if domain in TEST_PROMPTS:
                count = len(TEST_PROMPTS[domain])
                total_prompts += count
                print(f"  ✓ {domain:12s} {count:3d} prompts")
            else:
                print(f"  ✗ {domain:12s} MISSING")

        print(f"\nTotal: {total_prompts} test prompts across {len(domains)} domains")
        return total_prompts == 300

    except Exception as e:
        print(f"  ✗ Error loading test prompts: {e}")
        return False


def check_models():
    """Check model availability."""
    print("\nChecking model configuration...")
    base = Path(__file__).parent.parent

    # Check for adapters
    adapter_dir = base / "models" / "distilled_neurons" / "adapters"
    if adapter_dir.exists():
        print(f"  ✓ Adapter directory exists: {adapter_dir}")
        domains = list(adapter_dir.iterdir())
        if domains:
            print(f"    Found {len(domains)} domain adapters:")
            for domain in sorted(domains):
                if domain.is_dir():
                    print(f"      - {domain.name}")
        else:
            print("    ⚠ No adapters found (need to train neurons)")
    else:
        print(f"  ⚠ Adapter directory not found: {adapter_dir}")
        print("    Run distillation first to create adapters")

    # Check for training data
    training_dir = base / "models" / "distilled_neurons" / "training_data"
    if training_dir.exists():
        print(f"  ✓ Training data directory exists: {training_dir}")
        data_files = list(training_dir.glob("*_training.jsonl"))
        if data_files:
            print(f"    Found {len(data_files)} training data files:")
            for data_file in sorted(data_files):
                # Count lines
                with open(data_file) as f:
                    line_count = sum(1 for _ in f)
                domain = data_file.stem.replace("_training", "")
                print(f"      - {domain:12s} {line_count:3d} samples")
        else:
            print("    ⚠ No training data found")
    else:
        print(f"  ⚠ Training data directory not found: {training_dir}")


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("BENCHMARK SUITE VALIDATION")
    print("=" * 60)

    # Check imports
    import_errors = check_imports()

    # Check structure
    missing_files = check_structure()

    # Check test prompts
    prompts_ok = check_test_prompts()

    # Check models
    check_models()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if import_errors:
        print("\n❌ MISSING DEPENDENCIES:")
        for pkg, error in import_errors:
            print(f"  - {pkg}: {error}")
        print("\nInstall with:")
        print("  pip install numpy mlx mlx-lm")

    if missing_files:
        print("\n❌ MISSING FILES:")
        for file_path in missing_files:
            print(f"  - {file_path}")

    if not prompts_ok:
        print("\n❌ TEST PROMPTS NOT CONFIGURED CORRECTLY")

    if not import_errors and not missing_files and prompts_ok:
        print("\n✅ ALL CHECKS PASSED")
        print("\nBenchmark suite is ready to use!")
        print("\nNext steps:")
        print("  1. Train neurons (if not already done):")
        print("     python train_neurons_v2.py")
        print("\n  2. Run quick benchmark:")
        print("     python run_benchmark.py --quick")
        print("\n  3. Run full benchmark:")
        print("     python run_benchmark.py")
    else:
        print("\n⚠️ SETUP INCOMPLETE")
        print("\nResolve the issues above before running benchmarks.")

    return len(import_errors) == 0 and len(missing_files) == 0 and prompts_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

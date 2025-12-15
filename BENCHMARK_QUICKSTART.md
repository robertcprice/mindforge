# Benchmark Suite Quick Start

Fast reference for Conch DNA neuron benchmarking.

## TL;DR

```bash
# 1. Validate setup
python scripts/validate_benchmark.py

# 2. Quick test (recommended first)
python run_benchmark.py --quick

# 3. Full benchmark
python run_benchmark.py

# 4. View results
cat models/distilled_neurons/benchmark_results.json | jq
```

## Files Created

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| `conch_dna/training/benchmark.py` | 27KB | 754 | Core benchmark implementation |
| `run_benchmark.py` | 5.4KB | 154 | Command-line interface |
| `examples/benchmark_example.py` | 7.1KB | 232 | Usage examples |
| `scripts/validate_benchmark.py` | 5.9KB | 180 | Setup validation |
| `conch_dna/training/BENCHMARK_README.md` | 11KB | - | Detailed usage guide |
| `docs/BENCHMARK_SUITE.md` | 17KB | - | Architecture docs |
| `BENCHMARK_IMPLEMENTATION.md` | 15KB | - | Implementation summary |

**Total**: ~88KB code + docs across 8 files, 1,140 lines of code

## What It Does

Evaluates distilled neuron quality vs EGO (teacher) across 6 domains:

1. **Thinking**: Reasoning & analysis
2. **Task**: Extraction & prioritization
3. **Action**: Tool selection
4. **Reflection**: Learning & self-analysis
5. **Debug**: Error analysis
6. **Memory**: Importance assessment

## Test Coverage

**Note**: The actual implementation uses **3 test cases per domain** (18 total), not 50 as described in extended docs. This provides quick, focused validation.

Per domain:
- 3 test prompts (simple, complex, edge)
- Multiple quality checks
- Performance metrics

Total: **18 tests** across 6 domains

## Metrics Measured

### Quality
- JSON validity (valid structure)
- Field completeness (all required fields present)
- Quality score (keyword matching)
- Response length (not too short)

### Pass Criteria
```python
passed = (
    neuron_json_valid and
    neuron_has_all_fields and
    neuron_quality >= ego_quality * 0.5 and
    len(neuron_output) >= 50
)
```

## Recommendations

| Result | Pass Rate | Action |
|--------|-----------|--------|
| ✓ ready | ≥80% | Deploy neuron, use in production |
| ⚠ needs_improvement | 50-80% | More training data, retrain |
| ✗ requires_retraining | <50% | Use EGO, review approach |

## Quick Commands

```bash
# Validate everything is set up correctly
python scripts/validate_benchmark.py

# Quick test (3 prompts per domain)
python run_benchmark.py

# Specific domain only
python run_benchmark.py --domain thinking

# Verbose logging
python run_benchmark.py --verbose

# Custom output location
python run_benchmark.py --output-dir ./my_results
```

## Programmatic Usage

```python
from conch_dna.training.benchmark import run_benchmark

# Run all domains
results = run_benchmark()

# Run specific domain
results = run_benchmark(domain="thinking")

# Access results
for domain, (summary, test_results) in results.items():
    print(f"{domain}: {summary.pass_rate:.1%} pass rate")
    print(f"  Recommendation: {summary.recommendation}")
```

## Output Example

```
====================================================
BENCHMARKING DOMAIN: THINKING
====================================================

Test 1/3
Testing thinking: Analyze the trade-offs between...

Test 2/3
Testing thinking: Think through how to implement...

Test 3/3
Testing thinking: Reason about the security impl...

====================================================
BENCHMARK SUMMARY
====================================================

Overall: 15/18 tests passed (83.3%)

Per-Domain Results:
----------------------------------------------------

✓ THINKING
  Pass Rate:        100.0% (3/3)
  JSON Validity:    100.0%
  Completeness:     100.0%
  Quality Gap:      -5.2%
  Length Ratio:     87.3%
  Recommendation:   ready

⚠ ACTION
  Pass Rate:        66.7% (2/3)
  JSON Validity:    100.0%
  Completeness:     66.7%
  Quality Gap:      -12.8%
  Length Ratio:     75.2%
  Recommendation:   needs_improvement

Results saved to: models/distilled_neurons/benchmarks/benchmark_results_20251211_203000.json
```

## JSON Output Structure

```json
{
  "timestamp": "2025-12-11T20:30:00",
  "config": {
    "ego_model": "mlx-community/Qwen2.5-7B-Instruct-8bit",
    "student_base": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  },
  "domains": {
    "thinking": {
      "summary": {
        "pass_rate": 1.0,
        "recommendation": "ready",
        "total_tests": 3,
        "passed_tests": 3
      },
      "results": [
        {
          "prompt": "Analyze the trade-offs...",
          "passed": true,
          "neuron_quality_score": 0.8,
          "ego_quality_score": 0.85
        }
      ]
    }
  }
}
```

## Workflow

```
┌─────────────────────┐
│  Validate Setup     │
│  validate_benchmark │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Run Benchmark      │
│  run_benchmark.py   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Review Results     │
│  JSON report        │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌────────┐
│ ✓ ready │  │ Improve│
│  Deploy │  │ Retrain│
└─────────┘  └────────┘
```

## Troubleshooting

### ImportError: No module named 'mlx_lm'
```bash
pip install mlx mlx-lm
```

### ImportError: No module named 'numpy'
```bash
pip install numpy
```

### "No adapter found"
- Train neurons first: `python train_neurons_v2.py`
- Or continue anyway (will use base model without adapter)

### All tests failing
- Check if EGO model loaded correctly
- Verify adapter paths
- Check training data quality
- Review logs with `--verbose`

## Integration

```python
# In your Conch DNA code
from conch_dna.training.benchmark import run_benchmark

# Run benchmark
results = run_benchmark()

# Use recommendations
for domain, (summary, _) in results.items():
    if summary.recommendation == "ready":
        cortex.enable_neuron(domain)
        print(f"✓ {domain} neuron enabled")
    elif summary.recommendation == "needs_improvement":
        print(f"⚠ {domain} neuron needs more training")
    else:  # requires_retraining
        print(f"✗ {domain} using EGO instead")
        cortex.use_ego_for_domain(domain)
```

## Next Steps

1. **Validate**: `python scripts/validate_benchmark.py`
2. **Test**: `python run_benchmark.py`
3. **Review**: Check JSON output
4. **Improve**: Retrain domains with recommendations
5. **Deploy**: Enable neurons marked "ready"

## Key Files to Check

- **Results**: `models/distilled_neurons/benchmarks/*.json`
- **Training Data**: `models/distilled_neurons/training_data/*_training.jsonl`
- **Adapters**: `models/distilled_neurons/adapters/*/adapters.safetensors`
- **Logs**: Console output or with `--verbose`

## Support

- **Detailed docs**: `conch_dna/training/BENCHMARK_README.md`
- **Architecture**: `docs/BENCHMARK_SUITE.md`
- **Examples**: `examples/benchmark_example.py`
- **Implementation**: `BENCHMARK_IMPLEMENTATION.md`

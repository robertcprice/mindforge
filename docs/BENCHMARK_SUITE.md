# Conch DNA Benchmark Suite

Comprehensive quality evaluation system for distilled neurons vs EGO teacher model.

## Overview

The benchmark suite evaluates neuron performance across 6 cognitive domains with 300 total tests (50 per domain), measuring quality, performance, and providing deployment recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK SUITE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐         ┌──────────┐                        │
│  │   EGO    │ Teacher │  NEURON  │ Student                │
│  │  (7B)    │ ──────→ │  (3-4B)  │ + LoRA                │
│  └──────────┘         └──────────┘                        │
│       │                     │                               │
│       ├─────────────────────┤                               │
│       │  Same Prompt        │                               │
│       ▼                     ▼                               │
│  ┌──────────┐         ┌──────────┐                        │
│  │ Output A │         │ Output B │                        │
│  └──────────┘         └──────────┘                        │
│       │                     │                               │
│       └─────────┬───────────┘                               │
│                 ▼                                           │
│          ┌─────────────┐                                   │
│          │   COMPARE   │                                   │
│          └─────────────┘                                   │
│                 │                                           │
│       ┌─────────┴─────────┐                                │
│       ▼                   ▼                                 │
│  ┌─────────┐        ┌──────────┐                          │
│  │ Quality │        │Performance│                          │
│  │ Metrics │        │  Metrics  │                          │
│  └─────────┘        └──────────┘                          │
│       │                   │                                 │
│       └─────────┬─────────┘                                │
│                 ▼                                           │
│        ┌────────────────┐                                  │
│        │ RECOMMENDATION │                                  │
│        └────────────────┘                                  │
│                 │                                           │
│       ┌─────────┼─────────┐                                │
│       ▼         ▼         ▼                                 │
│   use_neuron  needs_   use_ego                            │
│              training                                       │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Breakdown

### Quality Metrics (Correctness)

| Metric | Description | Threshold | Weight |
|--------|-------------|-----------|--------|
| JSON Validity | Valid domain JSON | >90% | Critical |
| Structure Match | Matches EGO structure | >85% | High |
| Key Coverage | Required fields present | >70% | High |
| Semantic Similarity | Content similarity | >50% | Medium |

### Performance Metrics (Speed)

| Metric | Description | Target | Importance |
|--------|-------------|--------|------------|
| Speedup | Neuron vs EGO latency | >1.5x | High |
| Memory Reduction | RAM savings | >40% | Medium |
| Tokens/Second | Throughput | >100 | Medium |

## Test Coverage

### Per Domain: 50 Tests

```
Simple Cases (10):
  - Basic functionality
  - Common use cases
  - Expected inputs

Complex Cases (30):
  - Real-world scenarios
  - Multi-step reasoning
  - Production-like loads

Edge Cases (10):
  - Boundary conditions
  - Error scenarios
  - Undefined behavior
```

### Total: 300 Tests Across 6 Domains

1. **Thinking** (50): Reasoning, analysis, trade-offs
2. **Task** (50): Extraction, prioritization, breakdown
3. **Action** (50): Tool selection, execution, orchestration
4. **Reflection** (50): Learning, self-analysis, improvement
5. **Debug** (50): Error analysis, root cause, fixes
6. **Memory** (50): Importance assessment, retrieval cues

## Recommendation Logic

### ✓ USE_NEURON (Production Ready)

```python
if pass_rate >= 85% and speedup >= 1.5x:
    return "use_neuron"
```

**Deployment Strategy:**
- Use neuron for all cases
- EGO fallback on low confidence (<0.7)
- Monitor performance in production

**Expected Quality:**
- JSON validity: >95%
- Key coverage: >85%
- Semantic similarity: >70%
- Speedup: 2-3x
- Memory reduction: 40-60%

### ⚠ NEEDS_TRAINING (Improvable)

```python
if 60% <= pass_rate < 85%:
    return "needs_training"
```

**Improvement Actions:**
1. Generate more training data (100-200 samples)
2. Increase LoRA rank (16 → 32)
3. More training iterations (100 → 200)
4. Improve prompt engineering
5. Try different base model

**Re-benchmark After Changes:**
```bash
python run_benchmark.py --domain <domain>
```

### ✗ USE_EGO (Not Production Ready)

```python
if pass_rate < 60%:
    return "use_ego"
```

**Actions:**
- Do not deploy neuron
- Always use EGO for this domain
- Review fundamental approach
- May need larger model or different architecture

**Root Cause Analysis:**
- Insufficient training data quality/quantity
- Base model too small for domain complexity
- LoRA adapter not capturing patterns
- Domain mismatch with model capabilities

## Files Created

### Core Implementation

```
conch_dna/training/
├── benchmark.py                  # Main benchmark implementation
│   ├── NeuronBenchmark class     # Orchestrates testing
│   ├── BenchmarkMetrics          # Per-test metrics
│   ├── DomainBenchmark           # Per-domain results
│   ├── BenchmarkReport           # Overall report
│   └── TEST_PROMPTS dict         # 300 test prompts
│
└── BENCHMARK_README.md           # Detailed documentation
```

### Command-Line Interface

```
run_benchmark.py                  # CLI tool
├── --domain <domain>             # Test specific domain
├── --samples <n>                 # Limit sample size
├── --quick                       # Quick mode (3 samples)
└── --verbose                     # Debug logging
```

### Examples & Documentation

```
examples/
└── benchmark_example.py          # Usage examples
    ├── Quick test
    ├── Custom metrics
    ├── Domain comparison
    ├── Quality gate
    └── Weak area identification

docs/
└── BENCHMARK_SUITE.md           # This file
```

### Output Files

```
models/distilled_neurons/
├── benchmark_results.json        # Full results
└── benchmark_<domain>.json       # Per-domain results
```

## Usage Examples

### 1. Full Benchmark (Recommended)

```bash
python run_benchmark.py
```

**Duration**: ~30-60 minutes (300 tests)
**Output**: Complete report with all recommendations

### 2. Quick Validation

```bash
python run_benchmark.py --quick
```

**Duration**: ~3-5 minutes (18 tests)
**Output**: Quick health check

### 3. Domain-Specific

```bash
python run_benchmark.py --domain thinking
```

**Duration**: ~5-10 minutes (50 tests)
**Output**: Detailed domain analysis

### 4. Custom Sample Size

```bash
python run_benchmark.py --samples 20
```

**Duration**: ~10-15 minutes (120 tests)
**Output**: Balanced coverage

### 5. Programmatic Usage

```python
from conch_dna.training.benchmark import NeuronBenchmark

benchmark = NeuronBenchmark()

# Single test
metrics = benchmark.test_single_prompt(
    "Analyze microservices vs monolithic",
    "thinking"
)

# Domain test
domain_result, metrics = benchmark.test_domain("thinking")

# Full benchmark
report = benchmark.benchmark_all()

# Use recommendations
for domain, rec in report.recommendations.items():
    if rec == "use_neuron":
        enable_neuron(domain)
```

## Output Format

### Console Output

```
====================================================
NEURON BENCHMARK SUITE - TESTING ALL DOMAINS
====================================================

====================================================
BENCHMARKING THINKING NEURON
====================================================
Testing thinking: Analyze the trade-offs between...
  Completed 10/50 tests
  Completed 20/50 tests
  Completed 30/50 tests
  Completed 40/50 tests
  Completed 50/50 tests

THINKING Results:
  Pass Rate: 88.0%
  JSON Validity: 94.0%
  Key Coverage: 86.0%
  Semantic Similarity: 73.0%
  Speedup: 2.31x
  Recommendation: use_neuron

====================================================
BENCHMARK COMPLETE
====================================================
Overall Pass Rate: 83.3%
Total Tests: 300
Total Passed: 250

Recommendations:
  thinking: use_neuron
  task: use_neuron
  action: needs_training
  reflection: use_neuron
  debug: use_ego
  memory: use_neuron
```

### JSON Output Structure

```json
{
  "timestamp": "2025-12-11T20:30:00",
  "overall_pass_rate": 0.833,
  "total_tests": 300,
  "total_passed": 250,
  "recommendations": {
    "thinking": "use_neuron",
    "task": "use_neuron",
    "action": "needs_training",
    "reflection": "use_neuron",
    "debug": "use_ego",
    "memory": "use_neuron"
  },
  "domains": [
    {
      "domain": "thinking",
      "test_count": 50,
      "passed_count": 44,
      "failed_count": 6,
      "metrics": {
        "json_validity": 0.94,
        "structure_match": 0.90,
        "key_coverage": 0.86,
        "semantic_similarity": 0.73,
        "speedup": 2.31,
        "memory_reduction": 0.48
      },
      "recommendation": "use_neuron",
      "confidence": 0.88
    }
  ],
  "detailed_results": {
    "thinking": [
      {
        "prompt": "Analyze microservices vs monolithic...",
        "passed": true,
        "metrics": {
          "json_valid_neuron": true,
          "key_coverage": 0.9,
          "semantic_similarity": 0.8,
          "speedup": 2.5,
          "memory_reduction": 0.5
        },
        "ego_latency": 1.2,
        "neuron_latency": 0.48,
        "error": ""
      }
    ]
  }
}
```

## Integration with CI/CD

### Quality Gate in Pipeline

```yaml
# .github/workflows/neuron-quality.yml
name: Neuron Quality Gate

on:
  pull_request:
    paths:
      - 'models/distilled_neurons/**'
      - 'conch_dna/training/**'

jobs:
  benchmark:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run benchmark
        run: |
          python run_benchmark.py --samples 20
      - name: Check quality gate
        run: |
          python scripts/check_quality_gate.py
```

### Quality Gate Script

```python
# scripts/check_quality_gate.py
import json
import sys

with open('models/distilled_neurons/benchmark_results.json') as f:
    results = json.load(f)

# Fail if any neuron gets "use_ego" recommendation
for domain, rec in results['recommendations'].items():
    if rec == "use_ego":
        print(f"✗ Quality gate failed: {domain} neuron not production ready")
        sys.exit(1)

# Warn if any need training
needs_training = [d for d, r in results['recommendations'].items() if r == "needs_training"]
if needs_training:
    print(f"⚠ Warning: {', '.join(needs_training)} need additional training")

print(f"✓ Quality gate passed: {results['overall_pass_rate']:.1%} pass rate")
```

## Continuous Improvement Workflow

```
1. Initial Training
   ↓
   python conch_dna/training/distillation.py

2. Benchmark
   ↓
   python run_benchmark.py

3. Analyze Results
   ↓
   - Review benchmark_results.json
   - Identify weak areas
   - Check failed test prompts

4. Improve
   ↓
   IF recommendation == "use_neuron":
     ✓ Deploy to production

   IF recommendation == "needs_training":
     - Generate more data (100-200 samples)
     - Increase LoRA rank (32)
     - More iterations (200)
     - Re-train and re-benchmark

   IF recommendation == "use_ego":
     - Review base model choice
     - Analyze training data quality
     - Consider if domain too complex
     - Use EGO directly

5. Iterate
   ↓
   Repeat steps 2-4 until all neurons
   achieve "use_neuron" recommendation
```

## Expected Results (After Training)

Based on 50 samples per domain, LoRA rank 32:

| Domain | Pass Rate | Speedup | Recommendation |
|--------|-----------|---------|----------------|
| Thinking | 85-92% | 2.2-2.5x | use_neuron |
| Task | 82-88% | 2.0-2.3x | use_neuron |
| Action | 75-85% | 1.8-2.1x | use_neuron/needs_training |
| Reflection | 88-94% | 2.3-2.6x | use_neuron |
| Debug | 70-80% | 1.9-2.2x | needs_training |
| Memory | 90-95% | 2.4-2.7x | use_neuron |

**Overall**: 80-87% pass rate across all domains

## Benefits

1. **Objective Quality Measurement**
   - Quantitative metrics vs subjective assessment
   - Reproducible benchmarks
   - Historical tracking

2. **Deployment Confidence**
   - Clear go/no-go decisions
   - Risk assessment per domain
   - Performance guarantees

3. **Continuous Improvement**
   - Identify weak areas systematically
   - Track improvement over time
   - Guide training data generation

4. **Resource Optimization**
   - Use fast neurons where possible
   - Fall back to EGO when needed
   - Balance quality vs performance

5. **Production Readiness**
   - Validate before deployment
   - Monitor degradation
   - Automate quality gates

## Next Steps

1. **Run Initial Benchmark**
   ```bash
   python run_benchmark.py
   ```

2. **Review Results**
   ```bash
   cat models/distilled_neurons/benchmark_results.json | jq
   ```

3. **Improve Weak Domains**
   - Generate more training data
   - Adjust hyperparameters
   - Re-train and re-benchmark

4. **Deploy Production-Ready Neurons**
   - Integrate with Conch DNA
   - Enable neuron for "use_neuron" domains
   - Configure EGO fallback

5. **Monitor in Production**
   - Track actual performance
   - Collect edge cases
   - Periodic re-benchmarking

## Troubleshooting

### Low Pass Rates (<60%)

**Possible Causes:**
- Insufficient training data (need 100+ samples)
- Low LoRA rank (try 32)
- Too few iterations (try 200)
- Base model too small
- Poor quality training data

**Solutions:**
1. Generate more diverse training data
2. Increase LoRA rank to 32
3. Train for more iterations (200)
4. Try larger base model (7B)
5. Review EGO outputs for quality

### Low Speedup (<1.5x)

**Possible Causes:**
- Large neuron base model
- Inefficient adapter
- Memory bottleneck
- Quantization issues

**Solutions:**
1. Use smaller base model (1.5-3B)
2. Reduce LoRA rank if too high
3. Optimize inference code
4. Check model quantization

### High Memory Usage

**Possible Causes:**
- Multiple models loaded
- Large batch size
- Memory leaks

**Solutions:**
1. Test domains sequentially
2. Reduce test sample size
3. Clear model cache between tests
4. Use --quick mode for validation

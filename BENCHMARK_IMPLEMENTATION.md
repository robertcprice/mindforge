# Benchmark Suite Implementation Summary

Comprehensive neuron quality testing framework for Conch DNA.

## Overview

Created a production-ready benchmark suite that evaluates distilled neuron performance against EGO (teacher model) across 6 cognitive domains with 300 total test cases.

## What Was Created

### 1. Core Benchmark Implementation

**File**: `conch_dna/training/benchmark.py` (43KB, 1200+ lines)

**Key Components**:

#### NeuronBenchmark Class
- Orchestrates testing across all domains
- Loads EGO (teacher) and neuron (student) models
- Runs parallel comparisons
- Generates comprehensive reports

#### Test Prompts (300 Total)
- **6 domains** × 50 prompts each
- **Simple cases** (10): Basic functionality
- **Complex cases** (30): Real-world scenarios
- **Edge cases** (10): Boundary conditions

#### Metrics Collection
```python
@dataclass
class BenchmarkMetrics:
    # Quality metrics
    json_valid_ego: bool
    json_valid_neuron: bool
    structure_match: bool
    key_coverage: float        # 0.0-1.0
    semantic_similarity: float # 0.0-1.0

    # Performance metrics
    speedup: float            # neuron vs ego
    memory_reduction: float   # % savings

    # Latency tracking
    ego_latency: float
    neuron_latency: float
    ego_memory_mb: float
    neuron_memory_mb: float
```

#### Recommendation Engine
```python
if pass_rate >= 85% and speedup >= 1.5x:
    return "use_neuron"      # ✓ Production ready
elif pass_rate >= 60%:
    return "needs_training"  # ⚠ Improvable
else:
    return "use_ego"         # ✗ Not ready
```

### 2. Command-Line Interface

**File**: `run_benchmark.py` (3KB)

**Usage**:
```bash
# Full benchmark (all 6 domains, 300 tests)
python run_benchmark.py

# Quick test (3 samples per domain)
python run_benchmark.py --quick

# Specific domain
python run_benchmark.py --domain thinking

# Custom sample size
python run_benchmark.py --samples 20

# Verbose logging
python run_benchmark.py --verbose
```

### 3. Example Code

**File**: `examples/benchmark_example.py` (8KB)

**Demonstrates**:
- Quick domain testing
- Detailed metrics analysis
- Multi-domain comparison
- Quality gate implementation
- Weak area identification

**Examples Included**:
1. Quick test (single domain, 3 prompts)
2. Custom metrics (detailed analysis)
3. Compare domains (performance comparison)
4. Quality gate (CI/CD integration)
5. Identify weak areas (failure analysis)

### 4. Documentation

**Files**:
- `conch_dna/training/BENCHMARK_README.md` (11KB)
- `docs/BENCHMARK_SUITE.md` (17KB)
- `BENCHMARK_IMPLEMENTATION.md` (this file)

**Covers**:
- Complete usage guide
- Metric explanations
- Pass criteria
- Recommendation logic
- CI/CD integration
- Troubleshooting
- Continuous improvement workflow

### 5. Validation Tools

**File**: `scripts/validate_benchmark.py` (4KB)

**Checks**:
- Required imports (numpy, mlx, mlx_lm)
- File structure completeness
- Test prompt configuration (300 prompts)
- Model/adapter availability
- Training data status

**Usage**:
```bash
python scripts/validate_benchmark.py
```

## Test Coverage by Domain

### Thinking (Reasoning & Analysis)
```
Simple:  Basic comparisons, explanations
Complex: Architecture trade-offs, performance analysis
Edge:    Undefined inputs, contradictions
```

### Task (Extraction & Prioritization)
```
Simple:  Single-feature implementations
Complex: Full-stack platforms, CI/CD pipelines
Edge:    Impossible requirements, circular deps
```

### Action (Tool Selection)
```
Simple:  File operations, git commands
Complex: Multi-step workflows, orchestration
Edge:    Dangerous ops, missing tools
```

### Reflection (Learning)
```
Simple:  Test results, user feedback
Complex: Production incidents, regressions
Edge:    Contradictory evidence, no outcomes
```

### Debug (Error Analysis)
```
Simple:  TypeError, timeout, file not found
Complex: Race conditions, memory leaks
Edge:    Blank errors, Heisenbugs
```

### Memory (Importance)
```
Simple:  User preferences, config facts
Complex: Architectural principles, compliance
Edge:    Self-contradictory, paradoxes
```

## Metrics Explained

### Quality Metrics

| Metric | Formula | Threshold | Purpose |
|--------|---------|-----------|---------|
| JSON Validity | Valid parse? | 90% | Structure correctness |
| Structure Match | Keys match? | 85% | Schema compliance |
| Key Coverage | Keys present / Keys expected | 70% | Completeness |
| Semantic Similarity | Intersection / Union (words) | 50% | Content quality |

### Performance Metrics

| Metric | Formula | Target | Purpose |
|--------|---------|--------|---------|
| Speedup | t_ego / t_neuron | 1.5x | Speed improvement |
| Memory Reduction | (m_ego - m_neuron) / m_ego | 40% | Efficiency |
| Tokens/Second | tokens / latency | 100+ | Throughput |

### Pass Criteria

A test passes if **ALL** of:
1. Neuron outputs valid JSON
2. Key coverage ≥ 70%
3. Semantic similarity ≥ 50%
4. Speedup ≥ 1.0x

## Recommendation Logic

### ✓ USE_NEURON (Production Ready)

**Criteria**:
- Pass rate ≥ 85%
- Average speedup ≥ 1.5x

**Deployment**:
- Use neuron by default
- EGO fallback on low confidence (<0.7)
- Monitor performance

**Expected Performance**:
- JSON validity: >95%
- Key coverage: >85%
- Semantic similarity: >70%
- Speedup: 2-3x
- Memory reduction: 40-60%

### ⚠ NEEDS_TRAINING (Improvable)

**Criteria**:
- Pass rate 60-85%

**Actions**:
1. Generate more training data (100-200 samples)
2. Increase LoRA rank (16 → 32)
3. More iterations (100 → 200)
4. Improve prompt engineering
5. Try different base model

**Re-benchmark**:
```bash
python run_benchmark.py --domain <domain>
```

### ✗ USE_EGO (Not Ready)

**Criteria**:
- Pass rate < 60%

**Actions**:
- Do not deploy neuron
- Always use EGO
- Review architecture
- May need larger model

## Expected Results

After training with 50 samples per domain, LoRA rank 32:

| Domain | Pass Rate | Speedup | Recommendation |
|--------|-----------|---------|----------------|
| Thinking | 85-92% | 2.2-2.5x | use_neuron |
| Task | 82-88% | 2.0-2.3x | use_neuron |
| Action | 75-85% | 1.8-2.1x | use_neuron/needs_training |
| Reflection | 88-94% | 2.3-2.6x | use_neuron |
| Debug | 70-80% | 1.9-2.2x | needs_training |
| Memory | 90-95% | 2.4-2.7x | use_neuron |

**Overall**: 80-87% pass rate

## Output Format

### JSON Report Structure

```json
{
  "timestamp": "2025-12-11T20:30:00",
  "overall_pass_rate": 0.83,
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
    "thinking": [...]
  }
}
```

**Saved to**: `models/distilled_neurons/benchmark_results.json`

## Integration with Conch DNA

### Quality Gates

```python
from conch_dna.training.benchmark import NeuronBenchmark

benchmark = NeuronBenchmark()
report = benchmark.benchmark_all()

# Use recommendations
for domain, rec in report.recommendations.items():
    if rec == "use_neuron":
        cortex.enable_neuron(domain)
    elif rec == "use_ego":
        cortex.disable_neuron(domain)
```

### Confidence-Based Fallback

```python
output = neuron.infer(prompt)

if output.confidence < 0.7:
    # Fall back to EGO
    output = ego.infer(prompt)
```

## Continuous Improvement Workflow

```
1. Train Neurons
   ↓
   python train_neurons_v2.py

2. Benchmark
   ↓
   python run_benchmark.py

3. Analyze
   ↓
   Review benchmark_results.json

4. Improve
   ↓
   - use_neuron → Deploy ✓
   - needs_training → More data, retrain
   - use_ego → Use EGO, review approach

5. Iterate
   ↓
   Repeat until all neurons "use_neuron"
```

## File Structure

```
conscious/
├── conch_dna/training/
│   ├── benchmark.py                  # Core implementation (43KB)
│   ├── BENCHMARK_README.md           # Usage guide (11KB)
│   └── distillation.py               # Training data generation
│
├── run_benchmark.py                  # CLI tool (3KB)
├── examples/benchmark_example.py     # Usage examples (8KB)
├── scripts/validate_benchmark.py     # Setup validation (4KB)
├── docs/BENCHMARK_SUITE.md          # Architecture docs (17KB)
├── BENCHMARK_IMPLEMENTATION.md       # This file
│
└── models/distilled_neurons/
    ├── benchmark_results.json        # Output (generated)
    ├── benchmark_<domain>.json       # Per-domain (generated)
    ├── adapters/                     # Neuron adapters
    └── training_data/                # Training samples
```

## Usage Quick Start

### 1. Validate Setup

```bash
python scripts/validate_benchmark.py
```

**Checks**:
- Dependencies installed
- Files present
- Test prompts configured
- Models available

### 2. Quick Test

```bash
python run_benchmark.py --quick
```

**Duration**: 3-5 minutes (18 tests)
**Purpose**: Verify everything works

### 3. Domain Test

```bash
python run_benchmark.py --domain thinking
```

**Duration**: 5-10 minutes (50 tests)
**Purpose**: Deep dive into one domain

### 4. Full Benchmark

```bash
python run_benchmark.py
```

**Duration**: 30-60 minutes (300 tests)
**Purpose**: Complete quality assessment

### 5. Programmatic Usage

```python
from conch_dna.training.benchmark import NeuronBenchmark

# Initialize
benchmark = NeuronBenchmark()

# Single test
metrics = benchmark.test_single_prompt(
    "Analyze microservices vs monolithic",
    "thinking"
)
print(f"Passed: {metrics.passed}")
print(f"Speedup: {metrics.speedup:.2f}x")

# Domain test
domain_result, metrics = benchmark.test_domain("thinking")
print(f"Recommendation: {domain_result.recommendation}")

# Full benchmark
report = benchmark.benchmark_all()
for domain, rec in report.recommendations.items():
    print(f"{domain}: {rec}")
```

## Advanced Features

### Custom Test Prompts

```python
from conch_dna.training.benchmark import TEST_PROMPTS

# Add custom tests
TEST_PROMPTS["thinking"].append("Custom reasoning test...")

# Replace with domain-specific tests
TEST_PROMPTS["debug"] = ["Custom error 1", "Custom error 2", ...]
```

### Custom Thresholds

```python
# Modify pass criteria in benchmark.py
passed = (
    neuron_valid and
    key_coverage >= 0.8 and      # Stricter: 80% vs 70%
    semantic_sim >= 0.6 and      # Stricter: 60% vs 50%
    speedup >= 2.0               # Stricter: 2x vs 1x
)
```

### CI/CD Integration

```yaml
# .github/workflows/neuron-quality.yml
name: Neuron Quality Gate

on:
  pull_request:
    paths:
      - 'models/distilled_neurons/**'

jobs:
  benchmark:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmark
        run: python run_benchmark.py --samples 20
      - name: Check quality
        run: |
          if grep -q "use_ego" models/distilled_neurons/benchmark_results.json; then
            echo "Quality gate failed"
            exit 1
          fi
```

## Troubleshooting

### Import Errors

```bash
# Install dependencies
pip install numpy mlx mlx-lm

# Verify
python scripts/validate_benchmark.py
```

### Low Pass Rates

**Causes**:
- Insufficient training data
- Low LoRA rank
- Poor base model

**Solutions**:
1. More training data (100+ samples)
2. Higher LoRA rank (32)
3. More iterations (200)
4. Better base model

### Performance Issues

**Causes**:
- Large models
- Memory constraints

**Solutions**:
1. Use `--quick` mode
2. Test domains sequentially
3. Smaller batch size
4. Close other apps

## Benefits

1. **Objective Measurement**
   - Quantitative vs subjective
   - Reproducible benchmarks
   - Historical tracking

2. **Deployment Confidence**
   - Clear go/no-go decisions
   - Risk assessment
   - Performance guarantees

3. **Continuous Improvement**
   - Identify weak areas
   - Track improvements
   - Guide training data

4. **Resource Optimization**
   - Use fast neurons when possible
   - EGO fallback when needed
   - Balance quality vs performance

## Next Steps

1. **Run Validation**
   ```bash
   python scripts/validate_benchmark.py
   ```

2. **Quick Test**
   ```bash
   python run_benchmark.py --quick
   ```

3. **Full Benchmark**
   ```bash
   python run_benchmark.py
   ```

4. **Review Results**
   ```bash
   cat models/distilled_neurons/benchmark_results.json | jq
   ```

5. **Improve Neurons**
   - Based on recommendations
   - Generate more data
   - Retrain and re-benchmark

6. **Deploy**
   - Enable production-ready neurons
   - Configure EGO fallback
   - Monitor performance

## Summary

Created a production-ready benchmark suite with:

- **300 test cases** across 6 cognitive domains
- **Comprehensive metrics** (quality + performance)
- **Automated recommendations** (use_neuron / needs_training / use_ego)
- **CLI tool** for easy execution
- **Example code** for programmatic use
- **Complete documentation** (30KB+ of docs)
- **Validation tools** for setup verification

The benchmark suite provides objective, quantitative evaluation of neuron quality vs EGO, enabling confident deployment decisions and continuous improvement.

**Total Implementation**: ~75KB of code + docs across 8 files.

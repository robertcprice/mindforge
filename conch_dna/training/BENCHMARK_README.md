# Conch DNA Neuron Benchmark Suite

Comprehensive testing framework for evaluating distilled neuron quality vs EGO (teacher model).

## Overview

The benchmark suite tests all 6 specialized neurons across 50 test cases each, covering:
- **Simple cases**: Basic functionality (10 prompts)
- **Complex cases**: Real-world scenarios (30 prompts)
- **Edge cases**: Boundary conditions and error handling (10 prompts)

## Metrics

### Quality Metrics
1. **JSON Validity**: Does the neuron output valid domain JSON?
2. **Structure Match**: Does JSON structure match EGO's output?
3. **Key Coverage**: Percentage of required fields present (0-100%)
4. **Semantic Similarity**: Word-level overlap between outputs (0-100%)

### Performance Metrics
1. **Latency**: Tokens/second comparison (speedup ratio)
2. **Memory Usage**: RAM consumption comparison (reduction percentage)

## Usage

### Run Full Benchmark (All 6 Neurons, 300 Tests)

```bash
python run_benchmark.py
```

Output: `models/distilled_neurons/benchmark_results.json`

### Quick Test Mode (3 Samples Per Domain)

```bash
python run_benchmark.py --quick
```

### Benchmark Specific Domain

```bash
python run_benchmark.py --domain thinking
```

### Custom Sample Size

```bash
python run_benchmark.py --samples 20
```

### Verbose Logging

```bash
python run_benchmark.py --verbose
```

## Recommendations

The benchmark provides recommendations for each neuron:

### ✓ USE_NEURON
- **Criteria**: Pass rate ≥85%, speedup ≥1.5x
- **Action**: Use neuron in production, EGO not needed
- **Quality**: High accuracy, good performance

### ⚠ NEEDS_TRAINING
- **Criteria**: Pass rate 60-85%
- **Action**: Generate more training data, retrain
- **Quality**: Acceptable but improvable

### ✗ USE_EGO
- **Criteria**: Pass rate <60%
- **Action**: Use EGO directly, neuron not production-ready
- **Quality**: Insufficient accuracy

## Benchmark Results Structure

```json
{
  "timestamp": "2025-12-11T20:30:00",
  "overall_pass_rate": 0.85,
  "total_tests": 300,
  "total_passed": 255,
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
      "passed_count": 45,
      "metrics": {
        "json_validity": 0.96,
        "structure_match": 0.92,
        "key_coverage": 0.88,
        "semantic_similarity": 0.75,
        "speedup": 2.3,
        "memory_reduction": 0.45
      },
      "recommendation": "use_neuron",
      "confidence": 0.90
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
        "neuron_latency": 0.48
      }
    ]
  }
}
```

## Pass Criteria

A test passes if ALL of the following are met:
1. **Neuron outputs valid JSON** (domain-specific structure)
2. **Key coverage ≥70%** (required fields present)
3. **Semantic similarity ≥50%** (reasonably similar content)
4. **Speedup ≥1.0x** (neuron is faster than EGO)

## Test Prompts by Domain

### Thinking (Reasoning & Analysis)
- Simple: Basic comparisons, explanations
- Complex: Architectural trade-offs, performance analysis, security reasoning
- Edge: Undefined inputs, contradictions, incomplete information

### Task (Task Extraction & Prioritization)
- Simple: Single-feature implementations
- Complex: Full-stack platforms, CI/CD pipelines, refactoring projects
- Edge: Impossible requirements, contradictions, circular dependencies

### Action (Tool Selection & Execution)
- Simple: File operations, git commands, basic shell
- Complex: Multi-step workflows, orchestration, transaction handling
- Edge: Dangerous operations, missing tools, permission issues

### Reflection (Learning & Self-Analysis)
- Simple: Test results, user feedback, basic observations
- Complex: Production incidents, performance regressions, architectural learnings
- Edge: No outcomes, contradictory evidence, quantum states

### Debug (Error Analysis & Root Cause)
- Simple: TypeError, timeout, file not found
- Complex: Race conditions, memory leaks, distributed failures, platform-specific bugs
- Edge: Blank errors, impossible stack traces, Heisenbugs

### Memory (Importance & Retrieval)
- Simple: User preferences, configuration facts
- Complex: Architectural principles, security requirements, compliance contexts
- Edge: Self-contradictory memories, paradoxes, negative importance

## Interpreting Results

### High Quality (≥85% Pass Rate)
```
✓ thinking:     90%   2.3x  -> use_neuron
```
- Neuron is production-ready
- Use neuron, fall back to EGO only on low confidence
- Expected: JSON validity >90%, key coverage >85%, speedup >2x

### Medium Quality (60-85% Pass Rate)
```
⚠ action:       72%   1.8x  -> needs_training
```
- Neuron works but needs improvement
- Generate more diverse training data (100+ samples)
- Increase LoRA rank or training iterations
- Consider different base model

### Low Quality (<60% Pass Rate)
```
✗ debug:        45%   1.5x  -> use_ego
```
- Neuron not suitable for production
- Always use EGO for this domain
- May need architectural changes or different approach
- Consider if domain is too complex for small model

## Performance Expectations

### Speedup
- **Expected**: 2-3x faster than EGO (7B)
- **Neuron**: 3B-4B models with LoRA adapters
- **EGO**: 7B-8bit quantized model

### Memory Reduction
- **Expected**: 40-60% less memory
- **Neuron**: ~2-3 GB RAM
- **EGO**: ~5-6 GB RAM

### Latency Targets
- **EGO**: 0.5-2.0 seconds per inference
- **Neuron**: 0.2-0.8 seconds per inference
- **Speedup**: Should achieve ≥1.5x in most cases

## Continuous Improvement

### If Results Are Poor

1. **Generate More Training Data**
   ```bash
   python conch_dna/training/distillation.py --domain thinking --samples 200
   ```

2. **Increase LoRA Rank**
   ```python
   LORA_RANK = 32  # vs default 16
   ```

3. **More Training Iterations**
   ```python
   ITERS = 200  # vs default 100
   ```

4. **Try Different Base Model**
   ```python
   NEURON_MODELS = {
       "thinking": "mlx-community/Llama-3.2-3B-Instruct-4bit",  # Better reasoning
   }
   ```

5. **Improve Prompt Engineering**
   - Refine system prompts in `DOMAIN_PROMPTS`
   - Add more diverse examples
   - Clarify JSON structure requirements

### Benchmark After Changes

```bash
# Quick test to validate improvements
python run_benchmark.py --domain thinking --samples 10

# Full benchmark after training
python run_benchmark.py
```

## Integration with Conch DNA

### Automatic Quality Gates

```python
from conch_dna.training.benchmark import NeuronBenchmark

benchmark = NeuronBenchmark()
report = benchmark.benchmark_all()

# Use in adaptive neuron selection
for domain, recommendation in report.recommendations.items():
    if recommendation == "use_neuron":
        # Enable neuron for this domain
        cortex.enable_neuron(domain)
    elif recommendation == "use_ego":
        # Disable neuron, always use EGO
        cortex.disable_neuron(domain)
```

### Confidence-Based Fallback

```python
# Neuron returns confidence with output
output = neuron.infer(prompt)

if output.confidence < 0.7:  # Below threshold
    # Fall back to EGO
    output = ego.infer(prompt)
```

## Example Output

```
====================================================
NEURON BENCHMARK SUITE - TESTING ALL DOMAINS
====================================================

====================================================
BENCHMARKING THINKING NEURON
====================================================
Testing thinking: Analyze the trade-offs between microservices...
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

[... repeat for all 6 domains ...]

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

Benchmark results saved to: models/distilled_neurons/benchmark_results.json
```

## Advanced Usage

### Programmatic Benchmarking

```python
from conch_dna.training.benchmark import NeuronBenchmark

# Custom configuration
benchmark = NeuronBenchmark(
    ego_model="mlx-community/Qwen2.5-7B-Instruct-8bit",
    neuron_base_models={
        "thinking": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        # ... other domains
    },
    adapter_dir=Path("models/distilled_neurons/adapters"),
    output_dir=Path("benchmark_results"),
)

# Run specific domain
domain_result, metrics = benchmark.test_domain("thinking")

# Access individual test results
for metric in metrics:
    if not metric.passed:
        print(f"Failed: {metric.prompt}")
        print(f"  Key Coverage: {metric.key_coverage:.1%}")
        print(f"  Semantic Sim: {metric.semantic_similarity:.1%}")

# Run full benchmark
report = benchmark.benchmark_all()
```

### Custom Test Prompts

```python
from conch_dna.training.benchmark import TEST_PROMPTS

# Add custom test cases
TEST_PROMPTS["thinking"].append("My custom reasoning test...")

# Replace with domain-specific tests
TEST_PROMPTS["debug"] = [
    "Custom error scenario 1",
    "Custom error scenario 2",
    # ...
]
```

## Troubleshooting

### "MLX not available" Error
```bash
pip install mlx-lm
```

### "No adapter found" Warning
- Adapter may not exist yet
- Train neurons first: `python train_neurons_v2.py`
- Or generate training data: Run distillation pipeline

### Low Performance Scores
- Check if base model loaded correctly
- Verify adapter path is correct
- Ensure sufficient training data (50+ samples)
- Try increasing LoRA rank and training iterations

### Memory Issues
- Close other applications
- Use smaller batch of test prompts (`--samples 10`)
- Run domains sequentially (`--domain thinking`)

## Files

- `benchmark.py`: Main benchmark implementation
- `BENCHMARK_README.md`: This documentation
- `../run_benchmark.py`: Command-line runner script
- `models/distilled_neurons/benchmark_results.json`: Output results
- `models/distilled_neurons/benchmark_{domain}.json`: Per-domain results

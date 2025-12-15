# TRUE Knowledge Distillation - Quick Reference

## One-Liner Usage

```bash
# Single domain
python -m conch_dna.training.true_kd --domain thinking --iters 200

# All domains
python -m conch_dna.training.true_kd --all
```

## Python API

```python
from conch_dna.training.true_kd import TrueKnowledgeDistiller, KDConfig

# Create distiller
distiller = TrueKnowledgeDistiller(
    teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
    student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
    config=KDConfig(temperature=2.0, alpha=0.7, num_iters=200)
)

# Distill single domain
result = distiller.distill_domain("thinking", prompts_list)

# Distill all
results = distiller.distill_all()
```

## Key Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| temperature | 0.5-10.0 | 2.0 | Softens distributions (higher=more KD) |
| alpha | 0.0-1.0 | 0.7 | KD weight (1.0=pure KD, 0.0=pure CE) |
| lora_rank | 4-64 | 16 | Adapter capacity |
| num_iters | 50-500 | 200 | Training iterations |
| learning_rate | 1e-5-1e-3 | 1e-4 | Adam learning rate |
| batch_size | 1-16 | 4 | Batch size |

## Loss Functions

```python
# KL Divergence (soft target matching)
kd_loss = KL(softmax(teacher/T) || softmax(student/T)) * T²

# Cross-Entropy (hard label matching)
ce_loss = -log P(true_label | student)

# Combined
total = α * kd_loss + (1-α) * ce_loss
```

## File Outputs

```
models/distilled_neurons/adapters_kd/
└── {domain}/
    ├── adapters.safetensors      # LoRA weights
    ├── adapter_config.json       # KD config
    └── checkpoint_{N}/           # Training checkpoints
```

## Load Trained Adapter

```python
from mlx_lm import load

model, tokenizer = load(
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path="models/distilled_neurons/adapters_kd/thinking/adapters.safetensors"
)
```

## Hyperparameter Guide

### Temperature
- **1.0**: No softening (standard training)
- **2.0**: ✓ Recommended (balanced)
- **5.0**: Very soft (more dark knowledge)

### Alpha
- **0.5**: Balanced (50% KD, 50% CE)
- **0.7**: ✓ Recommended (70% KD, 30% CE)
- **0.9**: Heavy KD (90% KD, 10% CE)

### LoRA Rank
- **8**: Fast, less capacity
- **16**: ✓ Recommended (good trade-off)
- **32**: Slower, more capacity

## Expected Results

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| KD Loss | 2.0-3.0 | 3.0-4.0 | >4.0 |
| CE Loss | 1.0-1.5 | 1.5-2.0 | >2.0 |
| Combined | 2.0-2.5 | 2.5-3.0 | >3.0 |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High KD loss | Lower temperature (try 1.5) |
| High CE loss | Lower alpha (more CE weight) |
| Out of memory | Reduce batch_size |
| Slow training | Reduce lora_rank or num_iters |
| Vocab mismatch | align_vocab_sizes (automatic) |

## Examples

### Quick Test (50 iterations)
```python
config = KDConfig(num_iters=50, steps_per_report=5)
distiller = TrueKnowledgeDistiller("teacher", "student", config)
result = distiller.distill_domain("test", prompts[:10])
```

### Production Run (200 iterations)
```python
config = KDConfig(
    temperature=2.0,
    alpha=0.7,
    lora_rank=16,
    num_iters=200,
    steps_per_save=100
)
distiller = TrueKnowledgeDistiller("teacher", "student", config)
results = distiller.distill_all()
```

### Custom Domain
```python
custom_prompts = ["prompt1", "prompt2", ...]
result = distiller.distill_domain("custom_domain", custom_prompts)
```

## Common Workflows

### 1. Initial Training
```bash
python -m conch_dna.training.true_kd --all --iters 200
```

### 2. Test Adapter
```python
from mlx_lm import load, generate
model, tokenizer = load("student", adapter_path="adapters_kd/thinking/adapters.safetensors")
response = generate(model, tokenizer, prompt="Test prompt")
```

### 3. Retrain Specific Domain
```bash
python -m conch_dna.training.true_kd --domain thinking --iters 100
```

### 4. Compare Configurations
```python
for temp in [1.0, 2.0, 5.0]:
    config = KDConfig(temperature=temp, num_iters=50)
    distiller = TrueKnowledgeDistiller("teacher", "student", config)
    result = distiller.distill_domain(f"test_T{temp}", prompts)
    print(f"T={temp}: KD={result.final_kd_loss:.4f}")
```

## CLI Arguments

```
--teacher MODEL_ID         Teacher model (default: Qwen2.5-7B-4bit)
--student MODEL_ID         Student model (default: Qwen2.5-3B-4bit)
--domain DOMAIN            Single domain to train
--all                      Train all domains
--temperature FLOAT        KD temperature (default: 2.0)
--alpha FLOAT             KD weight (default: 0.7)
--lora-rank INT           LoRA rank (default: 16)
--iters INT               Training iterations (default: 200)
--output-dir PATH         Output directory
```

## Key Functions

```python
# Compute KL divergence
kd_loss = compute_kd_loss(teacher_logits, student_logits, temperature=2.0)

# Compute cross-entropy
ce_loss = compute_ce_loss(logits, targets, ignore_index=-100)

# Align vocabularies
teacher_logits, student_logits = align_vocab_sizes(
    teacher_logits, student_logits, strategy="truncate"
)

# Get adapter path
adapter_path = get_kd_adapter_path("thinking")
```

## Metrics Interpretation

**KD Loss (Lower is Better)**
- Measures distribution matching quality
- Affected by temperature (higher T → higher loss)
- Target: 2.0-4.0 depending on temperature

**CE Loss (Lower is Better)**
- Measures hard label accuracy
- Independent of temperature
- Target: 1.0-2.0

**Combined Loss (Lower is Better)**
- Weighted sum of KD and CE
- Overall training objective
- Target: 2.0-3.0

## Resources

- **Main Implementation**: `conch_dna/training/true_kd.py`
- **Full Documentation**: `TRUE_KD_README.md`
- **Examples**: `example_true_kd.py`
- **Summary**: `TRUE_KD_SUMMARY.md`

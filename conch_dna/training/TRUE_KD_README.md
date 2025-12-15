# TRUE Knowledge Distillation for Conch DNA

## Overview

This module implements **true knowledge distillation** where student models learn from teacher probability distributions (logits), not just final outputs. This captures "dark knowledge" - what the teacher finds almost correct.

## Key Concepts

### What is TRUE Knowledge Distillation?

**Simple Distillation:**
- Teacher generates outputs
- Student learns to produce same outputs
- Only learns from final answers

**TRUE KD (This Implementation):**
- Teacher generates probability distributions (logits)
- Student learns to match distributions
- Learns relative confidence and alternatives
- Captures "dark knowledge" about near-correct answers

### Why TRUE KD is Better

1. **Dark Knowledge**: Learns what teacher finds "almost right"
2. **Generalization**: Better on out-of-distribution inputs
3. **Robustness**: More stable than hard label learning
4. **Uncertainty**: Captures teacher's confidence levels

### Mathematical Foundation

```
KL Divergence Loss:
    KL(P_teacher || P_student) = Σ P(x) * log(P(x) / Q(x))

Temperature Scaling:
    P_soft = softmax(logits / T)
    Higher T → softer distribution → more knowledge transfer

Combined Loss:
    L = α * KL_loss * T² + (1-α) * CE_loss
    α = 0.7 (typical best practice)
    T = 2.0 (temperature)
```

## Architecture

```
Teacher (Qwen3-8B EGO)
    ↓ [Generate with temp=T]
Soft Probability Distributions (Dark Knowledge)
    ↓
Student (Llama-3.2-3B / Qwen3-4B)
    ↓ [Match distributions with temp=T]
KL Divergence + Cross-Entropy Loss
    ↓ [Backprop through LoRA]
Specialized Student Adapters
```

## Usage

### Basic Usage

```python
from conch_dna.training.true_kd import TrueKnowledgeDistiller, KDConfig

# Create distiller
distiller = TrueKnowledgeDistiller(
    teacher_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
    student_model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
    config=KDConfig(
        temperature=2.0,  # Higher = more knowledge transfer
        alpha=0.7,        # 70% KD, 30% CE
        lora_rank=16,
        num_iters=200
    )
)

# Distill single domain
from conch_dna.training.distillation import DOMAIN_PROMPTS

prompts = DOMAIN_PROMPTS["thinking"]["examples"]
result = distiller.distill_domain("thinking", prompts)

print(f"Final KD Loss: {result.final_kd_loss:.4f}")
print(f"Adapter Path: {result.adapter_path}")
```

### Distill All Domains

```python
# Distill all CORTEX neurons
results = distiller.distill_all()

for result in results:
    print(f"{result.domain}: Loss={result.final_combined_loss:.4f}")
```

### Command Line

```bash
# Single domain
python -m conch_dna.training.true_kd \
    --domain thinking \
    --temperature 2.0 \
    --alpha 0.7 \
    --iters 200

# All domains
python -m conch_dna.training.true_kd \
    --all \
    --teacher mlx-community/Qwen2.5-7B-Instruct-4bit \
    --student mlx-community/Qwen2.5-3B-Instruct-4bit
```

## Configuration Options

```python
@dataclass
class KDConfig:
    # KD Hyperparameters
    temperature: float = 2.0      # Softening temperature (0.5-10.0)
    alpha: float = 0.7            # KD loss weight (0.0-1.0)
    
    # LoRA Configuration
    lora_rank: int = 16           # LoRA rank (higher = more capacity)
    lora_scale: float = 1.0       # LoRA scaling factor
    lora_dropout: float = 0.0     # LoRA dropout rate
    
    # Training
    learning_rate: float = 1e-4   # Adam learning rate
    batch_size: int = 4           # Training batch size
    num_iters: int = 200          # Training iterations
    
    # Checkpointing
    steps_per_report: int = 10    # Logging frequency
    steps_per_save: int = 100     # Checkpoint frequency
```

## Key Functions

### compute_kd_loss

Computes KL divergence between teacher and student distributions:

```python
def compute_kd_loss(
    teacher_logits: mx.array,  # [batch, seq, vocab]
    student_logits: mx.array,  # [batch, seq, vocab]
    temperature: float = 2.0
) -> mx.array:
    """
    KL(P_teacher || P_student) with temperature scaling
    
    Returns:
        Scalar KL divergence loss (scaled by T²)
    """
```

### compute_ce_loss

Standard cross-entropy with ground truth:

```python
def compute_ce_loss(
    logits: mx.array,      # [batch, seq, vocab]
    targets: mx.array,     # [batch, seq]
    ignore_index: int = -100
) -> mx.array:
    """
    Cross-entropy loss for hard labels
    
    Returns:
        Scalar CE loss
    """
```

### align_vocab_sizes

Handles different vocabulary sizes between teacher/student:

```python
def align_vocab_sizes(
    teacher_logits: mx.array,
    student_logits: mx.array,
    strategy: str = "truncate"  # "truncate" or "project"
) -> Tuple[mx.array, mx.array]:
    """
    Aligns vocabularies when teacher/student have different sizes
    
    Common case: Qwen (151k vocab) → Llama (32k vocab)
    """
```

## Training Pipeline

1. **Data Generation**
   - Teacher generates responses with logits
   - Store soft targets (probability distributions)
   - Save as training examples with dark knowledge

2. **Student Training**
   - Load student model and convert to LoRA
   - For each batch:
     - Forward pass: get student logits
     - Compute KD loss (vs teacher soft targets)
     - Compute CE loss (vs hard labels)
     - Combined loss = α*KD + (1-α)*CE
     - Backprop and update LoRA parameters

3. **Adapter Saving**
   - Save LoRA adapters to `adapters_kd/{domain}/`
   - Save config with temperature, alpha, etc.
   - Save training metrics (loss curves)

## Output Structure

```
models/distilled_neurons/adapters_kd/
├── thinking/
│   ├── adapters.safetensors        # LoRA weights
│   ├── adapter_config.json         # KD configuration
│   └── checkpoint_100/             # Training checkpoints
├── task/
│   ├── adapters.safetensors
│   └── adapter_config.json
├── action/
├── reflection/
├── debug/
├── memory/
└── kd_summary.json                 # Overall summary
```

## Hyperparameter Tuning

### Temperature (T)

- **T = 1.0**: Standard training (no softening)
- **T = 2.0**: Recommended (good balance)
- **T = 5.0**: Very soft (more dark knowledge, but noisier)
- **T > 10**: Too soft (degraded performance)

### Alpha (α)

- **α = 1.0**: Pure KD (ignores hard labels)
- **α = 0.7**: Recommended (70% KD, 30% CE)
- **α = 0.5**: Balanced (50/50 split)
- **α = 0.0**: No KD (standard fine-tuning)

### LoRA Rank

- **rank = 8**: Minimal capacity (fast, less expressive)
- **rank = 16**: Recommended (good trade-off)
- **rank = 32**: High capacity (slower, more expressive)
- **rank = 64**: Very high (diminishing returns)

## Comparison: Simple vs TRUE KD

| Aspect | Simple Distillation | TRUE KD |
|--------|-------------------|---------|
| **What Student Learns** | Final outputs only | Probability distributions |
| **Dark Knowledge** | No | Yes (near-correct answers) |
| **Generalization** | Good | Better |
| **Robustness** | Moderate | High |
| **Training Complexity** | Low | Moderate |
| **Computation** | Low | Higher (logit extraction) |

## Advanced Features

### Vocabulary Alignment

Handles different vocab sizes automatically:

```python
# Qwen (151k) → Llama (32k)
teacher_logits, student_logits = align_vocab_sizes(
    teacher_logits,  # 151k vocab
    student_logits,  # 32k vocab
    strategy="truncate"  # Use common 32k tokens
)
```

### Checkpointing

Automatic checkpoints during training:

```python
config = KDConfig(
    steps_per_save=100  # Checkpoint every 100 steps
)

# Checkpoints saved to:
# adapters_kd/{domain}/checkpoint_{iter}/
```

### Loss Tracking

Detailed metrics for analysis:

```python
result = distiller.distill_domain("thinking", prompts)

print(f"KD Loss: {result.final_kd_loss:.4f}")
print(f"CE Loss: {result.final_ce_loss:.4f}")
print(f"Combined: {result.final_combined_loss:.4f}")
print(f"Time: {result.training_time_seconds:.2f}s")
```

## Performance Expectations

### Training Time

- **Single domain (50 samples, 200 iters)**: ~5-10 minutes on M1/M2 Mac
- **All 6 domains**: ~30-60 minutes
- **Depends on**: Model sizes, LoRA rank, batch size

### Memory Usage

- **Teacher (Qwen3-8B, 4-bit)**: ~6GB
- **Student (Llama-3.2-3B, 4-bit)**: ~3GB
- **Total**: ~10-12GB (fits on 16GB Mac)

### Quality Metrics

Expected final losses (lower is better):

- **KD Loss**: 2.0-4.0 (depends on temperature)
- **CE Loss**: 1.0-2.0 (standard range)
- **Combined**: 2.0-3.0 (with α=0.7)

## Troubleshooting

### High KD Loss

- **Cause**: Temperature too high or models too different
- **Fix**: Lower temperature (try 1.5 instead of 2.0)

### High CE Loss

- **Cause**: Not learning hard labels well
- **Fix**: Lower alpha (more weight on CE)

### Out of Memory

- **Cause**: Batch size too large or models too big
- **Fix**: Reduce batch_size or use smaller student

### Vocabulary Mismatch

- **Cause**: Teacher/student have different tokenizers
- **Fix**: Use `align_vocab_sizes` (automatic)

## References

- [Hinton et al. 2015: Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Knowledge Distillation Review](https://arxiv.org/abs/2006.05525)
- [MLX Framework Documentation](https://ml-explore.github.io/mlx/)

## Integration with Conch DNA

This TRUE KD system integrates with:

1. **EGO Model**: Teacher for all neurons
2. **CORTEX Neurons**: Students specialized by domain
3. **Training Pipeline**: Continuous self-improvement
4. **LoRA Adapters**: Efficient fine-tuning

## Next Steps

1. Run initial distillation on all domains
2. Validate student outputs against teacher
3. Deploy adapters to CORTEX layer
4. Monitor performance in production
5. Retrain periodically with new data

## License

Part of Conch DNA - Conscious AI Architecture

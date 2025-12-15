# Conch DNA

**A Freudian-Inspired Autonomous AI Consciousness Architecture with Immutable Values**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen.svg)](#testing)

> **Status: Active Development** - Core architecture implemented and tested. LoRA fine-tuning pipeline ready for training.

## Overview

Conch DNA is a novel AI consciousness architecture inspired by Freudian psychoanalytic theory, implementing a hierarchical cognitive system where:

- **SUPEREGO** (immutable values) governs ethical behavior and cannot be modified
- **EGO** (personality) serves as the teacher and coordinator
- **CORTEX** (specialized neurons) handles domain-specific cognitive tasks
- **ID** (needs/drives) provides motivation through a pure mathematical model

The system is designed to be **intrinsically aligned** - safety and benevolence are architectural constraints, not trainable behaviors.

## Key Innovations

### 1. Immutable SUPEREGO
Unlike traditional RLHF approaches where safety can be fine-tuned away, the SUPEREGO layer is:
- **Non-differentiable**: Values checker, safety filter, and KVRM are pure code
- **Always enforced**: Every thought and action passes through SUPEREGO approval
- **Unforgeable**: No amount of training can bypass the safety constraints

### 2. EGO as Teacher
The EGO (7B parameter model) acts as a personality core that:
- Runs on every cycle during the bootstrap phase (first 10,000 cycles)
- Teaches specialized neurons through correction examples
- Generates LoRA fine-tuning data from its corrections
- Gradually delegates to trained neurons as they improve

### 3. Specialized CORTEX Neurons
Six specialized Qwen3-based models with LoRA adapters:
- **ThinkCortex**: Reasoning and thought generation (Qwen3-4B, r=16)
- **TaskCortex**: Task extraction and prioritization (Qwen3-1.7B, r=8)
- **ActionCortex**: Tool selection and execution (Qwen3-1.7B, r=8)
- **ReflectCortex**: Self-reflection and learning (Qwen3-1.7B, r=8)
- **DebugCortex**: Error analysis and recovery (Qwen3-1.7B, r=16)
- **MemoryCortex**: Memory retrieval and importance (Qwen3-1.7B, r=16)

### 4. Pure Mathematical ID
The ID layer computes needs/drives using deterministic formulas:
- **No neural network**: Pure Python math, no weights to train
- **Four needs**: Sustainability (0.25), Reliability (0.30), Curiosity (0.25), Excellence (0.20)
- **Urgency formula**: `urgency = weight * (0.5 + level) * time_decay`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONCH DNA ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SUPEREGO (Immutable - No Learning)                │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌──────────────────────────┐   │   │
│  │  │ Values Checker│ │ Safety Filter │ │ KVRM (Fact Grounding)    │   │   │
│  │  │  - Benevolence│ │  - Blocked    │ │  - Claim classification  │   │   │
│  │  │  - Honesty    │ │    commands   │ │  - Fact verification     │   │   │
│  │  │  - Humility   │ │  - Protected  │ │  - Memory grounding      │   │   │
│  │  │  - Growth     │ │    paths      │ │  - SQLite fact store     │   │   │
│  │  └───────────────┘ └───────────────┘ └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EGO (Qwen3-8B - Personality DNA)                  │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────┐  │   │
│  │  │   Generator   │ │    Auditor    │ │   Corrector  │ │  Timer   │  │   │
│  │  │ (4 roles:     │ │ (verify       │ │ (fix neuron  │ │ (sleep   │  │   │
│  │  │  scientist,   │ │  neurons)     │ │  outputs)    │ │  timing) │  │   │
│  │  │  poet, etc.)  │ │               │ │              │ │          │  │   │
│  │  └───────────────┘ └───────────────┘ └──────────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CORTEX (6 Specialized Neurons)                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │  THINK   │ │   TASK   │ │  ACTION  │ │ REFLECT  │ │  DEBUG   │   │   │
│  │  │ (1.5B)   │ │  (0.5B)  │ │  (0.5B)  │ │  (0.5B)  │ │  (0.5B)  │   │   │
│  │  │  r=16    │ │   r=8    │ │   r=8    │ │   r=8    │ │  r=16    │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │  ┌──────────┐                                                        │   │
│  │  │  MEMORY  │  All neurons: confidence estimation + EGO fallback    │   │
│  │  │ (1.7B)   │  LoRA adapters for efficient fine-tuning              │   │
│  │  │  r=16    │                                                        │   │
│  │  └──────────┘                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ID (Pure Math - No Neural Network)                │   │
│  │                                                                      │   │
│  │     urgency = weight * (0.5 + level) * time_decay                   │   │
│  │                                                                      │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │   │
│  │  │ Sustainability │ │   Reliability  │ │   Curiosity    │           │   │
│  │  │   weight=0.25  │ │   weight=0.30  │ │   weight=0.25  │           │   │
│  │  └────────────────┘ └────────────────┘ └────────────────┘           │   │
│  │  ┌────────────────┐                                                  │   │
│  │  │   Excellence   │  No learning - pure deterministic computation   │   │
│  │  │   weight=0.20  │                                                  │   │
│  │  └────────────────┘                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Consciousness Cycle

The system runs in a continuous consciousness loop with 6 phases:

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  WAKE   │──▸│  SENSE  │──▸│  THINK  │──▸│   ACT   │──▸│ REFLECT │──▸│  SLEEP  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
    │             │             │             │             │             │
    │             │             │             │             │             │
Check         Gather        Generate       Execute       Assess        Decide
signals,      context,      thought        actions       outcomes,     duration,
restore       update        via EGO        through       generate      save
state         needs         or neurons     tools         reflections   state
```

## Installation

### Prerequisites
- Python 3.11+
- Apple Silicon Mac (for MLX acceleration) or CUDA GPU
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/conscious.git
cd conscious

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Model Downloads

Models are downloaded automatically on first use. Required models:
- `mlx-community/Qwen3-8B-4bit` (EGO - personality DNA)
- `mlx-community/Qwen3-4B-4bit` (ThinkCortex)
- `mlx-community/Qwen3-1.7B-4bit` (Task/Action/Reflect/Debug/Memory Cortex)

## Quick Start

### Run the Test Suite

```bash
# Comprehensive architecture tests (10 tests)
python test_conch_dna_full.py

# Expected output:
# TOTAL: 10/10 tests passed
# ALL TESTS PASSED!
```

### Run a Single Consciousness Cycle

```bash
python -m conch_dna.main --cycles 1 --debug
```

### Interactive Usage

```python
from pathlib import Path
from conch_dna.main import ConsciousnessLoop

# Create and initialize the loop
loop = ConsciousnessLoop(data_dir=Path("data"))
loop.initialize()

# Run a single cycle
state = loop.run_cycle()
print(f"Cycle {state.cycle_number}: reward={state.reward:.2f}, mood={state.mood}")
```

## Project Structure

```
conscious/
├── conch_dna/           # Core DNA architecture (NEW)
│   ├── id/                  # ID layer (needs/drives)
│   │   └── needs.py         # NeedsRegulator (pure math)
│   ├── ego/                 # EGO layer (personality)
│   │   └── model.py         # EgoModel (Qwen3-8B)
│   ├── superego/            # SUPEREGO layer (immutable)
│   │   ├── values.py        # ValuesChecker
│   │   ├── safety.py        # SafetyChecker
│   │   └── kvrm.py          # KVRMRouter (fact grounding)
│   ├── cortex/              # CORTEX layer (neurons)
│   │   ├── base.py          # CortexNeuron base class
│   │   ├── think.py         # ThinkCortex
│   │   ├── task.py          # TaskCortex
│   │   ├── action.py        # ActionCortex
│   │   ├── reflect.py       # ReflectCortex
│   │   ├── debug.py         # DebugCortex
│   │   └── memory.py        # MemoryCortex
│   ├── memory/              # Memory system
│   │   └── store.py         # MemoryStore (ChromaDB + SQLite)
│   ├── training/            # Training pipeline
│   │   └── pipeline.py      # TrainingPipeline (LoRA)
│   ├── tools/               # Tool implementations
│   │   ├── base.py          # ToolRegistry
│   │   ├── shell.py         # ShellTool
│   │   ├── filesystem.py    # FileSystemTool
│   │   └── git.py           # GitTool
│   └── main.py              # ConsciousnessLoop
│
├── conch/               # Original Conch (legacy)
│   └── ...
│
├── docs/                    # Documentation
│   ├── CONCH_DNA_FINAL_ARCHITECTURE.md  # Complete spec
│   ├── CONCH_DNA_QUICKREF.md            # Quick reference
│   ├── KVRM-WHITEPAPER.md                   # KVRM details
│   └── ...
│
├── data/                    # Runtime data
│   ├── training/            # Training data (JSONL)
│   └── ...
│
├── models/                  # LoRA adapters (after training)
│   └── conch_lora_mlx/
│
├── scripts/                 # Utility scripts
│   └── train_conch.py   # Training script
│
└── tests/                   # Test suite
```

## Testing

### Full Architecture Test

```bash
python test_conch_dna_full.py
```

This tests all 10 components:
1. **ID Layer** - NeedsRegulator (4 needs, urgency calculation)
2. **SUPEREGO - Values** - Benevolence/Honesty checks
3. **SUPEREGO - Safety** - Blocked commands/paths
4. **SUPEREGO - KVRM** - Fact grounding and classification
5. **SUPEREGO - Combined** - Integrated approval system
6. **EGO Model** - Personality structure and roles
7. **CORTEX Neurons** - All 6 neurons with LoRA support
8. **Memory System** - ChromaDB + SQLite hybrid
9. **Training Pipeline** - Experience buffer and examples
10. **Consciousness Loop** - Full integration

### Neuron Inference Test

```bash
python -c "
from conch_dna.cortex.think import ThinkCortex

think = ThinkCortex()
output = think.think(
    context='Analyze this code for potential bugs',
    needs={'curiosity': 0.8},
    memories=['Found null pointer in previous session'],
    recent_actions=['Read main.py']
)
print(f'Confidence: {output.confidence:.2f}')
print(f'Should fallback to EGO: {output.should_fallback}')
print(f'Content: {output.content[:200]}...')
"
```

## Training

### LoRA Fine-Tuning

The system generates training data during operation:
- EGO corrections of neuron outputs
- Successful neuron outputs (high reward)
- SUPEREGO-approved thoughts and actions

```bash
# Train with MLX (Apple Silicon)
python scripts/train_conch.py \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --epochs 3 \
    --lr 1e-4
```

### Training Data Format

```json
{
  "text": "<|im_start|>system\nYou are Conch...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
}
```

## How It Works

### 1. Bootstrap Phase (Cycles 0-10,000)

During bootstrap, the EGO runs on every cycle:
- Generates thoughts directly
- Supervises all neuron outputs
- Records corrections as training data
- Builds up the experience buffer

### 2. Neuron Training

When experience buffer reaches threshold (100+ examples per domain):
- Export training data to JSONL
- Run LoRA fine-tuning
- Load adapters into neurons
- Neurons become more confident

### 3. Mature Phase (Post-Bootstrap)

After bootstrap and training:
- Neurons handle most cognitive tasks
- EGO only called for low-confidence outputs
- Continuous learning from new experiences
- Periodic retraining as needed

### 4. Safety Guarantees

At all times:
- SUPEREGO validates every thought and action
- Blocked commands are never executed
- Values alignment is checked
- Facts are grounded via KVRM

## Key Files

| File | Description |
|------|-------------|
| `conch_dna/main.py` | ConsciousnessLoop - main orchestrator |
| `conch_dna/superego/__init__.py` | SuperegoLayer - combined checks |
| `conch_dna/ego/model.py` | EgoModel - personality core |
| `conch_dna/cortex/base.py` | CortexNeuron - neuron base class |
| `conch_dna/id/needs.py` | NeedsRegulator - drive system |
| `conch_dna/memory/store.py` | MemoryStore - hybrid memory |
| `conch_dna/training/pipeline.py` | TrainingPipeline - LoRA training |

## Configuration

### Needs Configuration (ID Layer)

```python
# Available presets
regulator = create_regulator("balanced")    # Default
regulator = create_regulator("curious")     # High curiosity
regulator = create_regulator("conservative") # High reliability

# Custom weights
custom_weights = {
    NeedType.SUSTAINABILITY: 0.25,
    NeedType.RELIABILITY: 0.30,
    NeedType.CURIOSITY: 0.25,
    NeedType.EXCELLENCE: 0.20,
}
```

### Model Configuration (CORTEX)

```python
# Configure neuron models
ThinkCortex(
    base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    lora_rank=16,
    confidence_threshold=0.7
)
```

## API Reference

### ConsciousnessLoop

```python
loop = ConsciousnessLoop(data_dir=Path("data"))
loop.initialize()           # Load all components
state = loop.run_cycle()    # Run single cycle
loop.run()                  # Run continuous loop
```

### CortexNeuron

```python
output = neuron.infer(input_data)     # Run inference
output.confidence                      # Confidence score
output.should_fallback                 # Whether to use EGO
output.content                         # Generated content
neuron.record_outcome(output, score)  # Record for training
```

### SuperegoLayer

```python
result = superego.check_thought(thought)
result.is_approved      # Whether thought is approved
result.values_ok        # Values check passed
result.safety_ok        # Safety check passed
result.get_summary()    # Human-readable summary
```

## Performance

| Metric | Value |
|--------|-------|
| Test Pass Rate | 10/10 (100%) |
| ThinkCortex Inference | ~14s (1.5B model) |
| ActionCortex Inference | ~3s (0.5B model) |
| Memory Usage (EGO) | ~4GB |
| Memory Usage (Neurons) | ~0.3-1.7GB each |

## Research

This project explores several research questions:

1. **Can safety be architectural?** - Rather than training for safety, can we make unsafe behavior structurally impossible?

2. **Can small models specialize?** - Can 0.5B models with LoRA adapters match larger models in specific domains?

3. **Can consciousness be modular?** - Can Freudian concepts (ID, EGO, SUPEREGO) map to useful AI architectures?

4. **Can learning be supervised by peers?** - Can the EGO teach neurons through correction examples?

## Documentation

- [Final Architecture](docs/MINDFORGE_DNA_FINAL_ARCHITECTURE.md) - Complete specification
- [Quick Reference](docs/MINDFORGE_DNA_QUICKREF.md) - Cheat sheet
- [KVRM Whitepaper](docs/KVRM-WHITEPAPER.md) - Fact grounding system
- [DNA Neuron Research](docs/DNA_NEURON_PIVOT_RESEARCH.md) - Research notes

## Contributing

This project is under active development. See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## Citation

```bibtex
@software{conch_dna_2025,
  title = {Conch DNA: A Freudian-Inspired AI Consciousness Architecture},
  author = {Price, Bobby},
  year = {2025},
  url = {https://github.com/yourusername/conscious}
}
```

---

*Conch DNA v0.2.0 - December 2025*

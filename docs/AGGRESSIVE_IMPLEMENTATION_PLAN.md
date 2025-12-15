# Conch DNA: Aggressive 3-Week Implementation Plan

**Timeline**: 3 Weeks (21 Days)
**Original Estimate**: 6+ weeks
**Compression Strategy**: Parallel work streams, minimal viable features first, defer optimization

---

## Week 1: Foundation (Days 1-7)

### Day 1-2: Project Scaffolding + ID Layer
- [x] Create project directory structure
- [ ] Implement `conch_dna/id/needs.py` (NeedsRegulator) - REUSE existing
- [ ] Write tests for NeedsRegulator
- [ ] Verify: needs increase/decrease correctly

**Deliverable**: Working NeedsRegulator with tests

### Day 3-4: Superego Layer
- [ ] Implement `conch_dna/superego/values.py` (CoreValues)
- [ ] Implement `conch_dna/superego/safety.py` (SafetyChecker)
- [ ] Implement `conch_dna/superego/kvrm.py` (KVRMRouter)
- [ ] Write tests for Superego
- [ ] Verify: blocked commands rejected, values checked, claims grounded

**Deliverable**: Complete Superego layer with veto power

### Day 5-6: EGO Model
- [ ] Install MLX and dependencies
- [ ] Download Qwen3-8B-Instruct (4-bit)
- [ ] Implement `conch_dna/ego/model.py` (EgoModel class)
- [ ] Write personality prompt (PERSONALITY_PROMPT)
- [ ] Test basic inference works
- [ ] Implement timing decision function

**Deliverable**: EGO generating personality-consistent responses

### Day 7: Main Loop V1 (EGO-Only)
- [ ] Implement `conch_dna/main.py` loop structure
- [ ] EGO-only cycle (no neurons yet)
- [ ] Integrate tools (reuse existing shell, filesystem)
- [ ] Data logging for all EGO outputs
- [ ] Verify: cycle runs end-to-end without crash

**Deliverable**: EGO-only consciousness loop running

---

## Week 2: Cortex Neurons (Days 8-14)

### Day 8-9: Cortex Base + Action Neuron
- [ ] Implement `conch_dna/cortex/base.py` (CortexNeuron base)
- [ ] Download Qwen3-1.7B-Instruct
- [ ] Implement `conch_dna/cortex/action.py` (ActionCortex)
- [ ] Create action training data from EGO outputs
- [ ] Set up LoRA training pipeline
- [ ] Train action_v1 LoRA (r=8)

**Deliverable**: ActionCortex with trained LoRA

### Day 10-11: Task + Think Neurons
- [ ] Implement `conch_dna/cortex/task.py` (TaskCortex)
- [ ] Train task_v1 LoRA (r=8)
- [ ] Download Qwen3-4B-Instruct
- [ ] Implement `conch_dna/cortex/think.py` (ThinkCortex)
- [ ] Train think_v1 LoRA (r=16)

**Deliverable**: Task and Think neurons operational

### Day 12-13: Reflect + Debug Neurons
- [ ] Implement `conch_dna/cortex/reflect.py` (ReflectCortex)
- [ ] Train reflect_v1 LoRA (r=8)
- [ ] Implement `conch_dna/cortex/debug.py` (DebugCortex)
- [ ] Train debug_v1 LoRA (r=16)

**Deliverable**: Reflect and Debug neurons operational

### Day 14: Memory Neuron + Integration
- [ ] Download Qwen3-1.7B
- [ ] Implement `conch_dna/cortex/memory.py` (MemoryCortex)
- [ ] Integrate ChromaDB for vectors
- [ ] Implement importance scoring
- [ ] Integrate all 6 neurons into main loop

**Deliverable**: All 6 cortex neurons integrated

---

## Week 3: Training + Stabilization (Days 15-21)

### Day 15-16: Training Pipeline
- [ ] Implement `conch_dna/training/pipeline.py`
- [ ] Success recording working
- [ ] Failure correction working (EGO generates corrections)
- [ ] Automatic retraining trigger (100 new examples)

**Deliverable**: Self-improving training pipeline

### Day 17-18: Fallback Logic + Confidence
- [ ] Implement confidence-based EGO fallback
- [ ] Calibrate confidence thresholds per neuron
- [ ] Test fallback triggers appropriately
- [ ] Track fallback rate metrics

**Deliverable**: Graceful degradation working

### Day 19-20: Data Collection Run
- [ ] Run 1,000+ live cycles
- [ ] Monitor neuron accuracy
- [ ] Monitor EGO fallback rate
- [ ] Fix any stability issues
- [ ] Verify: system stable for 24+ hours

**Deliverable**: 1,000+ cycles completed, system stable

### Day 21: Polish + Documentation
- [ ] Performance optimization pass
- [ ] Update documentation
- [ ] Create startup scripts
- [ ] Final integration tests

**Deliverable**: Production-ready Conch DNA

---

## Success Metrics

| Metric | Target | Week 1 | Week 2 | Week 3 |
|--------|--------|--------|--------|--------|
| Cycle time | 30-90s | 120-180s | 60-90s | 30-60s |
| EGO fallback rate | <20% | 100% | <50% | <20% |
| Neuron accuracy | >85% | N/A | >70% | >85% |
| Uptime | >99% | >80% | >95% | >99% |

---

## Directory Structure

```
conch_dna/
├── __init__.py
├── main.py                  # Consciousness loop entry point
├── config.yaml              # Configuration
│
├── id/                      # ID Layer (drives)
│   ├── __init__.py
│   └── needs.py             # NeedsRegulator
│
├── superego/                # Superego Layer (constraints)
│   ├── __init__.py
│   ├── values.py            # CoreValues (immutable)
│   ├── safety.py            # SafetyChecker
│   └── kvrm.py              # KVRMRouter
│
├── ego/                     # EGO Model (personality DNA)
│   ├── __init__.py
│   └── model.py             # EgoModel (Qwen3-8B)
│
├── cortex/                  # Cortex Neurons (6 specialists)
│   ├── __init__.py
│   ├── base.py              # CortexNeuron base class
│   ├── think.py             # ThinkCortex (1.5B)
│   ├── task.py              # TaskCortex (0.5B)
│   ├── action.py            # ActionCortex (0.5B)
│   ├── reflect.py           # ReflectCortex (0.5B)
│   ├── debug.py             # DebugCortex (0.5B)
│   └── memory.py            # MemoryCortex (1.7B)
│
├── tools/                   # Tool implementations
│   ├── __init__.py
│   ├── base.py              # Tool base class
│   ├── shell.py
│   ├── filesystem.py
│   └── git.py
│
├── training/                # Training pipeline
│   ├── __init__.py
│   ├── pipeline.py          # Main training orchestration
│   ├── experience.py        # Experience buffer
│   └── lora.py              # LoRA training utilities
│
├── memory/                  # Memory system
│   ├── __init__.py
│   └── store.py             # ChromaDB + SQLite hybrid
│
└── data/                    # Runtime data
    ├── facts.db             # KVRM fact store
    ├── memories.db          # Memory store
    ├── adapters/            # LoRA adapters
    │   ├── action_v1.safetensors
    │   ├── task_v1.safetensors
    │   └── ...
    └── training/            # Training data
        ├── ego_outputs.jsonl
        └── corrections.jsonl
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MLX model loading slow | Pre-download, cache aggressively |
| LoRA training fails | Start with small r, increase if needed |
| EGO too slow | Accept 60-120s initially, optimize later |
| Neurons drift | Weekly alignment audits from EGO |
| Memory grows unbounded | Implement CLaRa compression in Week 4 |

---

## Day 1 Starting Commands

```bash
# Create project structure
mkdir -p conch_dna/{id,superego,ego,cortex,tools,training,memory,data/adapters,data/training}

# Create __init__.py files
touch conch_dna/__init__.py
touch conch_dna/{id,superego,ego,cortex,tools,training,memory}/__init__.py

# Install dependencies
pip install mlx mlx-lm chromadb sentence-transformers

# Download base models
mlx_lm.convert --hf-path Qwen/Qwen3-8B-Instruct -q
mlx_lm.convert --hf-path Qwen/Qwen3-1.7B-Instruct -q
```

---

**Let's build this. Day 1 starts now.**

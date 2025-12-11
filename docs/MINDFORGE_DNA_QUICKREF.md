# MindForge DNA: Quick Reference & Execution Checklist

## Architecture At a Glance

```
SUPEREGO (Immutable) ─────────────────────────────────────────
│ • Core Values (benevolence, honesty, humility, safety)
│ • KVRM (fact verification, zero hallucination)
│ • Safety (blocked commands, rate limits, timeouts)
│ NO LEARNING. These are axioms.
│
▼
EGO (Qwen3-8B-4bit @ MLX) ────────────────────────────────────
│ • THE personality DNA
│ • Teacher: generates training data
│ • Corrector: fixes neuron mistakes
│ • Arbiter: final say on edge cases
│ • Timer: decides sleep duration (15s-30min)
│ Runs on EVERY cycle for first 10,000 cycles
│
▼
CORTEX (6 Neurons) ───────────────────────────────────────────
│ Think     │ Task      │ Action    │ Reflect   │ Debug     │ Memory
│ 1.5B+LoRA │ 0.5B+LoRA │ 0.5B+LoRA │ 0.5B+LoRA │ 0.5B+LoRA │ 1.7B+LoRA
│           │           │           │           │           │ +CLaRa
│ Distilled from EGO. Inherit personality DNA.
│
▼
ID (Pure Math) ───────────────────────────────────────────────
  • NeedsRegulator: sustainability, reliability, curiosity, excellence
  • urgency = weight × (0.5 + level) × time_decay
  NO LEARNING. Drives are designed, not discovered.
```

---

## The 6 Cortex Neurons

| Neuron | Base Model | LoRA Rank | Function |
|--------|------------|-----------|----------|
| **Think** | Qwen2.5-1.5B | r=16 | Thought generation, reasoning |
| **Task** | Qwen2.5-0.5B | r=8 | Task extraction, prioritization |
| **Action** | Qwen2.5-0.5B | r=8 | Tool selection, call formatting |
| **Reflect** | Qwen2.5-0.5B | r=8 | Reflection, journaling, mood |
| **Debug** | Qwen2.5-0.5B | r=16 | Error analysis, fix suggestions |
| **Memory** | SmolLM2-1.7B | r=16 | Retrieve, compress, reconstruct |

---

## Memory System Rules

```
IF importance >= 0.75:
    STORE RAW (never compress)          ← Sacred memories
ELSE:
    STORE COMPRESSED (CLaRa)            ← Routine memories

ALL memories get KVRM keys:
    mem:reflection:20251211:a1b2c3d4
    mem:fact:learned:20251211:x9y8z7
```

---

## Training Pipeline

```
SUCCESS (reward > 0.7)
    → Store as positive training example
    
FAILURE
    → EGO analyzes: what went wrong, why, correct answer
    → Store correction as training example (HIGHEST VALUE DATA)
    
EGO FALLBACK (confidence < 0.75)
    → EGO handles it
    → Store EGO response as training example
    
EVERY 100 NEW EXAMPLES
    → Retrain neuron's LoRA
    → Validate against held-out set
    → Accept only if improved
```

---

## Timing Decision (EGO Function)

```
Range: 15 seconds to 30 minutes

WAKE QUICKLY when:
    • Any need > 0.85
    • Critical task pending
    • Just failed something
    • Human just interacted

SLEEP LONGER when:
    • Needs satisfied
    • No critical tasks
    • Feeling content
    • Want to "dream"
```

---

## External Control Signals

```bash
# Immediate wake
echo "wake" > /tmp/mindforge_signal

# Force 10-minute sleep
echo "sleep 600" > /tmp/mindforge_signal

# Emergency shutdown
echo "die" > /tmp/mindforge_signal
```

---

## Hardware Budget (M4 Pro 24GB)

| Component | VRAM |
|-----------|------|
| EGO (8B 4-bit) | ~5GB |
| Think (1.5B) | ~1.5GB |
| Memory (1.7B) | ~1.7GB |
| Other neurons (0.5B × 4) | ~2GB |
| ChromaDB + overhead | ~3GB |
| **Total** | **~14GB ✓** |

---

## Latency Targets

| Operation | Target |
|-----------|--------|
| EGO inference | 60-120s |
| Neuron inference | 0.5-2s |
| Memory retrieval | <500ms |
| KVRM grounding | <50ms |
| Full cycle (bootstrap) | 120-180s |
| Full cycle (steady) | 30-60s |

---

# Day-by-Day Execution Checklist

## Week 1: Foundation + EGO

### Day 1 ☐
- [ ] Create project directory structure
- [ ] Implement `id/needs.py` (NeedsRegulator)
- [ ] Write tests for NeedsRegulator
- [ ] Verify: needs increase/decrease correctly

### Day 2 ☐
- [ ] Implement `superego/values.py` (CoreValues)
- [ ] Implement `superego/safety.py` (SafetyChecker)
- [ ] Write tests for Superego
- [ ] Verify: blocked commands rejected, values checked

### Day 3 ☐
- [ ] Implement `superego/kvrm.py` (KVRMRouter)
- [ ] Set up SQLite fact store
- [ ] Write tests for KVRM
- [ ] Verify: claims classified, facts verified

### Day 4 ☐
- [ ] Install MLX and dependencies
- [ ] Download Qwen3-8B-Instruct
- [ ] Get basic inference working
- [ ] Verify: model generates responses

### Day 5 ☐
- [ ] Write personality prompt (PERSONALITY_PROMPT)
- [ ] Implement `ego/model.py` EgoModel class
- [ ] Test personality consistency
- [ ] Verify: EGO sounds like Echo

### Day 6 ☐
- [ ] Implement `decide_next_wakeup()` in EgoModel
- [ ] Test timing decisions
- [ ] Implement `correct_failure()` in EgoModel
- [ ] Verify: timing in range, corrections useful

### Day 7 ☐
- [ ] Implement `generate_distillation_example()` in EgoModel
- [ ] Implement `audit_neuron_response()` in EgoModel
- [ ] Full EGO integration test
- [ ] Verify: all EGO functions working

**Week 1 Milestone**: EGO model complete, can generate thoughts, timing, corrections

---

## Week 2: Main Loop + Data Collection

### Day 8 ☐
- [ ] Implement `cortex/base.py` (CortexNeuron base)
- [ ] Implement basic `main.py` loop structure
- [ ] EGO-only cycle (no neurons yet)
- [ ] Verify: cycle runs, doesn't crash

### Day 9 ☐
- [ ] Implement `tools/shell.py`
- [ ] Implement `tools/filesystem.py`
- [ ] Integrate tools with main loop
- [ ] Verify: tools execute, results captured

### Day 10 ☐
- [ ] Implement data logging (all EGO outputs)
- [ ] Set up training data directory
- [ ] Log format: JSONL with input/output/timestamp
- [ ] Verify: all cycles logged

### Day 11 ☐
- [ ] Run 500 cycles (EGO only)
- [ ] Review output quality
- [ ] Adjust personality prompt if needed
- [ ] Verify: outputs are in-character

### Day 12-13 ☐
- [ ] Run 2,000+ more cycles
- [ ] Monitor for errors
- [ ] Fix any issues that arise
- [ ] Verify: stable operation

### Day 14 ☐
- [ ] Total: 5,000+ cycles collected
- [ ] Organize training data by domain
- [ ] Calculate statistics (tokens, domains, etc.)
- [ ] Verify: sufficient data for training

**Week 2 Milestone**: 5,000+ gold-standard (input, EGO output) pairs collected

---

## Week 3: First Neurons

### Day 15 ☐
- [ ] Implement `cortex/action.py` (ActionCortex)
- [ ] Extract action-specific training data
- [ ] Format for LoRA training
- [ ] Verify: 1,000+ action examples

### Day 16 ☐
- [ ] Set up LoRA training pipeline
- [ ] Download Qwen2.5-0.5B base
- [ ] Train action_v1 LoRA (r=8)
- [ ] Verify: training completes

### Day 17 ☐
- [ ] Evaluate action_v1 accuracy
- [ ] Compare to EGO outputs
- [ ] Target: >85% match
- [ ] Verify: meets accuracy threshold

### Day 18 ☐
- [ ] Integrate ActionCortex with main loop
- [ ] Implement confidence-based fallback
- [ ] Test with real cycles
- [ ] Verify: fallback triggers appropriately

### Day 19 ☐
- [ ] Implement `cortex/task.py` (TaskCortex)
- [ ] Extract task-specific training data
- [ ] Train task_v1 LoRA
- [ ] Verify: task extraction working

### Day 20-21 ☐
- [ ] Implement `cortex/think.py` (ThinkCortex)
- [ ] Extract think-specific training data
- [ ] Train think_v1 LoRA (1.5B base, r=16)
- [ ] Verify: thoughts are in-character

**Week 3 Milestone**: Action, Task, Think neurons operational

---

## Week 4: Remaining Neurons + Memory

### Day 22-23 ☐
- [ ] Implement `cortex/reflect.py` (ReflectCortex)
- [ ] Train reflect_v1 LoRA
- [ ] Implement mood assessment
- [ ] Verify: reflections match personality

### Day 24-25 ☐
- [ ] Implement `cortex/debug.py` (DebugCortex)
- [ ] Train debug_v1 LoRA
- [ ] Test on actual failures
- [ ] Verify: suggestions are useful

### Day 26-28 ☐
- [ ] Implement `cortex/memory.py` (MemoryCortex)
- [ ] Set up ChromaDB for vectors
- [ ] Implement importance scoring
- [ ] Implement retrieve/store
- [ ] Verify: memories stored and retrieved

**Week 4 Milestone**: All 6 cortex neurons operational

---

## Week 5: Training Pipeline

### Day 29-30 ☐
- [ ] Implement `training/pipeline.py`
- [ ] Success recording working
- [ ] Failure correction working
- [ ] Verify: examples accumulating

### Day 31-32 ☐
- [ ] Implement automatic retraining trigger
- [ ] Retrain when 100 new examples
- [ ] Validation before accepting
- [ ] Verify: neurons improving

### Day 33-35 ☐
- [ ] Run 1,000+ live cycles
- [ ] Monitor neuron accuracy
- [ ] Monitor EGO fallback rate
- [ ] Verify: fallback rate decreasing

**Week 5 Milestone**: Self-improving system operational

---

## Week 6+: Stabilization

### Day 36-42 ☐
- [ ] Implement alignment auditing
- [ ] Weekly personality checks
- [ ] Fix any drift issues
- [ ] Verify: personality stable

### Day 43-49 ☐
- [ ] Performance optimization
- [ ] Reduce latency where possible
- [ ] Memory optimization
- [ ] Verify: hitting latency targets

### Day 50+ ☐
- [ ] Long-term operation testing
- [ ] Documentation completion
- [ ] Edge case handling
- [ ] Verify: production-ready

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Cycle time (steady) | 30-60s | Time from wake to sleep decision |
| EGO fallback rate | <20% | Count fallbacks / total cycles |
| Neuron accuracy | >88% | Match rate vs EGO outputs |
| Memory retrieval | <500ms | Time to find relevant memories |
| Personality consistency | Subjective | Alignment audit score |
| Uptime | >99% | Hours running / hours attempted |

---

## Emergency Procedures

### System Hangs
```bash
# Kill and restart
pkill -f mindforge
python main.py
```

### Neuron Performing Badly
```python
# Rollback to previous adapter version
neuron.load_adapter("adapters/action_v2.safetensors")  # Previous version
```

### Memory Corruption
```bash
# Backup and recreate
cp data/memories.db data/memories.db.backup
# Delete and let system rebuild
rm data/memories.db
```

### EGO Giving Bad Outputs
1. Check personality prompt for drift
2. Verify model loaded correctly
3. Check temperature settings
4. Restart with fresh state

---

## Key Files to Watch

| File | What to Check |
|------|---------------|
| `data/training/*.jsonl` | Examples accumulating |
| `data/adapters/*.safetensors` | New versions appearing |
| `data/memories.db` | Size growing appropriately |
| `/tmp/mindforge.log` | Error messages |

---

## Final Reminders

1. **EGO runs on EVERY cycle for first 10,000 cycles** — accept this
2. **Failures are the best training data** — embrace them
3. **Sacred memories (≥0.75 importance) are never compressed**
4. **Values are immutable** — Superego cannot learn
5. **Timing is conscious** — every sleep is a decision

---

**You have everything you need. Go build it.**

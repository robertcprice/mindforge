# Conch Documentation

This folder now hosts the Conch DNA (SUPEREGO/EGO/CORTEX/ID) architecture docs. The older LangGraph/agent writeups are kept as legacy references and are called out below.

## Key Docs (Current Architecture)

- `MINDFORGE_DNA_FINAL_ARCHITECTURE.md` — locked, final architecture spec for Conch DNA
- `MINDFORGE_DNA_QUICKREF.md` — concise checklist for running the SUPEREGO/EGO/CORTEX/ID stack
- `KVRM-WHITEPAPER.md` — fact-grounding and key/value routing details
- `DNA_NEURON_PIVOT_RESEARCH.md` — neuron specialization research notes
- `TEST_RESULTS.md` — prior capability evaluations

## Legacy (Pre-DNA Agent) Materials

These describe the earlier LangGraph-based agent and are retained for history:

- `research/CONCH_RESEARCH_PAPER.md`
- `research/TASK_COMPLETION_EVIDENCE.md`
- `research/ARCHITECTURE_OVERVIEW.md`
- `cycles/CYCLE_CAPTURE_LOG.md` and `cycles/raw_cycles.json`
- `screenshots/`

## Run / Validation Pointers

- Full DNA test suite: `python3 test_conch_dna_full.py`
- Single consciousness cycle (DNA path): `python -m conch_dna.main --cycles 1 --debug`
- Training entry point: `scripts/train_conch.py`

## Citation

```bibtex
@software{conch_dna_2025,
  title = {Conch DNA: A Freudian-Inspired AI Consciousness Architecture},
  author = {Price, Bobby},
  year = {2025},
  url = {https://github.com/username/conch}
}
```

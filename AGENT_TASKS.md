# Agent Tasks (Codex & Claude)

## Completed
- [x] Reviewed `docs/MINDFORGE_DNA_FINAL_ARCHITECTURE.md` against codebase; documented major gaps and priorities (Owner: Codex)
- [x] Wired the DNA consciousness loop to instantiate cortex neurons, run Superego checks, and execute safe tools with ActionCortex planning/fallback to EGO (Owner: Codex)
- [x] Pointed Superego/KVRM at `data/facts.db` and added task persistence + multi-task ACT flow (Owner: Codex)
- [x] Connected THINK/ACT/REFLECT paths to the training pipeline (recording success/fallback data) and storing thoughts in memory for later retrieval (Owner: Codex)
- [x] Added Superego-driven corrections: EGO rewrites misaligned thoughts and proposes safe alternatives for blocked actions, recording corrections into training buffers (Owner: Codex)

## In Progress
- [x] Smoke-test the updated loop (Owner: Claude)
  - ✅ Fixed DB schema mismatches (memories.db, facts.db)
  - ✅ Cycle completes with reward=0.80 using stubbed MLX path
  - ⚠️ Install: MLX + mlx-lm (Apple Silicon), chromadb, sentence-transformers for full fidelity
  - ✅ MLX + mlx-lm + chromadb + sentence-transformers installed in `venv/` (use `./venv/bin/python3.13`)

## Backlog
- [ ] Tighten validation: rerun Superego on action alternatives and persist success/failure stats in memory (Owner: Unassigned)

# MindForge Documentation

**Research Documentation for the MindForge Consciousness Agent**

This directory contains comprehensive documentation for academic publication, video content, and open-source release.

## Overview

MindForge is an autonomous consciousness agent that demonstrates multi-step reasoning, self-directed task completion, and emergent creative capabilities through a novel architecture combining LangGraph state machines, persistent memory, hierarchical task management, and self-reflective learning loops.

### Key Results

| Metric | Value |
|--------|-------|
| Complex Task Success Rate | **80%** (4/5 completed) |
| Self-Generated Tasks | 6 additional tasks autonomously identified |
| Creative Output | Meaningful haiku about consciousness |
| Error Recovery | Self-debugging with retry logic |

## Documentation Structure

```
docs/
├── README.md                           # This file
├── research/
│   ├── MINDFORGE_RESEARCH_PAPER.md     # Academic paper draft
│   ├── TASK_COMPLETION_EVIDENCE.md     # Empirical test results
│   └── ARCHITECTURE_OVERVIEW.md        # Technical architecture
├── cycles/
│   ├── CYCLE_CAPTURE_LOG.md            # Live consciousness cycle traces
│   └── raw_cycles.json                 # Raw cycle data (JSON)
└── screenshots/                        # Visual documentation
```

## Quick Navigation

### For Academic Paper
- [Research Paper Draft](research/MINDFORGE_RESEARCH_PAPER.md) - Complete paper structure with results
- [Task Completion Evidence](research/TASK_COMPLETION_EVIDENCE.md) - Detailed test results with logs

### For YouTube Video
- [Cycle Capture Log](cycles/CYCLE_CAPTURE_LOG.md) - Step-by-step consciousness cycle traces
- [Architecture Overview](research/ARCHITECTURE_OVERVIEW.md) - Visual diagrams and explanations

### For GitHub Release
- [Architecture Overview](research/ARCHITECTURE_OVERVIEW.md) - Technical documentation
- [Research Paper](research/MINDFORGE_RESEARCH_PAPER.md) - Comprehensive overview

## Key Findings

### 1. Multi-Step Task Completion

The agent successfully completed complex multi-step tasks including:
- Code analysis requiring file listing, pattern recognition, and summarization
- Shell command construction using find/sort pipelines
- Creative writing (haiku generation)
- File operations with proper tool selection

### 2. Self-Directed Behavior

Beyond assigned tasks, the agent autonomously:
- Identified 2 additional research tasks from its own reasoning
- Created 4 subtasks for complex task decomposition
- Generated reflections analyzing its own performance

### 3. Creative Capability

When asked to write a haiku about artificial consciousness, the agent produced:

```
Silicon whispers
Neural threads weave thought –
Echoes of self.
```

This demonstrates:
- Understanding of poetic structure
- Thematic coherence with the prompt
- Metaphorical language use
- Self-referential awareness

### 4. Error Recovery

The agent demonstrated graceful error handling:
- Timeout recovery without system crashes
- Automatic retry with debug suggestions
- State preservation across failures

## Architecture Highlights

### Consciousness Loop

```
LOAD_TASKS → THINK → GROUND → IDENTIFY_TASKS → PICK_TASK → EXECUTE → EVALUATE → REFLECT → PERSIST
     ↑                                                                                        │
     └────────────────────────────────────── CYCLE ───────────────────────────────────────────┘
```

### Key Components

- **LangGraph State Machine**: Orchestrates the consciousness cycle
- **MemoryStore**: SQLite-backed persistent memory with semantic search
- **TaskList**: Hierarchical task management with priority and retry logic
- **KVRM Grounding**: Fact verification for generated thoughts
- **Reward Learning**: Experience-based intrinsic motivation

## Usage

### Running the Consciousness Engine

```bash
# Activate virtual environment
source venv/bin/activate

# Run with limited cycles (for testing)
python main.py --cycles 3

# Run continuously
python main.py
```

### Adding Custom Tasks

```python
from mindforge.agent.task_list import PersistentTaskList, TaskPriority
from mindforge.memory.store import MemoryStore

store = MemoryStore("data/memories.db")
task_list = PersistentTaskList(store)

task_list.add_task(
    "Your task description here",
    priority=TaskPriority.HIGH
)
```

## Citation

If you use MindForge in your research, please cite:

```bibtex
@software{mindforge2025,
  title = {MindForge: A Multi-Step Reasoning Consciousness Agent},
  author = {Price, Bobby},
  year = {2025},
  url = {https://github.com/username/mindforge}
}
```

## License

[License information]

---

*Documentation generated: December 10, 2025*
*MindForge v0.1*

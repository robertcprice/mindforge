# MindForge

**An Autonomous Consciousness Agent with Multi-Step Reasoning and Self-Directed Task Completion**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-green.svg)](https://github.com/langchain-ai/langgraph)

MindForge is a novel consciousness simulation engine that demonstrates emergent cognitive capabilities through a unique architecture combining LangGraph state machines, persistent memory, hierarchical task management, and zero-hallucination grounding via KVRM (Key-Value Response Mapping).

## Key Features

- **Autonomous Consciousness Loop**: Think → Ground → Decide → Act → Reflect → Learn
- **Multi-Step Task Completion**: 80% success rate on complex multi-step tasks
- **Self-Directed Behavior**: Autonomously identifies and creates subtasks
- **Zero-Hallucination Grounding**: KVRM system verifies factual claims
- **Persistent Memory**: SQLite-backed episodic and semantic memory
- **Reward-Based Learning**: Experience buffer with intrinsic motivation
- **Error Recovery**: Self-debugging with intelligent retry logic
- **Multi-Backend Inference**: Ollama, MLX (Apple Silicon), Transformers

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/mindforge.git
cd mindforge

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start Ollama (required for inference)
ollama serve &
ollama pull qwen2.5:7b

# Run the consciousness engine
python main.py --cycles 3
```

## Architecture

```
MindForge Consciousness Loop
============================

┌─────────────────────────────────────────────────────────────────────┐
│                         CONSCIOUSNESS CYCLE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────────┐     │
│  │  LOAD   │───▸│  THINK  │───▸│  GROUND  │───▸│  IDENTIFY   │     │
│  │  TASKS  │    │         │    │  (KVRM)  │    │   TASKS     │     │
│  └─────────┘    └─────────┘    └──────────┘    └─────────────┘     │
│       │                                               │             │
│       │              WORK LOOP                        ▼             │
│       │         ┌────────────────────────────────────────┐          │
│       │         │  ┌─────────┐   ┌─────────┐   ┌───────┐ │          │
│       │         │  │  PICK   │──▸│ EXECUTE │──▸│ EVAL  │ │          │
│       │         │  │  TASK   │   │  TASK   │   │RESULT │ │          │
│       │         │  └─────────┘   └─────────┘   └───────┘ │          │
│       │         │       ▲                          │     │          │
│       │         │       └────── DEBUG ◀────────────┘     │          │
│       │         └────────────────────────────────────────┘          │
│       │                              │                              │
│       │         ┌────────────────────▼────────────────────┐         │
│       │         │  ┌─────────┐   ┌─────────┐   ┌────────┐ │         │
│       │         │  │ REFLECT │──▸│ JOURNAL │──▸│ UPDATE │ │         │
│       │         │  │         │   │  ENTRY  │   │ NEEDS  │ │         │
│       │         │  └─────────┘   └─────────┘   └────────┘ │         │
│       │         └────────────────────────────────────────┘          │
│       │                              │                              │
│       └──────────────────────────────┴──────────────────────────────│
│                           SLEEP & REPEAT                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mindforge/
├── agent/                  # Core consciousness agent
│   ├── langgraph_agent.py  # Main consciousness loop (LangGraph state machine)
│   ├── task_list.py        # Hierarchical persistent task management
│   ├── journal.py          # Self-reflective journaling system
│   └── tool_adapter.py     # Tool integration layer
├── core/                   # Core systems
│   ├── needs.py            # Intrinsic motivation (sustainability, curiosity, etc.)
│   ├── thought.py          # Thought generation
│   └── state.py            # Consciousness state management
├── kvrm/                   # Zero-Hallucination Grounding
│   ├── grounding.py        # Factual claim verification
│   ├── key_stores.py       # Memory, fact, and external key stores
│   └── parser.py           # Claim extraction and classification
├── memory/                 # Memory systems
│   ├── store.py            # SQLite-backed persistent memory
│   ├── experience.py       # Experience buffer for reward learning
│   └── semantic.py         # Semantic search and retrieval
├── tools/                  # Tool implementations
│   ├── shell.py            # Safe shell command execution
│   ├── filesystem.py       # File operations
│   ├── web.py              # Web fetching and search
│   ├── git.py              # Git operations
│   └── base.py             # Tool base classes and registry
├── training/               # Fine-tuning pipeline
│   ├── lora.py             # LoRA training
│   └── trainer.py          # Training orchestration
├── inference/              # LLM backends
│   ├── mlx_backend.py      # Apple Silicon MLX
│   └── transformers_backend.py
├── integrations/           # External services
│   ├── ollama.py           # Ollama API client
│   └── n8n.py              # n8n workflow automation
├── dashboard/              # Streamlit monitoring UI
└── api/                    # FastAPI REST interface

docs/
├── README.md               # Documentation overview
├── research/
│   ├── MINDFORGE_RESEARCH_PAPER.md
│   ├── ARCHITECTURE_OVERVIEW.md
│   └── TASK_COMPLETION_EVIDENCE.md
├── cycles/                 # Consciousness cycle traces
├── KVRM-WHITEPAPER.md      # Zero-hallucination system
└── REWARD_LEARNING_SYSTEM.md
```

## Key Components

### 1. Consciousness Agent (`mindforge/agent/langgraph_agent.py`)

The heart of MindForge - a LangGraph-based state machine that orchestrates the consciousness loop:

- **Think**: Generate contextual thoughts based on current needs and memory
- **Ground**: Verify thoughts against factual knowledge (KVRM)
- **Execute**: Work through tasks using available tools
- **Reflect**: Analyze outcomes and update internal state
- **Learn**: Store experiences and update reward-based motivation

### 2. KVRM Grounding (`mindforge/kvrm/`)

Zero-hallucination system that:
- Classifies claims as FACTUAL, OPINION, or QUESTION
- Verifies factual claims against stored facts
- Provides confidence scores and source attribution
- Prevents hallucinated responses

### 3. Task Management (`mindforge/agent/task_list.py`)

Hierarchical, persistent task system:
- Parent/child task relationships
- Priority levels (LOW, NORMAL, HIGH, CRITICAL)
- Automatic retry with debug suggestions
- Progress tracking and notes
- SQLite persistence across sessions

### 4. Tool System (`mindforge/tools/`)

Safe, sandboxed tool execution:
- Shell commands with safety guardrails
- File system operations
- Git integration
- Web fetching and search
- Extensible tool registry

## Performance

| Metric | Value |
|--------|-------|
| Complex Task Success Rate | **80%** (4/5 completed) |
| Self-Generated Tasks | 6 additional tasks autonomously identified |
| Error Recovery Rate | Automatic retry with debug suggestions |
| Memory Persistence | SQLite-backed with semantic search |

## Configuration

Configuration is managed through `config.yaml`:

```yaml
name: "Echo"  # Agent name

model:
  inference_backend: "mlx"  # or "ollama", "transformers"
  ollama_model_name: "qwen3:8b"

cycle:
  min_sleep_seconds: 30
  max_sleep_seconds: 300
  cycles_before_consolidation: 50
  cycles_before_mini_finetune: 200

needs:
  sustainability: 0.25
  reliability: 0.30
  curiosity: 0.25
  excellence: 0.20
```

## Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run mindforge/dashboard/app.py

# Or use the convenience script
./run.sh
```

The dashboard provides:
- Real-time consciousness cycle monitoring
- Task progress visualization
- Memory exploration
- Needs state display

## API Server

```bash
# Start the FastAPI server
uvicorn mindforge.api.main:app --reload --port 8000

# API docs at http://localhost:8000/docs
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_kvrm.py -v
pytest tests/test_memory.py -v
pytest tests/test_agents.py -v
```

## Documentation

- [Documentation Index](docs/README.md) - Full documentation overview
- [Research Paper](docs/research/MINDFORGE_RESEARCH_PAPER.md) - Academic paper draft
- [Architecture Overview](docs/research/ARCHITECTURE_OVERVIEW.md) - Technical details
- [KVRM Whitepaper](docs/KVRM-WHITEPAPER.md) - Zero-hallucination system
- [Reward Learning System](docs/REWARD_LEARNING_SYSTEM.md) - Intrinsic motivation

## Requirements

- Python 3.11+
- Ollama (for inference)
- SQLite (included with Python)

Optional:
- MLX (for Apple Silicon acceleration)
- n8n (for workflow automation)

## Citation

```bibtex
@software{mindforge2025,
  title = {MindForge: A Multi-Step Reasoning Consciousness Agent},
  author = {Price, Bobby},
  year = {2025},
  url = {https://github.com/yourusername/mindforge}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

*MindForge v0.1 - December 2025*

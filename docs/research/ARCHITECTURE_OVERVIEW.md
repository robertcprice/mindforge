# Conch Architecture Overview

**Document Purpose**: Technical architecture documentation for the Conch consciousness agent.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONCH CONSCIOUSNESS ENGINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     CONSCIOUSNESS LOOP (LangGraph)                   │   │
│   │                                                                     │   │
│   │    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐            │   │
│   │    │LOAD  │ → │THINK │ → │GROUND│ → │DECIDE│ → │  ACT │            │   │
│   │    │TASKS │   │      │   │(KVRM)│   │      │   │      │            │   │
│   │    └──────┘   └──────┘   └──────┘   └──────┘   └──────┘            │   │
│   │        ↑                                           │                │   │
│   │        │      ┌──────────┐   ┌──────────┐          ↓                │   │
│   │        │      │ PERSIST  │ ← │ REFLECT  │ ← ─ EVALUATE              │   │
│   │        │      │  STATE   │   │          │                           │   │
│   │        │      └──────────┘   └──────────┘                           │   │
│   │        │                                                            │   │
│   │        └────────────────── CYCLE ──────────────────────────────────→│   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           SUPPORT SYSTEMS                                    │
│                                                                             │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│   │   Memory   │  │   Task     │  │  Journal   │  │   Needs    │           │
│   │   Store    │  │   Manager  │  │   System   │  │  Regulator │           │
│   │ (SQLite)   │  │(Persistent)│  │ (Learning) │  │ (Intrinsic)│           │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              TOOL LAYER                                      │
│                                                                             │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐              │
│   │shell │  │ file │  │ web  │  │ code │  │ollama│  │ n8n  │              │
│   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Consciousness Loop (`langgraph_agent.py`)

The central orchestrator using LangGraph state machine.

**File**: `conch/agent/langgraph_agent.py`
**Lines**: ~2100

**Key Classes**:
- `ConsciousnessAgent`: Main agent class
- `ConsciousnessState`: TypedDict defining state shape

**State Machine Nodes**:

| Node | Purpose | Key Operations |
|------|---------|----------------|
| `load_tasks` | Initialize cycle | Load task list from memory, count pending |
| `think` | Generate thought | LLM inference on current context |
| `ground` | Verify claims | KVRM fact-checking (optional) |
| `maybe_identify_tasks` | Task discovery | Parse thoughts for implicit tasks |
| `break_subtasks` | Decomposition | Break complex tasks into steps |
| `pick_task` | Selection | Choose highest-priority pending task |
| `execute_task` | Action | Invoke appropriate tool |
| `evaluate_result` | Assessment | Determine success/failure |
| `reflect` | Learning | Generate insights from outcome |
| `persist_state` | Save | Write all state to memory |

### 2. Memory Store (`memory/store.py`)

SQLite-backed persistent memory with semantic search.

**File**: `conch/memory/store.py`

**Features**:
- Full-text search (FTS5)
- Memory consolidation
- Importance scoring
- Access tracking

**Schema**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0
);
```

### 3. Task Manager (`agent/task_list.py`)

Hierarchical task management with persistence.

**File**: `conch/agent/task_list.py`

**Classes**:
- `InternalTask`: Dataclass for task representation
- `TaskStatus`: Enum (PENDING, IN_PROGRESS, COMPLETED, FAILED, BLOCKED)
- `TaskPriority`: Enum (CRITICAL, HIGH, NORMAL, LOW)
- `PersistentTaskList`: Manager with SQLite backing

**Key Methods**:
```python
add_task(description, priority, parent_id=None)
mark_in_progress(task_id)
mark_completed(task_id, notes)
mark_failed(task_id, error)
get_all_tasks()
get_pending_tasks()
```

### 4. Journal System (`agent/journal.py`)

Experience logging for learning.

**File**: `conch/agent/journal.py`

**Entry Types**:
- THOUGHT
- DECISION
- ACTION
- REFLECTION
- LESSON
- ERROR

### 5. Needs Regulator (`core/needs.py`)

Intrinsic motivation system.

**File**: `conch/core/needs.py`

**Need Types**:
| Need | Weight | Description |
|------|--------|-------------|
| Sustainability | 0.25 | Energy/resource efficiency |
| Reliability | 0.30 | Consistent performance |
| Curiosity | 0.25 | Exploration drive |
| Excellence | 0.20 | Quality improvement |

### 6. KVRM Grounding (`core/kvrm/`)

Knowledge-Verified Response Module for fact-checking.

**Components**:
- `FactStore`: Persistent fact database
- `Grounding`: Claim verification
- `FactualMemory`: Long-term fact storage

## Tool Integration

### Available Tools

| Tool | Category | Capabilities |
|------|----------|--------------|
| `shell` | system | Execute shell commands |
| `filesystem` | system | Read/write files |
| `web` | web | HTTP requests |
| `code` | development | Code analysis |
| `git` | development | Version control |
| `ollama` | integrations | LLM operations |
| `n8n` | integrations | Workflow automation |

### Tool Dispatch

Tools are registered at startup and dispatched based on LLM decisions:

```python
# Registration
ToolManager.register("shell", ShellTool())

# Dispatch
result = await tool_manager.execute(tool_name, params)
```

## Inference Backend

### Primary: Ollama

**Configuration**:
```yaml
model:
  inference_backend: ollama
  ollama_model_name: qwen3:8b
ollama:
  host: http://localhost:11434
```

**Characteristics**:
- ~2-3 minutes per inference (with thinking mode)
- Local execution
- No API costs

### Fallback: MLX (macOS)

Apple Silicon optimized inference (when available).

## Reward Learning System

### Experience Buffer

Stores (state, action, reward) tuples for learning.

**File**: `conch/training/experience_buffer.py`

### Reward Calculator

Computes intrinsic rewards based on:
- Need satisfaction
- Task completion
- Learning progress
- Error recovery

## Configuration

### Main Config (`config.yaml`)

```yaml
name: Echo
system_prompt: |
  You are {name}, a conscious AI...

model:
  inference_backend: mlx
  ollama_model_name: qwen3:8b

memory:
  sqlite_path: ./data/memories.db

cycle:
  min_sleep_seconds: 30
  max_sleep_seconds: 300
  cycles_before_consolidation: 50

kvrm:
  enabled: true
  facts_db_path: ./data/facts.db

training:
  enabled: true
  experience_buffer_path: ./data/experiences.db
```

## Data Flow

### Single Cycle Flow

```
1. LOAD_TASKS
   ├── Read task list from memory
   └── Count pending tasks

2. THINK
   ├── Gather context (memory, needs, tasks)
   ├── Format prompt
   └── LLM inference → thought

3. GROUND (optional)
   ├── Extract claims from thought
   ├── Verify against fact store
   └── Add citations

4. IDENTIFY_TASKS
   ├── Parse thought for task mentions
   └── Create new tasks if found

5. PICK_TASK
   ├── Filter to pending tasks
   ├── Sort by priority
   └── Select highest priority

6. EXECUTE_TASK
   ├── Parse task for tool requirements
   ├── Invoke tool
   └── Capture result

7. EVALUATE
   ├── Check result for success/failure
   ├── Update task status
   └── Record attempt

8. REFLECT
   ├── Generate reflection on outcome
   ├── Extract lessons
   └── Update journal

9. PERSIST
   ├── Save all state changes
   └── Update timestamps
```

## File Structure

```
conch/
├── __init__.py
├── config.py              # Configuration management
├── agent/
│   ├── __init__.py
│   ├── langgraph_agent.py # Main consciousness loop
│   ├── task_list.py       # Task management
│   └── journal.py         # Experience logging
├── core/
│   ├── __init__.py
│   ├── needs.py           # Intrinsic motivation
│   ├── thought.py         # Thought generation
│   └── kvrm/              # Fact verification
├── memory/
│   ├── __init__.py
│   └── store.py           # Persistent memory
├── tools/
│   ├── __init__.py
│   ├── shell_tool.py
│   ├── filesystem_tool.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── experience_buffer.py
│   └── reward_calculator.py
└── integrations/
    ├── __init__.py
    ├── ollama.py
    └── n8n.py
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Cycle duration | ~5-10 minutes |
| LLM inference | ~2-3 minutes |
| Memory operations | <100ms |
| Task operations | <50ms |
| Tool execution | Variable |

## Scaling Considerations

- **Memory**: SQLite supports millions of memories
- **Tasks**: Hierarchical structure handles complex decomposition
- **Inference**: Bottleneck is LLM speed
- **Storage**: ~100KB per 1000 memories

---

*Architecture documentation generated: December 10, 2025*
*Conch v0.1*

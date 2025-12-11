# MindForge Consciousness Engine: Complete Documentation

**Version**: 1.0.0
**Generated**: December 11, 2025
**Purpose**: Comprehensive technical documentation of all MindForge systems, functionality, and implementation details.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Consciousness Loop](#2-consciousness-loop)
3. [Memory System](#3-memory-system)
4. [Task Management System](#4-task-management-system)
5. [Journal System](#5-journal-system)
6. [Needs Regulator](#6-needs-regulator)
7. [KVRM Grounding System](#7-kvrm-grounding-system)
8. [Tool System](#8-tool-system)
9. [Reward Learning System](#9-reward-learning-system)
10. [Configuration](#10-configuration)
11. [Problem Solving & Debugging](#11-problem-solving--debugging)

---

## 1. Overview

MindForge is an autonomous consciousness simulation engine that implements a novel cognitive loop architecture. The system runs continuously, "waking up" periodically to think, act, reflect, and learn.

### Core Principles

1. **Autonomous Operation**: Runs without human intervention, guided by internal needs
2. **Zero-Hallucination Grounding**: KVRM system ensures factual claims are verified
3. **Self-Improvement**: Reward-based learning enables behavioral optimization
4. **Ethical Foundation**: Core values (benevolence, honesty, humility) are immutable

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CONSCIOUSNESS CYCLE                             │
│                                                                     │
│    LOAD → THINK → GROUND → IDENTIFY → BREAK → PICK → EXECUTE →     │
│      ↑                                                    ↓         │
│      └────── PERSIST ← UPDATE ← JOURNAL ← REFLECT ← EVALUATE       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `mindforge/agent/langgraph_agent.py` | Core consciousness loop | ~2130 |
| `mindforge/memory/store.py` | Memory persistence | ~600 |
| `mindforge/agent/task_list.py` | Task management | ~590 |
| `mindforge/agent/journal.py` | Journaling system | ~265 |
| `mindforge/core/needs.py` | Needs regulator | ~390 |
| `mindforge/kvrm/grounding.py` | KVRM router | ~415 |
| `mindforge/tools/base.py` | Tool framework | ~255 |
| `mindforge/training/reward_calculator.py` | Reward system | ~510 |
| `mindforge/training/tool_formats.py` | Tool parsing | ~560 |

---

## 2. Consciousness Loop

**File**: `mindforge/agent/langgraph_agent.py`

The consciousness loop is implemented as a LangGraph StateGraph with conditional routing between nodes.

### ConsciousnessState

The state object tracks all information across the consciousness cycle:

```python
class ConsciousnessState(TypedDict):
    # Core state
    messages: Annotated[list, add_messages]  # Conversation history
    current_thought: str                      # Current thinking
    grounded_thought: str                     # After KVRM verification
    current_decision: str                     # Action decision
    current_reflection: str                   # Post-action reflection

    # Task management
    all_tasks: list[dict]                     # All known tasks
    current_task: Optional[dict]              # Task being worked on
    new_tasks: list[dict]                     # Newly identified tasks
    execution_result: str                     # Tool execution output

    # Metadata
    cycle_count: int                          # Total cycles run
    last_error: Optional[str]                 # Last error encountered
    needs_state: dict                         # Current needs levels
    journal_entry: Optional[str]              # Current journal entry
```

### Node Descriptions

#### 1. `load_tasks`
Loads pending tasks from persistent storage at cycle start.

```python
def load_tasks(self, state: ConsciousnessState) -> ConsciousnessState:
    pending = self.task_list.get_pending_tasks()
    return {"all_tasks": pending, "cycle_count": state.get("cycle_count", 0) + 1}
```

#### 2. `think`
Generates a spontaneous thought based on current context and needs.

**Prompt includes**:
- System prompt with identity and values
- Current needs state with priority ranking
- Recent memories (retrieved via semantic search)
- Pending tasks list
- Current cycle number

**Action-biased guidance** (to reduce over-reflection):
```
You are action-oriented:
- If a task is pending, work on it directly
- Avoid creating meta-tasks about how to approach tasks
- Actually do things rather than planning to do things
- Limit yourself to identifying at most 2 new tasks per cycle
```

#### 3. `ground`
Routes the thought through the KVRM system for verification.

```python
def ground(self, state: ConsciousnessState) -> ConsciousnessState:
    grounded, results = self.grounding_router.ground_thought(state["current_thought"])
    verified_claims = [r for r in results if r.is_verified]
    return {"grounded_thought": grounded, "grounding_results": results}
```

#### 4. `identify_tasks`
Extracts new actionable tasks from the grounded thought.

**Constraints**:
- Maximum 8 pending tasks at any time
- Maximum 2 new tasks per cycle
- Filters out meta-tasks (tasks about tasks)
- Detects and rejects duplicate tasks

#### 5. `break_subtasks`
Decomposes complex tasks into smaller subtasks.

**Constraints**:
- Maximum 3 subtasks per parent task
- Subtask depth limit of 2 levels
- Only breaks tasks marked as complex

#### 6. `pick_task`
Selects the next task to work on using priority-based selection.

**Priority Factors**:
- Task priority level (CRITICAL > HIGH > NORMAL > LOW)
- Parent task progress
- Retry attempts
- Dependencies

#### 7. `execute`
Executes the decided action using the tool system.

**Action Types**:
- `TOOL: tool_name(args)` - Execute a tool
- `DO_NOTHING: reason` - Conscious inaction
- `REFLECT: topic` - Continue thinking

**Execution Flow**:
1. Parse the decision using `parse_action()`
2. Validate against tool schemas
3. Execute via ToolRegistry
4. Capture result or error

#### 8. `evaluate`
Assesses the outcome of the execution.

```python
def evaluate(self, state: ConsciousnessState) -> ConsciousnessState:
    # Evaluate execution success
    success = "error" not in state.get("execution_result", "").lower()

    # Update task status based on result
    if success:
        self.task_list.mark_completed(task_id)
    else:
        self.task_list.mark_failed(task_id, reason)
```

#### 9. `debug`
Triggered when a task fails, generates debugging suggestions.

**Debug Process**:
1. Analyze the error message
2. Check previous attempts
3. Generate recovery suggestions
4. Store debug notes on the task

#### 10. `reflect`
Generates a reflection on what was learned.

**Reflection includes**:
- What was attempted
- What succeeded/failed
- Lessons learned
- Emotional/need state impact

#### 11. `journal`
Creates a journal entry summarizing the cycle.

```python
def journal(self, state: ConsciousnessState) -> ConsciousnessState:
    entry = self.journal.create_entry(
        entry_type=JournalEntryType.EXPERIENCE,
        title=f"Cycle {state['cycle_count']}",
        content=state["current_reflection"],
        mood=self.assess_mood(state),
        cycle_number=state["cycle_count"]
    )
    return {"journal_entry": entry.to_dict()}
```

#### 12. `update_needs`
Adjusts need levels based on cycle events.

**Events processed**:
- `user_helped`, `user_satisfied` → Decrease reliability/excellence
- `error_occurred`, `mistake_made` → Increase reliability
- `learned_something` → Decrease curiosity
- `resource_low` → Increase sustainability

#### 13. `persist`
Saves all state to persistent storage.

**Persisted data**:
- Memory store (new memories)
- Task list state
- Journal entries
- Needs state
- Experience buffer (for training)

#### 14. `sleep`
Determines sleep duration before next cycle.

**Sleep calculation**:
```python
def calculate_sleep(self, needs_state: dict) -> int:
    urgency = max(need["level"] for need in needs_state.values())
    if urgency > 0.8:
        return self.config.min_sleep_seconds  # 30s
    elif urgency > 0.5:
        return (self.config.min_sleep + self.config.max_sleep) // 2
    else:
        return self.config.max_sleep_seconds  # 300s
```

### Conditional Routing

```python
def should_debug(state: ConsciousnessState) -> str:
    """Route to debug if execution failed."""
    result = state.get("execution_result", "")
    if "error" in result.lower() or "failed" in result.lower():
        return "debug"
    return "reflect"

workflow.add_conditional_edges(
    "evaluate",
    should_debug,
    {"debug": "debug", "reflect": "reflect"}
)
```

---

## 3. Memory System

**File**: `mindforge/memory/store.py`

The memory system provides persistent storage with semantic search capabilities.

### Memory Data Structure

```python
@dataclass
class Memory:
    id: str                           # UUID
    content: str                      # Memory content
    memory_type: MemoryType           # Classification
    importance: float                 # 0.0 - 1.0
    timestamp: datetime               # Creation time
    access_count: int = 0             # Times retrieved
    last_accessed: Optional[datetime] = None
    related_to: Optional[str] = None  # Parent memory ID
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Memory Types

```python
class MemoryType(Enum):
    INTERACTION = "interaction"   # User interactions
    THOUGHT = "thought"           # Internal thoughts
    REFLECTION = "reflection"     # Post-action reflections
    LEARNING = "learning"         # Learned facts/skills
    FACT = "fact"                 # Verified factual knowledge
    PREFERENCE = "preference"     # User/system preferences
    PATTERN = "pattern"           # Recognized patterns
    SYSTEM = "system"             # System events
```

### MemoryStore Class

```python
class MemoryStore:
    def __init__(
        self,
        db_path: str = "./data/memories.db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._init_database()
```

### Key Methods

#### `store(memory: Memory) -> str`
Stores a new memory with embedding for semantic search.

```python
def store(self, memory: Memory) -> str:
    embedding = self._generate_embedding(memory.content)
    cursor.execute("""
        INSERT INTO memories
        (id, content, memory_type, importance, timestamp, embedding, ...)
        VALUES (?, ?, ?, ?, ?, ?, ...)
    """, (memory.id, memory.content, memory.memory_type.value, ...))
    return memory.id
```

#### `retrieve(memory_id: str) -> Optional[Memory]`
Retrieves a specific memory by ID.

#### `search(query: str, limit: int = 5, memory_type: Optional[MemoryType] = None) -> List[Memory]`
Semantic search using embedding similarity.

```python
def search(self, query: str, limit: int = 5, ...) -> List[Memory]:
    query_embedding = self._generate_embedding(query)
    # Uses FTS5 for initial filtering, then embedding similarity
    results = self._semantic_search(query_embedding, limit, memory_type)
    # Update access counts
    for memory in results:
        self._update_access(memory.id)
    return results
```

#### `search_by_text(pattern: str) -> List[Memory]`
Full-text search using SQLite FTS5.

```python
def search_by_text(self, pattern: str) -> List[Memory]:
    cursor.execute("""
        SELECT m.* FROM memories m
        JOIN memories_fts fts ON m.id = fts.rowid
        WHERE memories_fts MATCH ?
    """, (pattern,))
```

### Importance Scoring

Memories have importance scores (0.0 - 1.0) that affect:
- Search result ranking
- Consolidation priority
- Decay rate

```python
def calculate_importance(self, content: str, memory_type: MemoryType) -> float:
    base_importance = {
        MemoryType.FACT: 0.9,
        MemoryType.LEARNING: 0.8,
        MemoryType.REFLECTION: 0.7,
        MemoryType.INTERACTION: 0.6,
        MemoryType.THOUGHT: 0.5,
    }.get(memory_type, 0.5)

    # Boost for certain keywords
    if any(kw in content.lower() for kw in ["important", "critical", "remember"]):
        base_importance = min(1.0, base_importance + 0.2)

    return base_importance
```

### Memory Decay

Importance decays over time for infrequently accessed memories:

```python
def apply_decay(self, decay_rate: float = 0.01):
    """Reduce importance of old, rarely-accessed memories."""
    cursor.execute("""
        UPDATE memories
        SET importance = importance * (1 - ?)
        WHERE last_accessed < datetime('now', '-7 days')
        AND importance > 0.1
    """, (decay_rate,))
```

### Database Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    timestamp TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    related_to TEXT,
    tags TEXT,  -- JSON array
    metadata TEXT,  -- JSON object
    embedding BLOB  -- Vector embedding
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid'
);
```

---

## 4. Task Management System

**File**: `mindforge/agent/task_list.py`

The task management system handles hierarchical task tracking with persistence.

### Task Data Structure

```python
@dataclass
class InternalTask:
    id: str                              # UUID
    description: str                      # What to do
    status: TaskStatus                    # Current status
    priority: TaskPriority                # Priority level
    created_at: datetime                  # Creation time
    updated_at: datetime                  # Last update
    parent_id: Optional[str] = None       # Parent task ID
    subtask_ids: List[str] = field(default_factory=list)
    attempts: int = 0                     # Retry count
    max_attempts: int = 3                 # Max retries
    progress_notes: List[str] = field(default_factory=list)
    debug_suggestions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Task Status

```python
class TaskStatus(Enum):
    PENDING = "pending"         # Not started
    IN_PROGRESS = "in_progress" # Currently working
    COMPLETED = "completed"     # Successfully done
    FAILED = "failed"           # Failed after max attempts
    BLOCKED = "blocked"         # Waiting on dependency
```

### Task Priority

```python
class TaskPriority(Enum):
    CRITICAL = 1    # Must do immediately
    HIGH = 2        # Do soon
    NORMAL = 3      # Standard priority
    LOW = 4         # Can wait
```

### PersistentTaskList Class

```python
class PersistentTaskList:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.tasks: Dict[str, InternalTask] = {}
        self._load_from_memory()
```

### Key Methods

#### `add_task(description: str, priority: TaskPriority, parent_id: Optional[str]) -> InternalTask`

```python
def add_task(self, description: str, priority: TaskPriority = TaskPriority.NORMAL,
             parent_id: Optional[str] = None) -> InternalTask:
    # Check for duplicates
    if self._is_duplicate(description):
        logger.warning(f"Duplicate task rejected: {description}")
        return None

    task = InternalTask(
        id=str(uuid.uuid4()),
        description=description,
        status=TaskStatus.PENDING,
        priority=priority,
        parent_id=parent_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    self.tasks[task.id] = task

    # Link to parent if exists
    if parent_id and parent_id in self.tasks:
        self.tasks[parent_id].subtask_ids.append(task.id)

    self._persist_task(task)
    return task
```

#### `get_pending_tasks() -> List[InternalTask]`
Returns all pending tasks sorted by priority.

```python
def get_pending_tasks(self) -> List[InternalTask]:
    pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
    return sorted(pending, key=lambda t: (t.priority.value, t.created_at))
```

#### `pick_next_task() -> Optional[InternalTask]`
Selects the highest priority task, considering:
- Priority level
- Parent task status
- Blocked dependencies

```python
def pick_next_task(self) -> Optional[InternalTask]:
    pending = self.get_pending_tasks()

    for task in pending:
        # Skip if parent is not in progress
        if task.parent_id and self.tasks[task.parent_id].status != TaskStatus.IN_PROGRESS:
            continue
        # Skip blocked tasks
        if task.status == TaskStatus.BLOCKED:
            continue
        return task

    return None
```

#### `mark_completed(task_id: str)`

```python
def mark_completed(self, task_id: str):
    task = self.tasks.get(task_id)
    if task:
        task.status = TaskStatus.COMPLETED
        task.updated_at = datetime.now()
        self._persist_task(task)

        # Check if all subtasks done → complete parent
        if task.parent_id:
            self._check_parent_completion(task.parent_id)
```

#### `mark_failed(task_id: str, reason: str)`

```python
def mark_failed(self, task_id: str, reason: str):
    task = self.tasks.get(task_id)
    if task:
        task.attempts += 1
        task.progress_notes.append(f"Attempt {task.attempts} failed: {reason}")

        if task.attempts >= task.max_attempts:
            task.status = TaskStatus.FAILED
        else:
            task.status = TaskStatus.PENDING  # Will retry

        task.updated_at = datetime.now()
        self._persist_task(task)
```

#### `add_debug_suggestion(task_id: str, suggestion: str)`

```python
def add_debug_suggestion(self, task_id: str, suggestion: str):
    task = self.tasks.get(task_id)
    if task:
        task.debug_suggestions.append(suggestion)
        task.updated_at = datetime.now()
        self._persist_task(task)
```

### Work Log

```python
@dataclass
class WorkLogEntry:
    timestamp: datetime
    task_id: str
    action: str  # "started", "completed", "failed", "retried"
    details: str
```

### Hierarchical Task Trees

```
Task: "Implement authentication"
├── Subtask: "Design auth flow"
│   ├── Subtask: "Research OAuth patterns"
│   └── Subtask: "Create flow diagram"
├── Subtask: "Write auth middleware"
└── Subtask: "Add tests"
```

When all subtasks complete, the parent automatically completes.

---

## 5. Journal System

**File**: `mindforge/agent/journal.py`

The journal system provides structured logging of experiences and reflections.

### Journal Entry Structure

```python
@dataclass
class JournalEntry:
    id: str                           # UUID
    entry_type: JournalEntryType      # Type of entry
    title: str                        # Brief title
    content: str                      # Full content
    mood: Optional[str] = None        # Emotional state
    tags: List[str] = field(default_factory=list)
    cycle_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Entry Types

```python
class JournalEntryType(Enum):
    THOUGHT = "thought"           # Internal musings
    REFLECTION = "reflection"     # Post-action analysis
    LEARNING = "learning"         # New knowledge acquired
    CREATIVE = "creative"         # Creative output
    EXPERIENCE = "experience"     # General experiences
    GRATITUDE = "gratitude"       # Things appreciated
    GOAL = "goal"                 # Goals set
    MEMORY = "memory"             # Important memories
    DREAM = "dream"               # Idle thoughts/dreams
```

### Journal Class

```python
class Journal:
    def __init__(self, journal_path: str = "./data/journal.json"):
        self.journal_path = Path(journal_path)
        self.entries: List[JournalEntry] = []
        self._load()
```

### Key Methods

#### `create_entry()`

```python
def create_entry(
    self,
    entry_type: JournalEntryType,
    title: str,
    content: str,
    mood: Optional[str] = None,
    tags: Optional[List[str]] = None,
    cycle_number: Optional[int] = None
) -> JournalEntry:
    entry = JournalEntry(
        id=str(uuid.uuid4()),
        entry_type=entry_type,
        title=title,
        content=content,
        mood=mood,
        tags=tags or [],
        cycle_number=cycle_number
    )
    self.entries.append(entry)
    self._save()
    return entry
```

#### `get_entries_by_type(entry_type: JournalEntryType) -> List[JournalEntry]`
Filter entries by type.

#### `get_recent_entries(limit: int = 10) -> List[JournalEntry]`
Get most recent entries.

#### `search_entries(query: str) -> List[JournalEntry]`
Search entry content.

### Mood Assessment

```python
def assess_mood(self, state: ConsciousnessState) -> str:
    """Infer mood from state."""
    needs = state.get("needs_state", {})
    error = state.get("last_error")

    if error:
        return "frustrated"

    avg_satisfaction = 1.0 - (sum(n.get("level", 0.5) for n in needs.values()) / len(needs))

    if avg_satisfaction > 0.7:
        return "content"
    elif avg_satisfaction > 0.4:
        return "neutral"
    else:
        return "restless"
```

---

## 6. Needs Regulator

**File**: `mindforge/core/needs.py`

The needs system provides biologically-inspired motivation for autonomous behavior.

### Need Types

```python
class NeedType(Enum):
    SUSTAINABILITY = "sustainability"  # Maintain capability to help
    RELIABILITY = "reliability"        # Be consistently trustworthy
    CURIOSITY = "curiosity"            # Learn to provide better assistance
    EXCELLENCE = "excellence"          # Strive for quality in service
```

### Need Structure

```python
@dataclass
class Need:
    type: NeedType
    base_weight: float           # User-configured priority (0.0 - 1.0)
    current_level: float = 0.5   # Urgency (0.0 = satisfied, 1.0 = urgent)
    description: str = ""
    history: list = field(default_factory=list)

    @property
    def effective_priority(self) -> float:
        """Weight * urgency level."""
        return self.base_weight * (0.5 + self.current_level)
```

### NeedsRegulator Class

```python
class NeedsRegulator:
    def __init__(
        self,
        sustainability_weight: float = 0.25,
        reliability_weight: float = 0.30,
        curiosity_weight: float = 0.25,
        excellence_weight: float = 0.20,
    ):
        # Weights are normalized to sum to 1.0
        self.needs = {
            NeedType.SUSTAINABILITY: Need(type=NeedType.SUSTAINABILITY, ...),
            NeedType.RELIABILITY: Need(type=NeedType.RELIABILITY, ...),
            NeedType.CURIOSITY: Need(type=NeedType.CURIOSITY, ...),
            NeedType.EXCELLENCE: Need(type=NeedType.EXCELLENCE, ...),
        }
```

### Event Processing

```python
def process_event(self, event_type: str, context: dict = None) -> dict:
    """Update needs based on events."""

    event_effects = {
        "user_helped": {
            NeedType.RELIABILITY: -0.1,    # Satisfied
            NeedType.EXCELLENCE: -0.1,
        },
        "error_occurred": {
            NeedType.RELIABILITY: 0.2,     # Increased urgency
            NeedType.SUSTAINABILITY: 0.1,
        },
        "learned_something": {
            NeedType.CURIOSITY: -0.2,      # Satisfied
            NeedType.EXCELLENCE: -0.05,
        },
        # ... more events
    }

    # Apply effects
    for need_type, delta in event_effects[event_type].items():
        if delta > 0:
            self.needs[need_type].increase(delta)
        else:
            self.needs[need_type].satisfy(-delta)

    return self._get_guidance(context)
```

### Guidance Generation

```python
def _get_guidance(self, context: dict) -> dict:
    dominant = self.get_dominant_need()

    focus_suggestions = {
        NeedType.SUSTAINABILITY: "Focus on efficiency and resource management",
        NeedType.RELIABILITY: "Prioritize accuracy and thoroughness",
        NeedType.CURIOSITY: "Explore and ask clarifying questions",
        NeedType.EXCELLENCE: "Aim for exceptional quality and creativity",
    }

    return {
        "dominant_need": dominant.value,
        "suggested_focus": focus_suggestions[dominant],
        "ranking": self.get_priority_ranking(),
    }
```

### Presets

```python
presets = {
    "balanced": (0.25, 0.30, 0.25, 0.20),
    "learning": (0.20, 0.20, 0.40, 0.20),   # High curiosity
    "production": (0.30, 0.40, 0.15, 0.15), # High reliability
    "creative": (0.15, 0.20, 0.30, 0.35),   # High excellence
}
```

---

## 7. KVRM Grounding System

**File**: `mindforge/kvrm/grounding.py`

The Key-Value Response Mapping (KVRM) system ensures zero-hallucination on factual claims.

### Core Concept

KVRM routes claims through verified data stores, only allowing claims that can be traced to verified sources.

### Claim Types

```python
class ClaimType(Enum):
    FACTUAL = "factual"     # Verifiable factual claim
    MEMORY = "memory"       # Reference to past experience
    OPINION = "opinion"     # Subjective (not groundable)
    QUESTION = "question"   # Not a claim
    CREATIVE = "creative"   # Imaginative content
    ACTION = "action"       # Action statement
    UNKNOWN = "unknown"     # Cannot classify
```

### GroundingResult

```python
@dataclass
class GroundingResult:
    original: str                        # Original claim
    claim_type: ClaimType                # Classified type
    grounded: bool                       # Whether verified
    confidence: float                    # Verification confidence
    resolved_content: Optional[ResolvedContent] = None
    key_used: Optional[str] = None       # Verification key
    reason: str = ""                     # Failure reason
    suggestions: List[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.grounded and self.confidence >= 0.9

    @property
    def status(self) -> str:
        if self.is_verified:
            return "VERIFIED"
        elif self.grounded:
            return "GROUNDED"
        elif self.claim_type in (ClaimType.OPINION, ClaimType.CREATIVE):
            return "NOT_APPLICABLE"
        else:
            return "UNVERIFIED"
```

### GroundingRouter

```python
class GroundingRouter:
    def __init__(self, key_resolver: KeyResolver, inference_fn=None):
        self.key_resolver = key_resolver
        self.inference_fn = inference_fn

        # Key patterns for extraction
        self._key_patterns = [
            re.compile(r"mem:[a-z]+:\d{8}:[a-f0-9]+"),  # Memory keys
            re.compile(r"fact:[a-z_]+:[a-z0-9_]+"),    # Fact keys
            re.compile(r"ext:[a-z]+:[^\s]+"),          # External keys
        ]
```

### Grounding Process

```python
def ground(self, text: str, force_type: ClaimType = None) -> GroundingResult:
    # 1. Classify claim type
    claim_type = force_type or self._classify_claim(text)

    # 2. Skip non-groundable claims
    if claim_type in (ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.QUESTION):
        return GroundingResult(
            original=text,
            claim_type=claim_type,
            grounded=False,
            confidence=1.0,
            reason=f"Claim type '{claim_type.value}' does not require grounding"
        )

    # 3. Extract and resolve keys
    keys = self._extract_keys(text)
    for key in keys:
        resolved = self.key_resolver.resolve(key)
        if resolved:
            return GroundingResult(
                original=text,
                claim_type=claim_type,
                grounded=True,
                confidence=resolved.confidence,
                resolved_content=resolved,
                key_used=key
            )

    # 4. Try semantic search fallback
    search_results = self.key_resolver.search(text, limit=3)
    if search_results and search_results[0].confidence >= 0.7:
        return GroundingResult(
            grounded=True,
            confidence=search_results[0].confidence * 0.8,
            metadata={"match_type": "semantic"}
        )

    # 5. Return ungrounded
    return GroundingResult(
        original=text,
        claim_type=claim_type,
        grounded=False,
        confidence=0.0,
        reason="No verified content found"
    )
```

### Key Types

```
mem:thought:20241209:abc123     → Memory key
fact:geography:capital_france   → Fact key
ext:wikipedia:/wiki/Python      → External source key
```

### Claim Classification

```python
def _classify_claim(self, text: str) -> ClaimType:
    text_lower = text.lower().strip()

    # Question indicators
    if any(ind in text_lower for ind in ["?", "what", "how", "why"]):
        return ClaimType.QUESTION

    # Opinion indicators
    if any(ind in text_lower for ind in ["i think", "i believe", "probably"]):
        return ClaimType.OPINION

    # Factual indicators
    if any(ind in text_lower for ind in ["is", "are", "states", "according to"]):
        return ClaimType.FACTUAL

    # Memory references
    if any(phrase in text_lower for phrase in ["i remember", "previously"]):
        return ClaimType.MEMORY

    return ClaimType.UNKNOWN
```

---

## 8. Tool System

**Files**: `mindforge/tools/base.py`, `mindforge/training/tool_formats.py`

### Tool Base Class

```python
class Tool(ABC):
    # Blocked dangerous patterns
    BLOCKED_PATTERNS = [
        "rm -rf /", "sudo rm", "mkfs", "dd if=",
        ":(){:|:&};:",  # Fork bomb
    ]

    def __init__(self, name: str, description: str, requires_confirmation: bool = False):
        self.name = name
        self.description = description
        self.requires_confirmation = requires_confirmation
        self.history: list[ToolResult] = []

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def is_safe(self, command: str) -> tuple[bool, str]:
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in command.lower():
                return False, f"Blocked pattern: {pattern}"
        return True, "OK"
```

### ToolResult

```python
@dataclass
class ToolResult:
    status: ToolStatus          # SUCCESS, ERROR, TIMEOUT, BLOCKED
    output: str                 # Tool output
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS
```

### Available Tools

| Tool | Description | Required Args |
|------|-------------|---------------|
| `shell` | Execute safe shell commands | `command` |
| `filesystem` | Read/write/list files | `operation`, `path` |
| `git` | Git repository operations | `operation` |
| `web` | Fetch web content, search | `operation` |
| `code` | Code analysis and editing | `operation` |
| `kvrm` | Fact verification | `operation` |
| `n8n` | Workflow automation | `operation` |
| `ollama` | Query local LLMs | `operation` |

### Tool Specifications

```python
TOOL_SPECS = {
    "shell": ToolSpec(
        name="shell",
        description="Execute safe shell commands",
        required_args=["command"],
        optional_args=[],
        examples=[
            'TOOL: shell(command="ls")',
            'TOOL: shell(command="pwd")',
        ],
        reward_on_success=1.0,
        reward_on_failure=-0.3,
    ),
    "filesystem": ToolSpec(
        name="filesystem",
        description="Read files and list directories",
        required_args=["operation", "path"],
        optional_args=["content"],
        examples=[
            'TOOL: filesystem(operation="read", path="./README.md")',
            'TOOL: filesystem(operation="list", path=".")',
        ],
    ),
    # ... more tools
}
```

### Action Types

```python
class ActionType(Enum):
    TOOL = "TOOL"              # Execute a tool
    DO_NOTHING = "DO_NOTHING"  # Conscious inaction
    REFLECT = "REFLECT"        # Continue thinking
```

### Action Parsing

```python
def parse_action(response: str) -> ParsedAction:
    response = response.strip()

    if response.upper().startswith("TOOL:"):
        return _parse_tool_action(response[5:].strip())

    if response.upper().startswith("DO_NOTHING:"):
        return ParsedAction(
            action_type=ActionType.DO_NOTHING,
            raw_text=response[11:].strip(),
            is_valid=True
        )

    if response.upper().startswith("REFLECT:"):
        return ParsedAction(
            action_type=ActionType.REFLECT,
            raw_text=response[8:].strip(),
            is_valid=True
        )

    return ParsedAction(
        action_type=ActionType.REFLECT,
        is_valid=False,
        validation_error="Response must start with TOOL:, DO_NOTHING:, or REFLECT:"
    )
```

### Argument Parsing Strategies

The parser uses 6 strategies for robustness:

1. **JSON-style**: Convert to dict and parse
2. **Triple-quoted strings**: Handle `"""content"""`
3. **Escaped quotes**: Handle `\"` in values
4. **Simple patterns**: `key="value"`
5. **Single quotes**: `key='value'`
6. **Unquoted values**: Numbers, booleans

---

## 9. Reward Learning System

**File**: `mindforge/training/reward_calculator.py`

### Reward Components

```python
class RewardType(Enum):
    FORMAT_COMPLIANCE = "format_compliance"     # Correct response format
    EXECUTION_SUCCESS = "execution_success"     # Tool worked
    NEEDS_SATISFACTION = "needs_satisfaction"   # Needs improved
    GOAL_PROGRESS = "goal_progress"             # Progress toward goals
    EXPLORATION_BONUS = "exploration_bonus"     # Tried something new
    CURIOSITY_DRIVE = "curiosity_drive"         # Learning behavior
    COMPETENCE_DRIVE = "competence_drive"       # Skill improvement
```

### Default Weights

```python
DEFAULT_WEIGHTS = {
    "format_compliance": 0.30,   # Following TOOL:/DO_NOTHING:/REFLECT:
    "execution_success": 0.25,   # Tool worked correctly
    "needs_satisfaction": 0.20,  # Needs improved
    "goal_progress": 0.15,       # Made progress on goals
    "exploration": 0.10,         # Tried something new
}
```

### RewardBreakdown

```python
@dataclass
class RewardBreakdown:
    format_compliance: float = 0.0
    execution_success: float = 0.0
    needs_satisfaction: float = 0.0
    goal_progress: float = 0.0
    exploration_bonus: float = 0.0
    curiosity_drive: float = 0.0
    competence_drive: float = 0.0
    total: float = 0.0
    notes: List[str] = field(default_factory=list)
```

### Reward Calculation

```python
class RewardCalculator:
    def calculate_reward(
        self,
        raw_response: str,
        execution_result: str = None,
        execution_success: bool = False,
        needs_before: dict = None,
        needs_after: dict = None,
        thought: str = None,
        cycle_id: int = 0
    ) -> RewardBreakdown:
        breakdown = RewardBreakdown()

        # 1. Format compliance
        parsed = parse_action(raw_response)
        breakdown.format_compliance = 0.5 if parsed.is_valid else -1.0

        # 2. Execution success
        if parsed.action_type == ActionType.TOOL:
            if execution_success:
                spec = TOOL_SPECS.get(parsed.tool_name)
                breakdown.execution_success = spec.reward_on_success  # +1.0
            else:
                breakdown.execution_success = spec.reward_on_failure  # -0.3

        # 3. Needs satisfaction
        breakdown.needs_satisfaction = self._calculate_needs_reward(
            needs_before, needs_after
        )

        # 4. Exploration bonus (first-time tool usage)
        usage_count = self.tool_usage_counts.get(parsed.tool_name, 0)
        if usage_count == 0:
            breakdown.exploration_bonus = 0.5  # Big bonus for new tools

        # 5. Curiosity drive
        if parsed.tool_name in ["shell", "filesystem", "git", "web", "kvrm"]:
            breakdown.curiosity_drive = 0.2  # Info-seeking

        breakdown.compute_total(self.weights)
        return breakdown
```

### Exploration Reward

```python
def _calculate_exploration_reward(self, parsed: ParsedAction) -> float:
    tool_name = parsed.tool_name
    usage_count = self.tool_usage_counts.get(tool_name, 0)

    if usage_count == 0:
        return 0.5           # First time - big bonus
    elif usage_count < 5:
        return 0.3 * exp(-usage_count / 5)  # Decaying bonus
    else:
        return 0.1 * exp(-usage_count / 20)  # Small variety bonus
```

### Competence Tracking

```python
def _calculate_competence_reward(self, parsed: ParsedAction) -> float:
    success_history = self.tool_success_rates.get(parsed.tool_name, [])

    if len(success_history) < 3:
        return 0.0  # Not enough data

    recent_rate = sum(success_history[-10:]) / len(success_history[-10:])

    if len(success_history) >= 10:
        old_rate = sum(success_history[-20:-10]) / 10
        improvement = recent_rate - old_rate
        return max(0, improvement * 0.5)  # Reward improvement

    return 0.0
```

### Goal Tracking

```python
def add_goal(
    self,
    description: str,
    keywords: List[str],
    preferred_tools: List[str] = None,
    priority: float = 1.0
):
    self.active_goals.append({
        "description": description,
        "keywords": keywords,
        "preferred_tools": preferred_tools or [],
        "priority": priority,
        "created_at": datetime.now().isoformat()
    })
```

---

## 10. Configuration

**File**: `config.yaml`

### Identity

```yaml
name: Echo
version: "1.0.0"
```

### Model Configuration

```yaml
model:
  base: "Qwen/Qwen3-8B-Instruct"
  fine_tuned_path: "./models/fine_tuned"
  inference_backend: "mlx"  # mlx, ollama, or llamacpp
  ollama_model_name: "qwen3:8b"
```

### Cycle Timing

```yaml
cycle:
  min_sleep_seconds: 30
  max_sleep_seconds: 300
  cycles_before_consolidation: 50
  cycles_before_mini_finetune: 200
```

### Needs Configuration

```yaml
needs:
  weights:
    sustainability: 0.25
    reliability: 0.30
    curiosity: 0.25
    excellence: 0.20
  initial_levels:
    sustainability: 0.8
    reliability: 0.3
    curiosity: 0.95
    excellence: 0.7
```

### Memory Settings

```yaml
memory:
  chroma_persist_dir: "./chroma_db"
  sqlite_path: "./data/memories.db"
  embedding_model: "all-MiniLM-L6-v2"
  max_memories_per_query: 5
  consolidation_threshold: 100
```

### Tool Configuration

```yaml
tools:
  shell:
    enabled: true
    allowed_commands: ["echo", "ls", "cat", "pwd", "date", "whoami"]
    blocked_commands: ["rm -rf", "sudo", "shutdown", "reboot"]
    timeout_seconds: 30
  filesystem:
    enabled: true
    allowed_paths: ["./", "/tmp"]
    blocked_patterns: ["*.key", "*.pem", ".env"]
  git:
    enabled: true
    allowed_operations: ["status", "log", "diff", "branch"]
    auto_commit: false
```

### Safety Settings

```yaml
safety:
  allow_do_nothing: true
  require_confirmation_destructive: true
  self_update_enabled: false
  max_actions_per_cycle: 5
```

### Training Configuration

```yaml
training:
  enabled: true
  experience_buffer_size: 10000
  min_experiences_for_training: 50

  reward_weights:
    format_compliance: 0.30
    execution_success: 0.25
    needs_satisfaction: 0.20
    goal_progress: 0.15
    exploration: 0.10

  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

---

## 11. Problem Solving & Debugging

### Debug Node Operation

When a task fails, the debug node:

1. **Analyzes the error** from execution result
2. **Reviews previous attempts** stored in task.progress_notes
3. **Generates suggestions** via LLM inference
4. **Stores suggestions** on the task for next attempt

```python
def debug(self, state: ConsciousnessState) -> ConsciousnessState:
    task = state.get("current_task")
    error = state.get("execution_result", "")

    # Build debug context
    context = f"""
    Task: {task['description']}
    Error: {error}
    Previous attempts: {task.get('progress_notes', [])}
    """

    # Generate suggestions
    suggestions = self._inference(DEBUG_PROMPT.format(context=context))

    # Store on task
    self.task_list.add_debug_suggestion(task['id'], suggestions)

    return {"debug_suggestions": suggestions}
```

### Retry Logic

```python
def mark_failed(self, task_id: str, reason: str):
    task = self.tasks[task_id]
    task.attempts += 1
    task.progress_notes.append(f"Attempt {task.attempts}: {reason}")

    if task.attempts >= task.max_attempts:
        task.status = TaskStatus.FAILED
        logger.warning(f"Task permanently failed: {task.description}")
    else:
        task.status = TaskStatus.PENDING  # Will retry
        logger.info(f"Task will retry ({task.attempts}/{task.max_attempts})")
```

### Error Recovery Patterns

1. **Tool Not Found**: Suggest alternative tool or approach
2. **Permission Denied**: Suggest different path or operation
3. **Timeout**: Suggest breaking into smaller operations
4. **Parse Error**: Suggest format corrections
5. **Network Error**: Suggest retry or offline alternatives

### Self-Correction

The system demonstrates self-correction through:

1. **Reflection node**: Analyzes what went wrong
2. **Journal entry**: Documents lessons learned
3. **Memory storage**: Persists error patterns for future reference
4. **Reward signal**: Negative reward reinforces avoidance

---

## Appendix A: File Structure

```
mindforge/
├── agent/
│   ├── __init__.py
│   ├── langgraph_agent.py     # Core consciousness loop
│   ├── task_list.py           # Task management
│   ├── journal.py             # Journal system
│   └── tool_adapter.py        # Tool integration
├── core/
│   ├── __init__.py
│   ├── needs.py               # Needs regulator
│   ├── thought.py             # Thought processing
│   └── mind.py                # Mind abstraction
├── memory/
│   ├── __init__.py
│   ├── store.py               # Memory store
│   ├── vector.py              # Vector embeddings
│   ├── short_term.py          # Short-term memory
│   └── long_term.py           # Long-term memory
├── kvrm/
│   ├── __init__.py
│   ├── grounding.py           # Grounding router
│   ├── resolver.py            # Key resolver
│   ├── key_store.py           # Key stores
│   └── tool.py                # KVRM tool
├── tools/
│   ├── __init__.py
│   ├── base.py                # Tool base classes
│   ├── shell.py               # Shell tool
│   ├── filesystem.py          # Filesystem tool
│   ├── git.py                 # Git tool
│   ├── web.py                 # Web tool
│   └── code.py                # Code tool
├── training/
│   ├── __init__.py
│   ├── reward_calculator.py   # Reward system
│   ├── tool_formats.py        # Tool parsing
│   ├── experience_buffer.py   # Experience storage
│   ├── intrinsic_motivation.py # Intrinsic rewards
│   ├── lora.py                # LoRA training
│   └── pipeline.py            # Training pipeline
└── dashboard/
    ├── __init__.py
    └── app.py                 # Streamlit dashboard
```

## Appendix B: State Flow Diagram

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ load_tasks  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    think    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ground    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │identify_tasks│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │break_subtasks│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  pick_task  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   decide    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   execute   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  evaluate   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │ success?                │
              ▼                         ▼
       ┌──────────┐              ┌──────────┐
       │ reflect  │              │  debug   │
       └────┬─────┘              └────┬─────┘
            │                         │
            └─────────┬───────────────┘
                      │
               ┌──────▼──────┐
               │   journal   │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │update_needs │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │   persist   │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │    sleep    │
               └──────┬──────┘
                      │
                      └──────────────► (next cycle)
```

---

*Documentation generated from MindForge source code - December 11, 2025*

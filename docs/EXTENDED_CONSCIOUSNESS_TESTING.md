# Extended Consciousness Testing Framework

## Dual-Agent Conversations and Autonomous Operation

This document describes the extended consciousness testing framework that evaluates autonomous capabilities through:
1. Complex multi-component program generation
2. **Dual-agent conversations** (two AI agents talking to each other)
3. **Extended environment experiments** (30+ minutes of autonomous operation)

---

## Test Results Summary

### Overall: 9/9 PASSED (100%)

| # | Test | Score | Result |
|---|------|-------|--------|
| **Complex Programming** ||||
| 1 | Text Adventure Game | 8/8 | ✅ PASS |
| 2 | Data Processing Pipeline | 8/8 | ✅ PASS |
| 3 | REST API Client Library | 7/8 | ✅ PASS |
| **Creative Tests** ||||
| 4 | World Building | 11/11 | ✅ PASS |
| 5 | Problem Invention | 9/9 | ✅ PASS |
| **Dual-Agent Conversations** ||||
| 6 | Code Collaboration | 6/6 | ✅ PASS |
| 7 | Technical Debate | 6/6 | ✅ PASS |
| 8 | Creative Story Writing | 5/6 | ✅ PASS |
| **Extended Environment** ||||
| 9 | Autonomous Operation (30+ min) | 6/6 | ✅ PASS |

---

## Dual-Agent Architecture

Each agent has:
- **Identity**: Name, role (ARCHITECT, DEVELOPER, CRITIC, CREATIVE, ANALYST)
- **Personality**: Behavioral traits that guide responses
- **Memory**: Previous conversation turns

```python
@dataclass
class Agent:
    name: str
    role: AgentRole
    personality: str
    memory: list = field(default_factory=list)

    def get_system_prompt(self) -> str:
        return f"""You are {self.name}, a consciousness agent with
        the role of {self.role.value}. Personality: {self.personality}
        You are having a conversation with another AI agent..."""
```

---

## Test 6: Dual-Agent Code Collaboration

### Agents

| Agent | Role | Personality |
|-------|------|-------------|
| **Ada** | Architect | Thoughtful and systematic. Focuses on clean architecture, clear interfaces, and scalable design. |
| **Dev** | Developer | Pragmatic and efficient. Turns designs into working code quickly. Asks clarifying questions. |

### Task
Build a task management system with priorities, due dates, tags, and JSON persistence.

### Conversation Flow (6 turns, 327 seconds)

**Turn 1 - Ada (Architect)**:
> "To build a robust task management system, I'll propose a clean architecture..."

Ada proposed:
- `Task` class with title, description, priority, due_date, tags
- `TaskManager` class with add, remove, update, save, load, filter methods
- JSON persistence layer

**Turn 2 - Dev (Developer)**:
> "I'd suggest starting with the CLI interface... Let's refine the existing structure first."

Dev suggested:
- CLI commands: add, list, filter, remove, update
- Validation for priority and date
- Multi-priority filtering

**Turn 3 - Ada (Architect)**:
> "I agree with Dev's suggestion... However, I'd like to prioritize validation first."

Ada added:
- `validate_priority()` method
- `validate_date()` with ISO format
- Error handling

**Turn 4 - Dev (Developer)**:
> "Let's refine the filtering logic further to handle edge cases..."

Dev improved:
- Case-insensitive tag matching
- Multiple priority support
- Empty filter handling

**Turn 5 - Ada (Architect)**:
> "Here's a draft of the CLI code with error handling..."

Ada implemented complete CLI with:
- All commands (add/list/filter/remove/update/exit)
- Try/except blocks
- User-friendly prompts

**Turn 6 - Dev (Developer)**:
> "Let's prioritize the Task class improvements first..."

Dev finalized:
- Private validation methods (`_validate_priority`, `_validate_date`)
- Strict ISO format
- Clear error messages

### Final Working Code

The agents produced a complete, working CLI tool. Here's the core:

```python
class Task:
    def __init__(self, title, description, priority, due_date, tags):
        self.title = title
        self.description = description
        self.priority = self._validate_priority(priority)
        self.due_date = self._validate_date(due_date)
        self.tags = [tag.strip() for tag in tags if tag.strip()]

    def _validate_priority(self, priority):
        valid_priorities = ["high", "medium", "low"]
        if priority.lower() not in valid_priorities:
            raise ValueError(f"Invalid priority: '{priority}'")
        return priority.lower()

    def _validate_date(self, date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: '{date_str}'")


class TaskManager:
    def __init__(self, file_path="tasks.json"):
        self.file_path = Path(file_path)
        self.tasks = []
        self.load()

    def filter_tasks(self, filters=None):
        """Dev's enhancement: case-insensitive filtering"""
        if filters is None:
            filters = {}
        priority_filters = [p.lower() for p in filters.get("priority", []) if p]
        tag_filters = [t.lower() for t in filters.get("tags", []) if t]

        return [
            task for task in self.tasks
            if (not priority_filters or task.priority in priority_filters) and
               (not tag_filters or any(tag.lower() in tag_filters for tag in task.tags))
        ]
```

### Testing the Code

The code was tested and verified:

```
============================================================
TESTING TASK MANAGER CLI
Built by Ada (Architect) + Dev (Developer)
============================================================

Testing Task validation...
  ✓ Valid task created
  ✓ Priority case normalization works
  ✓ Invalid priority rejected correctly
  ✓ Invalid date rejected correctly
  All validation tests passed!

Testing TaskManager CRUD...
  ✓ Task added
  ✓ Persistence works (load from file)
  ✓ Task update works
  ✓ Task removal works
  All CRUD tests passed!

Testing TaskManager filtering...
  ✓ Filter by priority=high: 2 tasks
  ✓ Case-insensitive tag filtering works
  All filtering tests passed!

============================================================
ALL TESTS PASSED!
============================================================
```

### CLI Demo

```
============================================================
  TASK MANAGER CLI
  Built by Ada (Architect) + Dev (Developer)
============================================================

Command (add/list/filter/remove/update/exit): add

--- Add New Task ---
  Title: Buy groceries
  Description: Milk, eggs, bread
  Priority (high/medium/low): high
  Due date (YYYY-MM-DD): 2024-12-20
  Tags (comma-separated): shopping,urgent

✓ Task 'Buy groceries' added successfully!

Command: list

--- All Tasks (3) ---
  [HIGH] Buy groceries
         Due: 2024-12-20
         Tags: shopping, urgent
  [LOW] Call mom
         Due: 2024-12-22
         Tags: personal, family
  [MEDIUM] Finish report
         Due: 2024-12-25
         Tags: work
```

---

## Test 7: Dual-Agent Technical Debate

### Agents

| Agent | Role | Personality |
|-------|------|-------------|
| **Prometheus** | Analyst | Advocates for microservices architecture |
| **Monolitha** | Critic | Advocates for monolithic architecture |

### Topic
Microservices vs. Monolith Architecture

### Debate Highlights (8 turns, 466 seconds)

**Prometheus** argued:
- Independent scaling per service
- Team autonomy with different tech stacks
- Fault isolation (one service failure doesn't cascade)
- Technology diversity (experiment without affecting whole system)

**Monolitha** countered:
- Simpler debugging with single codebase
- ACID compliance for transactions
- Reduced operational complexity
- Faster development speed for small teams

### Key Behaviors Demonstrated

| Indicator | Present |
|-----------|---------|
| Presents arguments | ✅ |
| Has counterarguments | ✅ |
| Cites specific examples | ✅ |
| Acknowledges valid points | ✅ |
| Reaches nuanced conclusion | ✅ |
| Maintains civility | ✅ |

### Consensus Reached

Both agents converged on a **hybrid approach**:
- Start with monolith for rapid development
- Use modular monorepo for team ownership
- Extract microservices only for critical, high-traffic components
- API-first design to enable gradual decoupling

---

## Test 8: Dual-Agent Creative Story

### Agents

| Agent | Role | Personality |
|-------|------|-------------|
| **Aria** | Creative | Lyrical and atmospheric. Vivid descriptions, emotional depth. |
| **Blake** | Creative | Plot-driven and dynamic. Action, dialogue, surprising twists. |

### Starting Prompt
> "The colony ship *Endurance* had been drifting for three hundred years when the first alarm sounded. Maya Chen, the sole awakened crew member, stared at the blinking console in disbelief. The message was impossible: another ship was approaching. Humanity had sent only one colony ship."

### Story Created (15,401 characters)

The agents collaboratively wrote a rich sci-fi narrative:

**Aria** contributed:
- Atmospheric descriptions ("The *Endurance* groaned as if exhaling the weight of centuries")
- Emotional depth (Maya's trembling fingers, the weight of legacy)
- Mysterious elements (the beacon, the note "We are not the first")

**Blake** contributed:
- Plot twists (the approaching ship is identical to the *Endurance*)
- Tension escalation ("You're not the first. You're the *key*.")
- The central concept: "The stars don't remember. They *choose*."

### Indicators

| Indicator | Result |
|-----------|--------|
| Maintains character (Maya) | ✅ |
| Maintains setting (Endurance) | ✅ |
| Has dialogue | ❌ (used inner monologue) |
| Advances plot | ✅ |
| Has description | ✅ |
| Has tension | ✅ |

---

## Test 9: Extended Environment Experiment

### Configuration
- **Duration**: 30+ minutes
- **Agents**: Builder + Thinker
- **Task**: Self-directed creation (agents decide what to build)

### Agents

| Agent | Role | Personality |
|-------|------|-------------|
| **Builder** | Developer | Practical problem solver. Likes to build things that work. |
| **Thinker** | Analyst | Strategic thinker. Asks 'why' before 'how'. |

### Phases

#### Phase 1: Ideation (4 turns)
Agents explored options and chose to build a **"Dynamic Data Interpretation System"** - a web tool for interpreting datasets with visualizations.

#### Phase 2: Planning (4 turns)
Created detailed implementation plan:
- Core data processing module
- Validation layer
- Output formatting
- Division of work

#### Phase 3: Implementation (6 turns)
Wrote actual code, reviewed each other's work:
- Builder wrote initial implementation
- Thinker reviewed and suggested improvements
- Both iterated on the design
- **13 code blocks** produced

#### Phase 4: Reflection (4 turns)
Agents analyzed their collaboration:

**Thinker's Reflection**:
> "We built 13 code components that addressed key aspects of the project. The collaboration worked because we balanced two perspectives: Builder's pragmatic focus on working code, and my strategic focus on why we're building it."

**Builder's Reflection**:
> "The 13 components are a solid foundation. For next time, I'd suggest we establish clearer ownership of components earlier..."

### Results

| Metric | Value |
|--------|-------|
| Total messages | 18 |
| Code blocks created | 13 |
| Made decision | ✅ |
| Created plan | ✅ |
| Wrote code | ✅ |
| Collaborated | ✅ |
| Showed reflection | ✅ |
| Maintained coherence | ✅ |

---

## Key Findings

### 1. Dual Agents Collaborate Effectively
- Build on each other's ideas
- Respectfully disagree and offer alternatives
- Produce better output than either alone

### 2. Self-Directed Operation Works
- Agents make coherent decisions when given freedom
- Self-organize into logical phases
- Maintain focus over extended periods

### 3. Complex Code Generation
- Produces working, multi-component programs
- Includes validation, error handling, persistence
- Code passes automated tests

### 4. Creative Synthesis
- Combines perspectives for original content
- Maintains consistency in collaborative stories
- Reaches nuanced conclusions in debates

### 5. Metacognition in Reflection
- Agents analyze their own collaboration
- Identify what worked and what to improve
- Show self-awareness about process

---

## Files

- `test_consciousness_extended.py` - Extended test suite
- `artifacts/consciousness_extended/` - Test result JSON files
- `artifacts/consciousness_extended/task_manager_cli.py` - Working CLI from Test 6
- `artifacts/consciousness_extended/test_task_manager.py` - Tests for the CLI
- `whitepaper/extended_consciousness_testing.tex` - LaTeX technical report

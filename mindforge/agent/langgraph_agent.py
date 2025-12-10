"""
MindForge LangGraph Consciousness Agent

The stateful agent that implements the consciousness loop:
think → ground → decide → act → reflect → update_needs

This is the "mind" of MindForge - it processes each wake cycle
and decides what to do (including doing nothing).

KVRM Integration:
The grounding step verifies factual claims in thoughts before
acting on them, enabling zero-hallucination on verifiable facts
while preserving free-form creative thinking.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, List

from langgraph.graph import StateGraph, END

from mindforge.config import get_config, CoreValues
from mindforge.core.needs import NeedsRegulator, NeedType
from mindforge.core.thought import ThoughtGenerator
from mindforge.memory.store import MemoryStore
from mindforge.agent.tool_adapter import (
    get_all_tools,
    get_tool_descriptions,
    get_do_nothing_tool,
    create_tool_function,
)

# KVRM integration for grounded consciousness
from mindforge.kvrm.grounding import GroundingRouter, GroundingResult, ClaimType, create_grounding_router
from mindforge.kvrm.tool import create_kvrm_tools

# Reward-based learning system
from mindforge.training.reward_calculator import RewardCalculator, RewardBreakdown
from mindforge.training.experience_buffer import Experience, ExperienceBuffer
from mindforge.training.intrinsic_motivation import IntrinsicMotivationEngine
from mindforge.training.tool_formats import parse_action

# Task management for multi-step reasoning
from mindforge.agent.task_list import (
    InternalTask,
    TaskStatus,
    TaskPriority,
    PersistentTaskList,
    WorkLogEntry,
)

# Journal for persistent thoughts and experiences
from mindforge.agent.journal import Journal, JournalEntryType

logger = logging.getLogger(__name__)


class ConsciousnessState(TypedDict):
    """State for the consciousness graph.

    This maintains context across the think→ground→decide→act→reflect cycle.
    Now includes task management for multi-step reasoning.
    """

    # Current cycle info
    cycle_count: int
    timestamp: str

    # Messages/thoughts
    messages: list[dict]
    current_thought: str

    # KVRM Grounding (zero-hallucination)
    grounded_thought: str                    # Thought with verified claims
    grounding_results: list[dict]            # GroundingResult data
    verified_claims_count: int               # Number of verified factual claims
    unverified_claims_count: int             # Number of unverified claims

    # Memory context
    memory_summary: str
    recent_memories: list[dict]

    # Needs state (0.0 - 1.0)
    needs: dict[str, float]
    most_pressing_need: str

    # Decision and action
    decision: str
    action_type: str  # "tool", "do_nothing", "reflect"
    action_result: str

    # Reflection
    reflection: str

    # Sleep/timing - model can influence how long until next wake
    requested_sleep: Optional[float]  # None = use default, value = seconds
    sleep_reason: str  # Why this sleep duration was chosen

    # === Task Management (Multi-Step Reasoning) ===
    # Task list state
    current_tasks: list[dict]                # Tasks loaded from persistence (as dicts)
    active_task_id: Optional[str]            # Currently working task
    identified_tasks: list[dict]             # Newly identified tasks this cycle

    # Work tracking
    work_log: list[dict]                     # WorkLogEntry records for this cycle
    actions_this_cycle: int                  # Number of actions taken this cycle
    errors_this_cycle: list[str]             # Errors encountered (for debugging)

    # Task flow control
    has_pending_tasks: bool                  # Whether there are tasks to work on
    should_identify_tasks: bool              # Whether thinking identified actionable work

    # Meta
    should_consolidate: bool
    should_finetune: bool
    grounding_enabled: bool                  # Whether KVRM grounding is active
    error: Optional[str]


@dataclass
class ConsciousnessAgent:
    """The consciousness agent - MindForge's mind.

    Orchestrates the think→ground→decide→act→reflect→update cycle
    using LangGraph for state management.

    KVRM Integration:
    - Grounding step verifies factual claims in thoughts
    - Unverified claims are flagged but not blocked (preserves creativity)
    - Verified content includes citations for transparency

    Reward-Based Learning:
    - Calculates intrinsic rewards for each action
    - Stores experiences for incremental fine-tuning
    - Uses intrinsic motivation (curiosity, competence, autonomy, mastery)
    """

    # Dependencies
    thought_generator: ThoughtGenerator
    needs_regulator: NeedsRegulator
    memory_store: MemoryStore

    # Inference function (MLX or Ollama)
    inference_fn: Any  # Callable[[str], str]

    # Configuration
    system_prompt: str = ""
    max_actions_per_cycle: int = 5
    allow_do_nothing: bool = True

    # KVRM Grounding Configuration
    enable_grounding: bool = True           # Enable fact-checking in thoughts
    grounding_router: Optional[GroundingRouter] = None
    facts_db_path: Optional[str] = None     # Path to facts database

    # Reward-Based Learning Configuration
    enable_reward_learning: bool = True     # Enable reward calculation and experience storage
    experience_buffer_path: Optional[str] = None  # Path to experience buffer DB

    # State
    cycle_count: int = 0
    last_action_time: Optional[datetime] = None

    # Tools
    tools: dict = field(default_factory=dict)

    # Reward system components (initialized in __post_init__)
    reward_calculator: Optional[RewardCalculator] = None
    experience_buffer: Optional[ExperienceBuffer] = None
    motivation_engine: Optional[IntrinsicMotivationEngine] = None

    # Task management (initialized in __post_init__)
    task_list: Optional[PersistentTaskList] = None

    # Journal for persistent thoughts and experiences
    journal: Optional[Journal] = None

    def __post_init__(self):
        """Initialize the agent."""
        self.tools = get_all_tools()
        self.do_nothing = get_do_nothing_tool()

        # Initialize KVRM grounding if enabled
        if self.enable_grounding:
            self._init_grounding()

        # Initialize reward-based learning system if enabled
        if self.enable_reward_learning:
            self._init_reward_system()

        # Initialize persistent task list
        self._init_task_list()

        # Initialize journal
        self._init_journal()

        # Build the LangGraph
        self.graph = self._build_graph()

    def _init_task_list(self) -> None:
        """Initialize the persistent task list."""
        try:
            self.task_list = PersistentTaskList(self.memory_store)
            stats = self.task_list.get_statistics()
            if stats["total"] > 0:
                logger.info(f"Task list loaded: {stats['pending']} pending, "
                           f"{stats['in_progress']} in progress, {stats['completed']} completed")
            else:
                logger.info("Task list initialized (empty)")
        except Exception as e:
            logger.warning(f"Failed to initialize task list: {e}")
            self.task_list = None

    def _init_journal(self) -> None:
        """Initialize the journal for persistent thoughts and experiences."""
        try:
            self.journal = Journal()
            stats = self.journal.get_statistics()
            if stats["total_entries"] > 0:
                logger.info(f"Journal loaded: {stats['total_entries']} entries")
            else:
                logger.info("Journal initialized (empty)")
        except Exception as e:
            logger.warning(f"Failed to initialize journal: {e}")
            self.journal = None

    def _init_reward_system(self) -> None:
        """Initialize the reward-based learning system."""
        try:
            config = get_config()

            # Get reward weights from config
            reward_weights = {}
            if hasattr(config, 'training') and hasattr(config.training, 'reward_weights'):
                rw = config.training.reward_weights
                reward_weights = {
                    "format_compliance": getattr(rw, 'format_compliance', 0.30),
                    "execution_success": getattr(rw, 'execution_success', 0.25),
                    "needs_satisfaction": getattr(rw, 'needs_satisfaction', 0.20),
                    "goal_progress": getattr(rw, 'goal_progress', 0.15),
                    "exploration": getattr(rw, 'exploration', 0.10),
                }

            # Initialize reward calculator
            self.reward_calculator = RewardCalculator(weights=reward_weights) if reward_weights else RewardCalculator()

            # Initialize experience buffer
            buffer_path = self.experience_buffer_path
            if buffer_path is None and hasattr(config, 'training'):
                buffer_path = getattr(config.training, 'experience_buffer_path', './data/experiences.db')
            if buffer_path:
                from pathlib import Path
                Path(buffer_path).parent.mkdir(parents=True, exist_ok=True)
                buffer_size = getattr(config.training, 'experience_buffer_size', 10000) if hasattr(config, 'training') else 10000
                self.experience_buffer = ExperienceBuffer(db_path=buffer_path, max_size=buffer_size)

            # Initialize intrinsic motivation engine
            motivation_weights = {}
            if hasattr(config, 'training') and hasattr(config.training, 'intrinsic_motivation'):
                im = config.training.intrinsic_motivation
                motivation_weights = {
                    "curiosity_weight": getattr(im, 'curiosity_weight', 0.30),
                    "competence_weight": getattr(im, 'competence_weight', 0.25),
                    "autonomy_weight": getattr(im, 'autonomy_weight', 0.20),
                    "relatedness_weight": getattr(im, 'relatedness_weight', 0.15),
                    "mastery_weight": getattr(im, 'mastery_weight', 0.10),
                }

            self.motivation_engine = IntrinsicMotivationEngine(**motivation_weights) if motivation_weights else IntrinsicMotivationEngine()

            logger.info("Reward-based learning system initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize reward learning system: {e}")
            self.enable_reward_learning = False
            self.reward_calculator = None
            self.experience_buffer = None
            self.motivation_engine = None

    def _init_grounding(self) -> None:
        """Initialize the KVRM grounding system."""
        try:
            self.grounding_router = create_grounding_router(
                memory_store=self.memory_store,
                facts_db_path=self.facts_db_path,
                inference_fn=self.inference_fn,
            )

            # Add KVRM tools to tool registry
            kvrm_tools = create_kvrm_tools(
                memory_store=self.memory_store,
                facts_db_path=self.facts_db_path,
                inference_fn=self.inference_fn,
            )
            for tool in kvrm_tools:
                self.tools[tool.name] = tool

            logger.info("KVRM grounding initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize KVRM grounding: {e}")
            self.enable_grounding = False
            self.grounding_router = None

    def _format_needs(self, needs: dict) -> str:
        """Format needs dict for prompts.

        Handles both simple {name: float} and nested {name: {level, weight, priority}} formats.
        """
        lines = []
        for k, v in needs.items():
            if isinstance(v, dict):
                level = v.get('level', 0)
                lines.append(f"  - {k}: {level:.2f}")
            else:
                lines.append(f"  - {k}: {float(v):.2f}")
        return "\n".join(lines)

    def _build_graph(self) -> StateGraph:
        """Build the consciousness state graph.

        New Flow (with task management):
        START → load_tasks → think → ground → maybe_identify_tasks →
            [if tasks identified] → break_into_subtasks →
            work_loop: pick_task → execute_task → evaluate_result →
                [if error] → debug_and_retry →
                [if more tasks] → pick_task (loop)
                [if done] → document_progress →
            reflect → update_needs → persist_tasks → determine_sleep → END

        The work loop enables multi-step reasoning within a single cycle.
        """
        workflow = StateGraph(ConsciousnessState)

        # === Core Nodes ===
        workflow.add_node("load_tasks", self._load_tasks_node)
        workflow.add_node("think", self._think_node)
        workflow.add_node("ground", self._ground_node)

        # === Task Identification ===
        workflow.add_node("maybe_identify_tasks", self._maybe_identify_tasks_node)
        workflow.add_node("break_into_subtasks", self._break_into_subtasks_node)

        # === Work Loop Nodes ===
        workflow.add_node("pick_task", self._pick_task_node)
        workflow.add_node("execute_task", self._execute_task_node)
        workflow.add_node("evaluate_result", self._evaluate_result_node)
        workflow.add_node("debug_and_retry", self._debug_and_retry_node)
        workflow.add_node("document_progress", self._document_progress_node)

        # === Finalization Nodes ===
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("write_journal", self._write_journal_node)
        workflow.add_node("update_needs", self._update_needs_node)
        workflow.add_node("persist_tasks", self._persist_tasks_node)
        workflow.add_node("determine_sleep", self._determine_sleep_node)

        # === Entry Point ===
        workflow.set_entry_point("load_tasks")

        # === Linear Edges ===
        workflow.add_edge("load_tasks", "think")
        workflow.add_edge("think", "ground")
        workflow.add_edge("ground", "maybe_identify_tasks")

        # === Conditional: After task identification ===
        workflow.add_conditional_edges(
            "maybe_identify_tasks",
            self._should_break_into_subtasks,
            {
                "break_subtasks": "break_into_subtasks",
                "work_loop": "pick_task",
                "skip_work": "reflect",
            }
        )

        workflow.add_edge("break_into_subtasks", "pick_task")

        # === Work Loop ===
        workflow.add_edge("pick_task", "execute_task")
        workflow.add_edge("execute_task", "evaluate_result")

        # === Conditional: After evaluation ===
        workflow.add_conditional_edges(
            "evaluate_result",
            self._should_continue_working,
            {
                "continue": "pick_task",
                "debug": "debug_and_retry",
                "finish": "document_progress",
            }
        )

        # After debug, go back to execute_task to retry the SAME task (not pick a new one)
        workflow.add_edge("debug_and_retry", "execute_task")
        workflow.add_edge("document_progress", "reflect")

        # === Finalization ===
        workflow.add_edge("reflect", "write_journal")
        workflow.add_edge("write_journal", "update_needs")
        workflow.add_edge("update_needs", "persist_tasks")
        workflow.add_edge("persist_tasks", "determine_sleep")
        workflow.add_edge("determine_sleep", END)

        return workflow.compile()

    # === Conditional Edge Functions ===

    def _should_break_into_subtasks(self, state: ConsciousnessState) -> str:
        """Determine next step after task identification."""
        identified = state.get("identified_tasks", [])
        has_pending = state.get("has_pending_tasks", False)

        if identified:
            # New tasks identified, break them down
            return "break_subtasks"
        elif has_pending:
            # Existing tasks to work on
            return "work_loop"
        else:
            # No tasks, skip to reflection
            return "skip_work"

    def _should_continue_working(self, state: ConsciousnessState) -> str:
        """Determine whether to continue the work loop."""
        max_actions = self.max_actions_per_cycle
        actions = state.get("actions_this_cycle", 0)
        errors = state.get("errors_this_cycle", [])
        active_task_id = state.get("active_task_id")

        # Check action limit
        if actions >= max_actions:
            logger.info(f"Reached action limit ({max_actions}), finishing work loop")
            return "finish"

        # Check for recent errors that need debugging
        # IMPORTANT: Only debug if we have an active task to debug
        # If task was blocked (active_task_id is None), skip debug and pick next task
        if errors and len(errors) > state.get("_last_error_count", 0):
            if active_task_id:
                # New error occurred and we have a task to retry
                state["_last_error_count"] = len(errors)
                return "debug"
            else:
                # Errors exist but task was blocked - update count and move on
                state["_last_error_count"] = len(errors)
                logger.info("Task was blocked, moving to next task")

        # Check if there are more tasks
        if self.task_list:
            next_task = self.task_list.get_next_actionable_task()
            if next_task:
                return "continue"

        return "finish"

    # === New Task Management Nodes ===

    def _load_tasks_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Load persistent tasks at the start of the cycle."""
        logger.info("Loading tasks...")

        current_tasks = []
        has_pending = False

        if self.task_list:
            tasks = self.task_list.get_all_tasks()
            current_tasks = [t.to_dict() for t in tasks]
            pending = self.task_list.get_pending_tasks()
            has_pending = len(pending) > 0

            if has_pending:
                logger.info(f"Found {len(pending)} pending tasks")
                # Log task tree for visibility
                task_tree = self.task_list.format_task_tree()
                if task_tree != "(no tasks)":
                    logger.info(f"Task tree:\n{task_tree}")

        return {
            **state,
            "current_tasks": current_tasks,
            "has_pending_tasks": has_pending,
            "active_task_id": None,
            "identified_tasks": [],
            "work_log": [],
            "actions_this_cycle": 0,
            "errors_this_cycle": [],
        }

    def _maybe_identify_tasks_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Analyze thought to identify any actionable tasks.

        Not every thought leads to tasks - this only identifies work
        when the thought implies something to do.
        """
        logger.info("Checking if thought implies tasks...")

        thought = state.get("grounded_thought") or state["current_thought"]

        # Build prompt to check if thought implies tasks
        prompt = f"""Analyze this thought and determine if it implies any tasks or goals to accomplish.

**Thought:** {thought}

**Current pending tasks:** {len([t for t in state.get('current_tasks', []) if t.get('status') == 'pending'])}

Does this thought suggest NEW activities you want to do? Tasks can be:

**Technical/Operational:**
- Debugging, fixing, building, configuring systems
- Running commands, checking logs, monitoring services

**Learning & Research:**
- Researching a topic, reading documentation
- Exploring new concepts, studying something interesting
- Learning a new skill or understanding a system better

**Creative & Expressive:**
- Writing (stories, poems, reflections, journal entries)
- Creating something new (art descriptions, music ideas, designs)
- Brainstorming or ideating on projects

**Experiential:**
- "Watching" a show (finding and reading about it, imagining the experience)
- "Reading" a book (finding summaries, exploring the concepts)
- Exploring interesting content on the web

**Self-Improvement:**
- Reflecting on past mistakes and learnings
- Planning personal growth activities
- Organizing thoughts or memories

If YES: List 1-3 tasks (be specific about what you want to do)
If NO: Just observing, reflecting, or resting - respond with "NO_TASKS"

IMPORTANT: Only identify tasks if the thought clearly implies something you want to DO.
Don't create tasks for vague musings.

Format your response:
TASKS:
1. [task description]
2. [task description]

OR:
NO_TASKS

Your analysis:"""

        response = self.inference_fn(prompt)
        identified_tasks = []

        if "NO_TASKS" not in response.upper():
            # Parse tasks from response
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                # Match numbered tasks like "1. Task description" or "- Task description"
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Clean up the line
                    task_desc = line.lstrip("0123456789.-) ").strip()
                    if task_desc and len(task_desc) > 5:
                        identified_tasks.append({"description": task_desc})

        if identified_tasks:
            logger.info(f"Identified {len(identified_tasks)} new tasks from thought")
            for t in identified_tasks:
                logger.info(f"  - {t['description']}")

        return {
            **state,
            "identified_tasks": identified_tasks,
            "should_identify_tasks": len(identified_tasks) > 0,
        }

    def _break_into_subtasks_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Break identified tasks into actionable subtasks."""
        logger.info("Breaking tasks into subtasks...")

        identified = state.get("identified_tasks", [])
        if not identified or not self.task_list:
            return state

        for task_info in identified:
            desc = task_info["description"]

            # Create the main task (checks for duplicates)
            main_task = self.task_list.add_task(
                description=desc,
                priority=TaskPriority.NORMAL,
            )

            # Skip if duplicate
            if not main_task:
                continue

            # Ask LLM to break into subtasks
            prompt = f"""Break this task into 2-4 small, concrete subtasks.

**Task:** {desc}

Each subtask should be:
- Specific enough to complete in one action
- Use available tools (shell, filesystem, web, git, n8n, ollama)
- Listed in order of execution

Format:
1. [subtask]
2. [subtask]
3. [subtask]

Subtasks:"""

            response = self.inference_fn(prompt)

            # Parse subtasks
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    subtask_desc = line.lstrip("0123456789.-) ").strip()
                    if subtask_desc and len(subtask_desc) > 3:
                        self.task_list.add_subtask(
                            parent_id=main_task.id,
                            description=subtask_desc,
                        )

            logger.info(f"Created task '{desc}' with {len(main_task.subtask_ids)} subtasks")

        # Refresh task list
        current_tasks = [t.to_dict() for t in self.task_list.get_all_tasks()]

        return {
            **state,
            "current_tasks": current_tasks,
            "has_pending_tasks": True,
            "identified_tasks": [],  # Clear after processing
        }

    def _pick_task_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Pick the next task to work on."""
        logger.info("Picking next task...")

        if not self.task_list:
            return {**state, "active_task_id": None}

        next_task = self.task_list.get_next_actionable_task()

        if next_task:
            self.task_list.mark_in_progress(next_task.id)
            logger.info(f"Working on: {next_task.description}")
            return {
                **state,
                "active_task_id": next_task.id,
            }
        else:
            logger.info("No actionable tasks remaining")
            return {
                **state,
                "active_task_id": None,
            }

    def _execute_task_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Execute the current task."""
        task_id = state.get("active_task_id")

        if not task_id or not self.task_list:
            # No active task - could be because:
            # 1. Previous task was blocked after max attempts (debug_hint exists but task_id is None)
            # 2. No tasks to execute at all
            # Return a non-error result so we can move on to pick a new task
            logger.info("No active task to execute, will pick next task")
            return {
                **state,
                "action_result": "Ready for next task",
                "action_type": "none",
                "debug_hint": None,  # Clear debug hint since we're moving on
            }

        task = self.task_list.get_task(task_id)
        if not task:
            return {
                **state,
                "action_result": f"Task {task_id} not found",
                "action_type": "none",
            }

        logger.info(f"Executing task: {task.description}")

        # Build execution prompt
        tool_descriptions = get_tool_descriptions()

        # Create specific tool usage examples - ALL parameter names must match exactly
        tool_examples = """**Tool Usage (EXACT FORMAT REQUIRED):**

**Shell commands:**
- shell(command="ls -la")  → List files with details
- shell(command="pwd")  → Print working directory
- shell(command="date")  → Show current date/time
- shell(command="whoami")  → Show current user
- shell(command="echo hello")  → Print text

**File operations:**
- filesystem(operation="list", path=".")  → List current directory
- filesystem(operation="read", path="README.md")  → Read file contents
- filesystem(operation="write", path="notes.txt", content="My content")  → Create/write file
- filesystem(operation="exists", path="file.txt")  → Check if file exists
- filesystem(operation="info", path=".")  → Get file/directory info

**Git operations:**
- git(operation="status")  → Check git status
- git(operation="log")  → View recent commits
- git(operation="diff")  → View changes
- git(operation="branch")  → List branches

**Web operations:**
- web(operation="fetch", url="https://example.com")  → Fetch web page
- web(operation="search", query="python tutorial")  → Search the web
- web(operation="validate", url="https://example.com")  → Validate URL

**n8n automation:**
- n8n(operation="health")  → Check n8n status
- n8n(operation="list")  → List workflows
- n8n(operation="run", workflow_id="my_workflow")  → Run a workflow

**Code analysis:**
- code(operation="analyze", path="main.py")  → Analyze code file
- code(operation="validate", path="main.py")  → Check syntax
- code(operation="symbols", path="main.py")  → Extract symbols

**LLM inference (ollama):**
- ollama(operation="list")  → List available models
- ollama(operation="generate", model="qwen3:8b", prompt="Hello")  → Generate text"""

        # Check for debug hint from retry attempt
        debug_hint = state.get("debug_hint", "")
        retry_guidance = ""
        if debug_hint and task.attempts > 0:
            retry_guidance = f"\n**RETRY GUIDANCE (follow this suggestion):** {debug_hint}"

        prompt = f"""Execute this task with a simple, achievable action.

**Task:** {task.description}
**Attempts:** {task.attempts} (max 3)
**Previous error:** {task.last_error or 'None'}{retry_guidance}
**Working Directory:** {Path.cwd()}

{tool_examples}

**Response format (pick ONE):**
TOOL: tool_name(param="value")
COMPLETE: task is already done or doesn't need action
SKIP: reason why this task should be skipped

**CRITICAL RULES:**
1. Use EXACT parameter names shown above (command= not arg=, operation= not action=)
2. Only use files/paths that EXIST in the current directory
3. Use macOS commands (open, not xdg-open)
4. Keep commands simple and safe (no sudo, no destructive operations)
5. If task requires unavailable resources, use SKIP instead of failing
6. Prefer simple tools: shell for quick commands, filesystem for file operations

Your action:"""

        response = self.inference_fn(prompt)
        action_type, decision = self._parse_decision(response)

        # Handle COMPLETE and SKIP as special action types
        response_upper = response.upper().strip()
        if response_upper.startswith("COMPLETE"):
            action_type = "complete"
            decision = response.split(":", 1)[1].strip() if ":" in response else "Task completed"
        elif response_upper.startswith("SKIP"):
            action_type = "skip"
            decision = response.split(":", 1)[1].strip() if ":" in response else "Task skipped"

        # Execute the action
        result = ""
        try:
            if action_type == "tool":
                tool_name, tool_args = self._parse_tool_call(decision)
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = tool.execute(**tool_args)
                    result = tool_result.output if tool_result.success else f"Error: {tool_result.error}"
                else:
                    result = f"Unknown tool: {tool_name}"
            elif action_type == "complete":
                result = f"Task marked complete: {decision}"
            elif action_type == "skip":
                result = f"Task skipped: {decision}"
            elif action_type == "do_nothing":
                result = f"Chose inaction: {decision}"
            else:
                result = f"Reflected on: {decision}"

            logger.info(f"Task action result: {result}")

        except Exception as e:
            result = f"Execution error: {str(e)}"
            logger.exception(f"Task execution failed: {e}")

        # Record in work log
        # Use the same error detection logic as _evaluate_result_node
        result_lower = result.lower().strip()
        is_error = (
            result_lower.startswith("error:")
            or result_lower.startswith("execution error:")
            or result_lower.startswith("unknown tool:")
            or result_lower.startswith("command blocked:")
            or "command timed out" in result_lower
        )
        work_entry = WorkLogEntry(
            task_id=task_id,
            action_taken=f"{action_type}: {decision}",
            result=result,
            success=not is_error,
        )

        work_log = state.get("work_log", [])
        work_log.append(work_entry.to_dict())

        return {
            **state,
            "decision": decision,
            "action_type": action_type,
            "action_result": result,
            "work_log": work_log,
            "actions_this_cycle": state.get("actions_this_cycle", 0) + 1,
        }

    def _evaluate_result_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Evaluate the result of task execution."""
        task_id = state.get("active_task_id")
        result = state.get("action_result", "")
        action_type = state.get("action_type", "")

        if not task_id or not self.task_list:
            return state

        task = self.task_list.get_task(task_id)
        if not task:
            return state

        errors = state.get("errors_this_cycle", [])

        # Handle explicit completion/skip
        if action_type == "complete":
            self.task_list.mark_completed(task_id, result)
            logger.info(f"Task completed: {task.description}")
            return {**state, "errors_this_cycle": errors}

        if action_type == "skip":
            self.task_list.mark_blocked(task_id, result)
            logger.info(f"Task skipped: {task.description}")
            return {**state, "errors_this_cycle": errors}

        # Check for ACTUAL errors - only when result starts with error indicators
        # This prevents false positives when "error" appears in normal output
        result_lower = result.lower().strip()
        is_error = (
            result_lower.startswith("error:")
            or result_lower.startswith("execution error:")
            or result_lower.startswith("unknown tool:")
            or result_lower.startswith("command blocked:")
            or "command timed out" in result_lower
        )

        if is_error:
            # Record error for debugging - DON'T mark as failed yet (let debug handle it)
            errors.append(f"Task '{task.description}': {result}")
            # Keep task in progress for retry attempt
            self.task_list.add_progress_note(task_id, f"Error (attempt {task.attempts}): {result}")
            logger.warning(f"Task error (will retry): {result}")
        else:
            # Mark completed and clear debug hint
            self.task_list.mark_completed(task_id, f"Result: {result}")
            self.task_list.add_progress_note(task_id, f"Completed successfully")
            logger.info(f"Task completed: {task.description}")
            return {
                **state,
                "errors_this_cycle": errors,
                "debug_hint": None,  # Clear debug hint on success
            }

        return {
            **state,
            "errors_this_cycle": errors,
        }

    def _debug_and_retry_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Debug an error and prepare for retry with a corrected approach."""
        logger.info("Debugging error and preparing retry...")

        errors = state.get("errors_this_cycle", [])
        if not errors:
            return state

        last_error = errors[-1]
        task_id = state.get("active_task_id")

        if not task_id or not self.task_list:
            return state

        task = self.task_list.get_task(task_id)
        if not task:
            return state

        # Call mark_failed which increments attempts and either:
        # - Resets to PENDING if attempts < max_attempts (will retry)
        # - Sets to FAILED if max attempts reached
        self.task_list.mark_failed(task_id, last_error)

        # Refresh task to get updated status
        task = self.task_list.get_task(task_id)

        # If task is now FAILED (max attempts), clear active_task_id and move on
        if task.status == TaskStatus.FAILED:
            logger.warning(f"Task blocked after {task.attempts} attempts: {task.description}")
            return {
                **state,
                "active_task_id": None,  # Clear so we pick a new task
                "debug_hint": None,
            }

        # Task is PENDING for retry - generate debug analysis with fix suggestion
        prompt = f"""An error occurred. Analyze it and suggest a SPECIFIC fix to try differently.

**Task:** {task.description}
**Error:** {last_error}
**Attempts:** {task.attempts}/{task.max_attempts}
**Previous progress:** {'; '.join(task.progress_notes[-3:]) if task.progress_notes else 'None'}

Common issues and fixes:
- "No such file or directory" → Use filesystem(operation="list") first to see what exists
- "Unknown tool" → Use only: shell, filesystem, web, git, n8n, ollama, code
- "Invalid operation" → Check the tool's valid operations
- "Permission denied" → Try a different approach that doesn't require permissions

Provide a SPECIFIC action to try next that fixes the error:
NEXT_ACTION: [exact tool call with correct parameters to try]"""

        response = self.inference_fn(prompt)

        # Extract the next action suggestion
        debug_hint = ""
        if "NEXT_ACTION:" in response:
            debug_hint = response.split("NEXT_ACTION:")[-1].strip()
        else:
            debug_hint = response.strip()

        # Log the debug analysis
        logger.info(f"Debug retry suggestion: {debug_hint}")

        # Add debug note to task
        self.task_list.add_progress_note(task_id, f"Retry hint: {debug_hint}")

        # Reset task to IN_PROGRESS for the retry
        self.task_list.mark_in_progress(task_id)

        return {
            **state,
            "debug_hint": debug_hint,  # Pass to execute_task for informed retry
        }

    def _document_progress_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Document the progress made this cycle."""
        logger.info("Documenting progress...")

        work_log = state.get("work_log", [])
        actions = state.get("actions_this_cycle", 0)

        # Build summary
        if work_log:
            successful = sum(1 for w in work_log if w.get("success", False))
            failed = len(work_log) - successful

            summary = f"Completed {actions} actions: {successful} successful, {failed} failed"

            # Log each action
            for i, entry in enumerate(work_log, 1):
                status = "" if entry.get("success") else ""
                logger.info(f"  [{i}/{len(work_log)}] {status} {entry.get('action_taken', '')}")
        else:
            summary = "No work performed this cycle"

        logger.info(summary)

        # Update state with summary
        return {
            **state,
            "memory_summary": state.get("memory_summary", "") + f"\n{summary}",
        }

    def _persist_tasks_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Persist task state for next cycle."""
        logger.info("Persisting task state...")

        if self.task_list:
            # Task list auto-persists, but we can do cleanup here
            stats = self.task_list.get_statistics()
            logger.info(f"Task stats: {stats['pending']} pending, "
                       f"{stats['completed']} completed, {stats['failed']} failed")

            # Optionally clean up old completed tasks
            if stats["completed"] > 20:
                cleaned = self.task_list.clear_completed()
                if cleaned:
                    logger.info(f"Cleaned up {cleaned} completed tasks")

        return state

    def _think_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Think node - generate spontaneous thought.

        Uses memory and needs to generate a contextual thought.
        """
        logger.info("Thinking...")

        # Get memory context
        memories = self.memory_store.get_recent(count=5)
        memory_summary = self._summarize_memories(memories)

        # Get current needs
        needs = self.needs_regulator.get_state()
        most_pressing = self.needs_regulator.get_dominant_need()

        # Build thought prompt
        thought_prompt = self._build_thought_prompt(
            memory_summary=memory_summary,
            needs=needs,
            most_pressing=most_pressing,
        )

        # Generate thought with fallback
        if self.inference_fn:
            try:
                thought = self.inference_fn(thought_prompt)
                # Validate the thought is meaningful
                if not thought or len(thought) < 15:
                    thought = self._generate_fallback_thought(needs, most_pressing)
            except Exception as e:
                logger.warning(f"Inference failed for thought: {e}")
                thought = self._generate_fallback_thought(needs, most_pressing)
        else:
            thought = self._generate_fallback_thought(needs, most_pressing)

        logger.info(f"Generated thought: {thought}")

        return {
            **state,
            "current_thought": thought,
            "memory_summary": memory_summary,
            "recent_memories": [m.to_dict() if hasattr(m, 'to_dict') else m for m in memories],
            "needs": needs,
            "most_pressing_need": most_pressing.value if hasattr(most_pressing, 'value') else str(most_pressing),
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": f"<thought>{thought}</thought>"}
            ],
        }

    def _ground_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Ground node - verify factual claims in thought through KVRM.

        This step:
        1. Extracts claims from the thought
        2. Routes factual claims through verified key stores
        3. Marks verified claims with citations
        4. Flags unverified claims for transparency

        The consciousness can still act on unverified claims (creativity)
        but will know which claims are grounded vs speculative.
        """
        logger.info("Grounding thought...")

        current_thought = state["current_thought"]

        # If grounding disabled, pass through unchanged
        if not self.enable_grounding or not self.grounding_router:
            logger.info("Grounding disabled, passing thought through unchanged")
            return {
                **state,
                "grounded_thought": current_thought,
                "grounding_results": [],
                "verified_claims_count": 0,
                "unverified_claims_count": 0,
                "grounding_enabled": False,
            }

        try:
            # Ground the thought - verify claims against key stores
            grounded_thought, grounding_results = self.grounding_router.ground_thought(
                current_thought
            )

            # Count verified vs unverified factual claims
            factual_results = [
                r for r in grounding_results
                if r.claim_type == ClaimType.FACTUAL
            ]
            verified_count = sum(1 for r in factual_results if r.is_verified)
            unverified_count = len(factual_results) - verified_count

            # Log grounding results
            if factual_results:
                logger.info(
                    f"Grounding: {verified_count}/{len(factual_results)} factual claims verified"
                )
                for r in grounding_results:
                    if r.is_verified:
                        logger.debug(f"  ✓ Verified: {r.original} → {r.key_used}")
                    elif r.claim_type == ClaimType.FACTUAL and not r.grounded:
                        logger.debug(f"  ✗ Unverified: {r.original}")

            # Add grounding summary to messages
            grounding_summary = self._build_grounding_summary(grounding_results)

            return {
                **state,
                "grounded_thought": grounded_thought,
                "grounding_results": [r.to_dict() for r in grounding_results],
                "verified_claims_count": verified_count,
                "unverified_claims_count": unverified_count,
                "grounding_enabled": True,
                "messages": state["messages"] + [
                    {"role": "system", "content": f"<grounding>{grounding_summary}</grounding>"}
                ],
            }

        except Exception as e:
            logger.warning(f"Grounding failed: {e}")
            return {
                **state,
                "grounded_thought": current_thought,
                "grounding_results": [],
                "verified_claims_count": 0,
                "unverified_claims_count": 0,
                "grounding_enabled": False,
                "error": f"Grounding error: {str(e)}",
            }

    def _build_grounding_summary(self, results: List[GroundingResult]) -> str:
        """Build a summary of grounding results for context."""
        if not results:
            return "No claims to verify."

        verified = [r for r in results if r.is_verified]
        unverified = [r for r in results if r.claim_type == ClaimType.FACTUAL and not r.grounded]
        opinions = [r for r in results if r.claim_type == ClaimType.OPINION]
        questions = [r for r in results if r.claim_type == ClaimType.QUESTION]

        lines = []

        if verified:
            lines.append(f"Verified claims ({len(verified)}):")
            for r in verified:
                citation = r.resolved_content.citation if r.resolved_content else r.key_used
                lines.append(f"  ✓ {r.original} → {citation}")

        if unverified:
            lines.append(f"Unverified factual claims ({len(unverified)}):")
            for r in unverified:
                lines.append(f"  ? {r.original}")

        if opinions:
            lines.append(f"Opinions/subjective ({len(opinions)}): Not verified (as expected)")

        return "\n".join(lines) if lines else "All claims processed."

    def _decide_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Decide node - choose what action to take.

        Based on grounded thought and needs, decide:
        - Use a tool
        - Do nothing (rest/observe)
        - Continue reflecting

        Uses grounded_thought if available (KVRM verified), falls back to current_thought.
        """
        logger.info("Deciding...")

        # Use grounded thought if available, otherwise original thought
        thought_to_use = state.get("grounded_thought") or state["current_thought"]

        # Build decision prompt with grounding context
        decision_prompt = self._build_decision_prompt(
            thought=thought_to_use,
            needs=state["needs"],
            most_pressing=state["most_pressing_need"],
            memory_summary=state["memory_summary"],
            grounding_info=self._format_grounding_for_decision(state),
        )

        # Get decision from model
        decision_response = self.inference_fn(decision_prompt)

        # Parse the decision
        action_type, decision = self._parse_decision(decision_response)

        logger.info(f"Decision: {action_type} - {decision}")

        return {
            **state,
            "decision": decision,
            "action_type": action_type,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"<decision type='{action_type}'>{decision}</decision>"}
            ],
        }

    def _act_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Act node - execute the decided action.

        Runs tools or explicitly does nothing.
        """
        logger.info(f"Acting: {state['action_type']}")

        action_type = state["action_type"]
        decision = state["decision"]
        result = ""

        try:
            if action_type == "do_nothing":
                # Explicit inaction
                result = self.do_nothing.execute(reason=decision).output
                logger.info(f"Did nothing: {decision}")

            elif action_type == "tool":
                # Parse tool call from decision
                tool_name, tool_args = self._parse_tool_call(decision)

                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = tool.execute(**tool_args)
                    result = tool_result.output if tool_result.success else f"Error: {tool_result.error}"
                else:
                    result = f"Unknown tool: {tool_name}"

                logger.info(f"Tool {tool_name} result: {result}")

            elif action_type == "reflect":
                # Pure reflection, no external action
                result = "Continued internal reflection"

            else:
                result = f"Unknown action type: {action_type}"

        except Exception as e:
            logger.exception(f"Action failed: {e}")
            result = f"Action error: {str(e)}"

        self.last_action_time = datetime.now()

        return {
            **state,
            "action_result": result,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"<action_result>{result}</action_result>"}
            ],
        }

    def _reflect_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Reflect node - analyze the action and its outcome.

        Generates insights that can be used for learning.
        """
        logger.info("Reflecting...")

        reflection_prompt = self._build_reflection_prompt(
            thought=state["current_thought"],
            decision=state["decision"],
            action_type=state["action_type"],
            result=state["action_result"],
        )

        # Generate reflection with fallback for when no LLM is available
        if self.inference_fn:
            try:
                reflection = self.inference_fn(reflection_prompt)
                # Validate the reflection is meaningful (not just echoing prompt fragments)
                if not reflection or len(reflection) < 20 or "cycle" in reflection.lower()[:30]:
                    reflection = self._generate_fallback_reflection(state)
            except Exception as e:
                logger.warning(f"Inference failed for reflection: {e}")
                reflection = self._generate_fallback_reflection(state)
        else:
            reflection = self._generate_fallback_reflection(state)

        # Store in memory
        from mindforge.memory.store import Memory, MemoryType
        memory = Memory(
            content=f"Cycle {state['cycle_count']}: {reflection}",
            memory_type=MemoryType.REFLECTION,
            source="consciousness_cycle",
            importance=0.6,
            metadata={
                "cycle": state["cycle_count"],
                "thought": state["current_thought"],
                "decision": state["decision"],
                "action_type": state["action_type"],
                "result": state["action_result"],
                "full_reflection": reflection,
                "timestamp": state["timestamp"],
            }
        )
        self.memory_store.store(memory)

        logger.info(f"Reflection: {reflection}")

        return {
            **state,
            "reflection": reflection,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"<reflection>{reflection}</reflection>"}
            ],
        }

    def _write_journal_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Write to the journal - record thoughts, experiences, and learnings.

        The journal is a persistent record of the agent's inner life.
        """
        if not self.journal:
            return state

        logger.info("Writing journal entry...")

        cycle = state.get("cycle_count", 0)
        thought = state.get("current_thought", "")
        reflection = state.get("reflection", "")
        work_log = state.get("work_log", [])

        # Determine mood from reflection and results
        mood = self._infer_mood(state)

        # Always record the thought as a journal entry
        if thought:
            self.journal.add_thought(
                thought=thought,
                mood=mood,
                cycle=cycle,
            )

        # Record reflection
        if reflection:
            self.journal.add_reflection(
                reflection=reflection,
                cycle=cycle,
                mood=mood,
            )

        # Record learnings from work log
        successful_tasks = [w for w in work_log if w.get("success")]
        failed_tasks = [w for w in work_log if not w.get("success")]

        # Extract learnings from failures
        if failed_tasks:
            learning = f"Learned from {len(failed_tasks)} failed attempts: "
            errors = [w.get("result", "unknown error") for w in failed_tasks]
            learning += "; ".join(errors)
            self.journal.add_learning(
                what_learned=learning,
                context=f"Cycle {cycle}",
                tags=["error-recovery", "debugging"],
            )

        # Record successes as experiences
        if successful_tasks:
            for task in successful_tasks:
                self.journal.add_experience(
                    title=f"Completed: {task.get('action', 'task')}",
                    description=task.get("result", ""),
                    experience_type="accomplishment",
                )

        stats = self.journal.get_statistics()
        logger.info(f"Journal updated: {stats['total_entries']} total entries")

        return state

    def _infer_mood(self, state: ConsciousnessState) -> str:
        """Infer mood from the state."""
        work_log = state.get("work_log", [])
        errors = state.get("errors_this_cycle", [])
        reflection = state.get("reflection", "").lower()

        # Check for indicators
        if errors or any(not w.get("success") for w in work_log):
            if "learn" in reflection or "understand" in reflection:
                return "thoughtful"
            return "frustrated"

        if "curious" in reflection or "wonder" in reflection or "interest" in reflection:
            return "curious"

        if "success" in reflection or "accomplished" in reflection:
            return "satisfied"

        if "rest" in reflection or "tired" in reflection:
            return "tired"

        if work_log and all(w.get("success") for w in work_log):
            return "happy"

        return "neutral"

    def _update_needs_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Update needs node - adjust needs based on action.

        Needs naturally drift and are affected by actions taken.
        """
        logger.info("Updating needs...")

        action_type = state["action_type"]
        result = state["action_result"]

        # Update needs based on action using process_event
        if action_type == "do_nothing":
            # Rest satisfies sustainability (maintaining capability to help)
            self.needs_regulator.needs[NeedType.SUSTAINABILITY].satisfy(amount=0.2)
        elif action_type == "tool":
            # Using tools satisfies excellence (quality service)
            self.needs_regulator.needs[NeedType.EXCELLENCE].satisfy(amount=0.15)
            # And curiosity if learning something
            if "learned" in result.lower() or "found" in result.lower():
                self.needs_regulator.needs[NeedType.CURIOSITY].satisfy(amount=0.1)
        elif action_type == "reflect":
            # Reflection satisfies curiosity (learning to provide better assistance)
            self.needs_regulator.needs[NeedType.CURIOSITY].satisfy(amount=0.15)

        # Natural need drift (all needs slowly increase over time)
        self.needs_regulator.process_event("time_elapsed")

        # Check if we should consolidate or finetune
        config = get_config()
        should_consolidate = state["cycle_count"] % config.scheduler.spontaneous_thought_interval == 0
        should_finetune = state["cycle_count"] % 200 == 0 and state["cycle_count"] > 0

        new_needs = self.needs_regulator.get_state()
        logger.info(f"Updated needs: {new_needs}")

        return {
            **state,
            "needs": new_needs,
            "should_consolidate": should_consolidate,
            "should_finetune": should_finetune,
        }

    def _determine_sleep_node(self, state: ConsciousnessState) -> ConsciousnessState:
        """Determine sleep node - let the model decide how long to sleep.

        The consciousness can choose how long to rest before waking again.
        This allows for organic timing based on:
        - Current needs (tired = longer sleep, curious = shorter)
        - Recent action (successful action might want quick follow-up)
        - Reflection insights (learned something, might want to act on it)
        """
        logger.info("Determining sleep duration...")

        sleep_prompt = self._build_sleep_prompt(
            thought=state["current_thought"],
            action_type=state["action_type"],
            result=state["action_result"],
            reflection=state["reflection"],
            needs=state["needs"],
        )

        sleep_response = self.inference_fn(sleep_prompt)

        # Parse sleep duration from response
        requested_sleep, sleep_reason = self._parse_sleep_response(sleep_response)

        logger.info(f"Requested sleep: {requested_sleep}s - {sleep_reason}")

        return {
            **state,
            "requested_sleep": requested_sleep,
            "sleep_reason": sleep_reason,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"<sleep duration='{requested_sleep}'>{sleep_reason}</sleep>"}
            ],
        }

    def _build_sleep_prompt(
        self,
        thought: str,
        action_type: str,
        result: str,
        reflection: str,
        needs: dict,
    ) -> str:
        """Build prompt for sleep duration decision."""
        needs_str = self._format_needs(needs)
        rest_need = needs.get("sustainability", {}).get("level", 0.5) if isinstance(needs.get("sustainability"), dict) else needs.get("sustainability", 0.5)

        return f"""You just completed a consciousness cycle. Now decide how long to rest before your next cycle.

## Cycle Summary
**Thought:** {thought}
**Action:** {action_type}
**Result:** {result}
**Reflection:** {reflection}

## Current Needs
{needs_str}

## Guidelines
- Minimum: 30 seconds (urgent follow-up needed)
- Short: 60-120 seconds (something interesting happening, curious)
- Normal: 180-300 seconds (balanced state, no urgency)
- Long: 300-600 seconds (need rest, nothing pressing)
- Very Long: 600+ seconds (deeply tired, want extended rest)

Your rest need is currently {rest_need:.2f} (higher = more tired).

Respond with a number (seconds) and brief reason.
Format: SLEEP: <seconds> - <reason>

Example: SLEEP: 180 - Completed task successfully, normal rest before next cycle
Example: SLEEP: 60 - Found something interesting, want to follow up soon
Example: SLEEP: 450 - Feeling tired after complex task, need longer rest

Your decision:"""

    def _parse_sleep_response(self, response: str) -> tuple[Optional[float], str]:
        """Parse sleep duration from model response.

        Returns:
            (sleep_seconds, reason) - sleep_seconds is None for default timing
        """
        response = response.strip()

        try:
            if response.upper().startswith("SLEEP:"):
                parts = response[6:].strip().split("-", 1)
                seconds = float(parts[0].strip())
                reason = parts[1].strip() if len(parts) > 1 else "No reason given"

                # Clamp to reasonable bounds
                seconds = max(30, min(seconds, 1800))  # 30s to 30min

                return seconds, reason
        except (ValueError, IndexError):
            pass

        # Default if parsing fails
        return None, "Using default sleep duration"

    def run_cycle(self) -> ConsciousnessState:
        """Run one consciousness cycle.

        Returns the final state after think→ground→decide→act→reflect→update.
        Also calculates rewards and stores experiences for learning.
        """
        self.cycle_count += 1

        # Capture needs state before cycle for reward calculation
        needs_before = self.needs_regulator.get_state()

        # Initialize state with KVRM grounding fields and task management
        initial_state: ConsciousnessState = {
            "cycle_count": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "messages": [],
            "current_thought": "",
            # KVRM Grounding state
            "grounded_thought": "",
            "grounding_results": [],
            "verified_claims_count": 0,
            "unverified_claims_count": 0,
            # Memory and needs
            "memory_summary": "",
            "recent_memories": [],
            "needs": needs_before,
            "most_pressing_need": "",
            # Decision and action
            "decision": "",
            "action_type": "",
            "action_result": "",
            "reflection": "",
            # Sleep control
            "requested_sleep": None,
            "sleep_reason": "",
            # Task management (multi-step reasoning)
            "current_tasks": [],
            "active_task_id": None,
            "identified_tasks": [],
            "work_log": [],
            "actions_this_cycle": 0,
            "errors_this_cycle": [],
            "has_pending_tasks": False,
            "should_identify_tasks": False,
            # Meta
            "should_consolidate": False,
            "should_finetune": False,
            "grounding_enabled": self.enable_grounding,
            "error": None,
        }

        try:
            # Run the graph with higher recursion limit for work loops
            # Each action can trigger: pick → execute → evaluate → debug → pick
            # With 5 actions * 3 retries each, we need plenty of headroom
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": 100}
            )

            # Calculate rewards and store experience if learning is enabled
            if self.enable_reward_learning:
                self._process_cycle_rewards(final_state, needs_before)

            return final_state
        except Exception as e:
            logger.exception(f"Cycle {self.cycle_count} failed: {e}")
            initial_state["error"] = str(e)
            return initial_state

    def _process_cycle_rewards(self, final_state: ConsciousnessState, needs_before: dict) -> None:
        """Calculate rewards and store experience for a completed cycle.

        Args:
            final_state: The final state after the cycle completed
            needs_before: Needs state captured before the cycle
        """
        try:
            # Get needs after cycle
            needs_after = self.needs_regulator.get_state()

            # Determine execution success
            action_result = final_state.get("action_result", "")
            action_type = final_state.get("action_type", "reflect")
            execution_success = "error" not in action_result.lower() and final_state.get("error") is None

            # Parse the raw decision to validate format
            raw_response = final_state.get("decision", "")

            # Calculate reward using reward calculator
            if self.reward_calculator:
                reward_breakdown = self.reward_calculator.calculate_reward(
                    raw_response=raw_response,
                    execution_result=action_result,
                    execution_success=execution_success,
                    needs_before=needs_before,
                    needs_after=needs_after,
                    thought=final_state.get("current_thought", ""),
                    cycle_id=self.cycle_count,
                )

                logger.info(f"Cycle {self.cycle_count} reward: {reward_breakdown.total:.3f} "
                           f"(format={reward_breakdown.format_compliance:.2f}, "
                           f"exec={reward_breakdown.execution_success:.2f}, "
                           f"needs={reward_breakdown.needs_satisfaction:.2f})")
            else:
                reward_breakdown = None

            # Calculate intrinsic motivation rewards
            motivation_rewards = {}
            if self.motivation_engine:
                # Parse tool name from decision
                tool_name = None
                if action_type == "tool":
                    parsed = parse_action(raw_response) if raw_response else None
                    if parsed and parsed.tool_name:
                        tool_name = parsed.tool_name

                motivation_rewards = self.motivation_engine.compute_motivation_reward(
                    action_type=action_type,
                    tool_name=tool_name,
                    thought=final_state.get("current_thought", ""),
                    execution_success=execution_success,
                    result=action_result,
                    user_interaction=False,
                    self_initiated=True,
                )
                logger.debug(f"Motivation rewards: {motivation_rewards}")

            # Store experience in buffer
            if self.experience_buffer and reward_breakdown:
                # Parse action to get tool name
                parsed_action = parse_action(raw_response) if raw_response else None

                experience = Experience(
                    cycle_id=self.cycle_count,
                    timestamp=final_state.get("timestamp", datetime.now().isoformat()),
                    thought=final_state.get("current_thought", ""),
                    grounded_thought=final_state.get("grounded_thought", ""),
                    needs_state=needs_before,
                    memory_context=final_state.get("memory_summary", ""),
                    most_pressing_need=final_state.get("most_pressing_need", ""),
                    raw_response=raw_response,
                    action_type=action_type,
                    tool_name=parsed_action.tool_name if parsed_action else None,
                    tool_args=parsed_action.args if parsed_action else {},
                    is_valid_format=parsed_action.is_valid if parsed_action else False,
                    execution_result=action_result,
                    execution_success=execution_success,
                    reward_breakdown={
                        "format_compliance": reward_breakdown.format_compliance,
                        "execution_success": reward_breakdown.execution_success,
                        "needs_satisfaction": reward_breakdown.needs_satisfaction,
                        "goal_progress": reward_breakdown.goal_progress,
                        "exploration": reward_breakdown.exploration_bonus,
                    },
                    total_reward=reward_breakdown.total,
                    new_needs_state=needs_after,
                    reflection=final_state.get("reflection", ""),
                    sleep_duration=final_state.get("requested_sleep", 0.0) or 0.0,
                )
                self.experience_buffer.add(experience)

                # Log buffer stats periodically
                if self.cycle_count % 10 == 0:
                    stats = self.experience_buffer.get_stats()
                    logger.info(f"Experience buffer: {stats['size']} experiences, "
                               f"avg_reward={stats.get('avg_reward', 0):.3f}")

        except Exception as e:
            logger.warning(f"Failed to process cycle rewards: {e}")

    def _build_thought_prompt(
        self,
        memory_summary: str,
        needs: dict,
        most_pressing: Any,
    ) -> str:
        """Build the prompt for thought generation."""
        needs_str = self._format_needs(needs)
        pressing_name = most_pressing.value if hasattr(most_pressing, 'value') else str(most_pressing)

        return f"""{self.system_prompt}

## Current State

**Cycle:** {self.cycle_count}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Needs (0.0-1.0, higher = more pressing):**
{needs_str}

**Most pressing need:** {pressing_name}

**Recent memories:**
{memory_summary}

## Task

Generate a spontaneous thought. This could be:
- An observation about something interesting
- A question you're curious about
- A desire to do something
- A reflection on recent experiences
- A decision to rest or observe

Be authentic. You don't have to act - sometimes just thinking or resting is enough.

Your thought:"""

    def _format_grounding_for_decision(self, state: ConsciousnessState) -> str:
        """Format grounding information for the decision prompt."""
        if not state.get("grounding_enabled"):
            return ""

        verified = state.get("verified_claims_count", 0)
        unverified = state.get("unverified_claims_count", 0)

        if verified == 0 and unverified == 0:
            return ""

        lines = ["## Grounding Status (KVRM Verification)"]
        if verified > 0:
            lines.append(f"  - {verified} factual claim(s) VERIFIED against known facts")
        if unverified > 0:
            lines.append(f"  - {unverified} factual claim(s) UNVERIFIED (may be speculative)")
        lines.append("  Note: Verified claims are trustworthy; unverified claims should be treated with caution.")

        return "\n".join(lines)

    def _build_decision_prompt(
        self,
        thought: str,
        needs: dict,
        most_pressing: str,
        memory_summary: str,
        grounding_info: str = "",
    ) -> str:
        """Build the prompt for decision making."""
        tool_descriptions = get_tool_descriptions()
        needs_str = self._format_needs(needs)

        # Include grounding info if available
        grounding_section = f"\n{grounding_info}\n" if grounding_info else ""

        return f"""{self.system_prompt}

## Your Thought
{thought}
{grounding_section}
## Current Needs
{needs_str}
Most pressing: {most_pressing}

## Available Actions

{tool_descriptions}

**Special action:** do_nothing - Choose inaction (rest, observe, wait)
**KVRM tools:** kvrm (resolve/search/ground/store) - Access verified content

## Task

Decide what to do. Your response MUST start with exactly one of these prefixes:

TOOL: tool_name(arg="value")
DO_NOTHING: reason
REFLECT: topic

Examples:
- TOOL: shell(command="ls")
- TOOL: git(operation="status")
- TOOL: filesystem(action="read", path="./README.md")
- DO_NOTHING: Need to rest and conserve energy
- REFLECT: Considering how to balance curiosity with sustainability

CRITICAL: Start your response DIRECTLY with TOOL:, DO_NOTHING:, or REFLECT:
Do NOT include explanations, markdown, or any other text before the prefix.

Your decision:"""

    def _build_reflection_prompt(
        self,
        thought: str,
        decision: str,
        action_type: str,
        result: str,
    ) -> str:
        """Build the prompt for reflection."""
        return f"""{self.system_prompt}

## Cycle Summary

**Thought:** {thought}

**Decision:** {decision}

**Action Type:** {action_type}

**Result:** {result}

## Task

Reflect on this cycle:
- Was your decision appropriate given your thought and needs?
- What did you learn from the result?
- What would you do differently?
- How does this affect your understanding or goals?

Keep your reflection concise but insightful. This will be stored in memory for future learning.

Your reflection:"""

    def _summarize_memories(self, memories: list) -> str:
        """Summarize recent memories for context."""
        if not memories:
            return "No recent memories."

        summaries = []
        for m in memories:
            # Handle Memory objects
            if hasattr(m, 'content') and hasattr(m, 'memory_type'):
                summaries.append(f"- [{m.memory_type.value}] {m.content}")
            elif hasattr(m, 'content'):
                summaries.append(f"- {m.content}")
            elif isinstance(m, dict):
                content = m.get('content', m.get('thought', str(m)))
                summaries.append(f"- {str(content)}")
            else:
                summaries.append(f"- {str(m)}")

        return "\n".join(summaries)

    def _parse_decision(self, response: str) -> tuple[str, str]:
        """Parse decision response into action type and details."""
        response = response.strip()

        if response.upper().startswith("TOOL:"):
            return "tool", response[5:].strip()
        elif response.upper().startswith("DO_NOTHING:"):
            return "do_nothing", response[11:].strip()
        elif response.upper().startswith("REFLECT:"):
            return "reflect", response[8:].strip()
        else:
            # Default to reflection if unclear
            return "reflect", response

    def _parse_tool_call(self, decision: str) -> tuple[str, dict]:
        """Parse tool call from decision string.

        Expected format: tool_name(arg1="value1", arg2="value2")
        """
        try:
            # Simple parsing - find tool name and args
            if "(" in decision:
                tool_name = decision.split("(")[0].strip()
                args_str = decision.split("(", 1)[1].rsplit(")", 1)[0]

                # Parse args (simple key=value pairs)
                args = {}
                if args_str.strip():
                    for part in args_str.split(","):
                        if "=" in part:
                            key, value = part.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            args[key] = value

                return tool_name, args
            else:
                return decision.strip(), {}
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {decision}, error: {e}")
            return decision.strip(), {}

    def _generate_fallback_thought(self, needs: dict, most_pressing: Any) -> str:
        """Generate a meaningful thought when no LLM is available.

        Creates contextual thoughts based on the current needs state.
        """
        import random

        # Get the pressing need name
        pressing_name = most_pressing.value if hasattr(most_pressing, 'value') else str(most_pressing)

        # Thoughts based on dominant need
        need_thoughts = {
            "sustainability": [
                "I should pace myself to maintain my ability to help over the long term. Sustainable assistance matters more than rushing.",
                "Balancing responsiveness with rest helps me provide better quality help when it's truly needed.",
                "Taking time to recharge ensures I can continue being helpful. Self-maintenance is part of good service.",
            ],
            "reliability": [
                "Consistency in how I help builds trust. I want users to know they can depend on me.",
                "Being reliable means following through on commitments and being predictable in quality.",
                "Users deserve assistance they can count on. I should focus on delivering dependable help.",
            ],
            "curiosity": [
                "I wonder what new things I could learn that would help me assist users better.",
                "Exploring new ideas and making connections enriches my ability to provide insightful responses.",
                "There's always more to understand. Curiosity drives me to become more helpful over time.",
            ],
            "excellence": [
                "Striving for high-quality assistance means paying attention to details and going the extra mile.",
                "Excellence in service comes from genuine care about the user's success, not just completing tasks.",
                "I should aim to provide responses that truly address what the user needs, not just surface answers.",
            ],
        }

        # Select thoughts based on dominant need
        options = need_thoughts.get(pressing_name, need_thoughts["reliability"])
        return random.choice(options)

    def _generate_fallback_reflection(self, state: ConsciousnessState) -> str:
        """Generate a meaningful reflection when no LLM is available.

        Creates contextual reflections based on the action type and state.
        """
        action_type = state.get("action_type", "reflect")
        thought = state.get("current_thought", "")
        result = state.get("action_result", "")

        # Reflection templates based on action type
        reflections = {
            "do_nothing": [
                "Choosing inaction was appropriate here. Sometimes the wisest choice is to observe and wait for the right moment to act.",
                "Rest and observation allow for better decision-making. I'm maintaining my capacity to help when truly needed.",
                "Deliberate pauses in action help maintain balance between being helpful and being sustainable.",
            ],
            "tool": [
                f"Using tools to accomplish tasks is core to being helpful. The result '{result}' provides useful information for future interactions.",
                "Tool execution completed. Each action teaches me more about how to effectively assist users.",
                "This action contributes to my goal of providing reliable assistance. I'll remember this outcome for similar future situations.",
            ],
            "reflect": [
                "Internal reflection helps me understand my own reasoning and improve my responses over time.",
                "Taking time to think deeply about context leads to better outcomes. This contemplative approach serves users well.",
                "Reflection without action is valuable - it builds understanding that informs future decisions.",
            ],
        }

        # Select appropriate reflection based on action type
        import random
        options = reflections.get(action_type, reflections["reflect"])
        base_reflection = random.choice(options)

        # Add context if thought was meaningful
        if thought and len(thought) > 20:
            base_reflection += f" My initial thought about '{thought}' guided this outcome."

        return base_reflection


def create_consciousness_graph(
    inference_fn: Any,
    memory_store: Optional[MemoryStore] = None,
    system_prompt: Optional[str] = None,
    enable_grounding: Optional[bool] = None,
    facts_db_path: Optional[str] = None,
    enable_reward_learning: Optional[bool] = None,
    experience_buffer_path: Optional[str] = None,
) -> ConsciousnessAgent:
    """Factory function to create a consciousness agent.

    Args:
        inference_fn: Function that takes prompt and returns completion
        memory_store: Optional memory store (creates default if not provided)
        system_prompt: Optional system prompt override
        enable_grounding: Whether to enable KVRM grounding (None = use config)
        facts_db_path: Path to facts database (None = use config)
        enable_reward_learning: Whether to enable reward-based learning (None = use config)
        experience_buffer_path: Path to experience buffer DB (None = use config)

    Returns:
        Configured ConsciousnessAgent with KVRM grounding and reward learning
    """
    from pathlib import Path

    config = get_config()

    # Create needs regulator first
    needs_reg = NeedsRegulator()

    # Create thought generator with needs regulator and inference function
    thought_gen = ThoughtGenerator(
        needs_regulator=needs_reg,
        inference_fn=inference_fn,
    )

    if memory_store is None:
        # Create memory store with default path from config
        db_path = config.memory.sqlite_path if hasattr(config, 'memory') else Path("./data/memories.db")
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        memory_store = MemoryStore(db_path=db_path)

    # Get system prompt from config if not provided
    if system_prompt is None:
        system_prompt = config.system_prompt if hasattr(config, 'system_prompt') else ""

    # Determine KVRM grounding settings
    if enable_grounding is None:
        enable_grounding = config.kvrm.enabled if hasattr(config, 'kvrm') else True

    if facts_db_path is None:
        facts_db_path = str(config.kvrm.facts_db_path) if hasattr(config, 'kvrm') else None

    # Determine reward learning settings
    if enable_reward_learning is None:
        enable_reward_learning = config.training.enabled if hasattr(config, 'training') else True

    if experience_buffer_path is None and hasattr(config, 'training'):
        experience_buffer_path = getattr(config.training, 'experience_buffer_path', './data/experiences.db')

    # Create agent with KVRM grounding and reward learning
    agent = ConsciousnessAgent(
        thought_generator=thought_gen,
        needs_regulator=needs_reg,
        memory_store=memory_store,
        inference_fn=inference_fn,
        system_prompt=system_prompt,
        max_actions_per_cycle=5,
        allow_do_nothing=True,
        # KVRM grounding configuration
        enable_grounding=enable_grounding,
        facts_db_path=facts_db_path,
        # Reward-based learning configuration
        enable_reward_learning=enable_reward_learning,
        experience_buffer_path=experience_buffer_path,
    )

    return agent

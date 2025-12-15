"""
Conch Agent System

LangGraph-based stateful agent for the consciousness loop.
Implements the think → decide → act → reflect → update_needs cycle.
Now includes task management for multi-step reasoning.
"""

from conch.agent.tool_adapter import (
    ToolAdapter,
    create_langgraph_tools,
    get_all_tools,
)
from conch.agent.langgraph_agent import (
    ConsciousnessState,
    ConsciousnessAgent,
    create_consciousness_graph,
)
from conch.agent.task_list import (
    InternalTask,
    TaskStatus,
    TaskPriority,
    PersistentTaskList,
    WorkLogEntry,
)
from conch.agent.journal import (
    Journal,
    JournalEntry,
    JournalEntryType,
)

__all__ = [
    # Tool adapters
    "ToolAdapter",
    "create_langgraph_tools",
    "get_all_tools",
    # Consciousness agent
    "ConsciousnessState",
    "ConsciousnessAgent",
    "create_consciousness_graph",
    # Task management
    "InternalTask",
    "TaskStatus",
    "TaskPriority",
    "PersistentTaskList",
    "WorkLogEntry",
    # Journal
    "Journal",
    "JournalEntry",
    "JournalEntryType",
]

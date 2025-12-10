"""
MindForge Memory System

Provides persistent memory capabilities for MindForge:
- Short-term (working) memory: Recent interactions and context
- Long-term memory: Persistent storage with semantic search
- Memory consolidation: Converting short-term to long-term

Storage backends:
- SQLite: Structured data and metadata
- ChromaDB: Vector embeddings for semantic search
"""

from mindforge.memory.store import MemoryStore, Memory, MemoryType
from mindforge.memory.short_term import ShortTermMemory
from mindforge.memory.long_term import LongTermMemory, VectorMemory

__all__ = [
    "MemoryStore",
    "Memory",
    "MemoryType",
    "ShortTermMemory",
    "LongTermMemory",
    "VectorMemory",
]

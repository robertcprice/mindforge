"""
MindForge Vector Memory - Alias module

This module re-exports VectorMemory from long_term for convenience.
"""

from mindforge.memory.long_term import VectorMemory, VectorSearchResult

__all__ = ["VectorMemory", "VectorSearchResult"]

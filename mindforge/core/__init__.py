"""
MindForge Core - The consciousness simulation engine.

This module contains the core components that simulate consciousness:
- Mind: The central orchestrator
- Thought: Thought generation and processing
- Needs: The needs-regulator system
"""

from mindforge.core.mind import Mind
from mindforge.core.needs import NeedsRegulator
from mindforge.core.thought import Thought, ThoughtGenerator

__all__ = ["Mind", "Thought", "ThoughtGenerator", "NeedsRegulator"]

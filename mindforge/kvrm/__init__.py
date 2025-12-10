"""
MindForge KVRM (Key-Value Response Mapping) Module

Zero-hallucination grounding through key-based routing to verified data stores.
Inspired by the Logos project's approach to provably accurate responses.

Core concepts:
- KeyStore: Abstract base for verified data stores
- KeyResolver: Resolves keys to verified content
- GroundingRouter: Routes claims through appropriate key stores
- KVRMTool: Tool interface for consciousness agent
"""

from mindforge.kvrm.key_store import (
    KeyStore,
    ResolvedContent,
    KeyType,
)
from mindforge.kvrm.resolver import (
    KeyResolver,
    MemoryKeyStore,
    FactKeyStore,
    ExternalKeyStore,
)
from mindforge.kvrm.grounding import (
    GroundingRouter,
    GroundingResult,
    ClaimType,
)
from mindforge.kvrm.tool import KVRMTool, GroundedThinkingTool, create_kvrm_tools

__all__ = [
    "KeyStore",
    "ResolvedContent",
    "KeyType",
    "KeyResolver",
    "MemoryKeyStore",
    "FactKeyStore",
    "ExternalKeyStore",
    "GroundingRouter",
    "GroundingResult",
    "ClaimType",
    "KVRMTool",
    "GroundedThinkingTool",
    "create_kvrm_tools",
]

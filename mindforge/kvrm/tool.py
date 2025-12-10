"""
KVRM Tool

Tool interface for the consciousness agent to access verified content
through key-value response mapping.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mindforge.tools import Tool, ToolResult, ToolStatus
from mindforge.kvrm.key_store import KeyType, ResolvedContent
from mindforge.kvrm.resolver import KeyResolver
from mindforge.kvrm.grounding import GroundingRouter, GroundingResult, ClaimType

logger = logging.getLogger(__name__)


class KVRMTool(Tool):
    """
    Tool for accessing verified content through KVRM.

    Operations:
    - resolve: Look up a specific key
    - search: Search for content by query
    - ground: Verify a claim against known facts
    - store: Store new verified content

    This tool enables the consciousness to:
    1. Ground factual claims before stating them
    2. Access verified memories and facts
    3. Store new learnings in a verifiable way
    """

    def __init__(
        self,
        key_resolver: KeyResolver,
        grounding_router: Optional[GroundingRouter] = None,
    ):
        """
        Initialize KVRM tool.

        Args:
            key_resolver: KeyResolver for content lookup
            grounding_router: Optional GroundingRouter for claim verification
        """
        super().__init__(
            name="kvrm",
            description=(
                "Access verified content through Key-Value Response Mapping. "
                "Use for: looking up facts, grounding claims, searching memories. "
                "Operations: resolve, search, ground, store"
            ),
            requires_confirmation=False,
        )
        self.key_resolver = key_resolver
        self.grounding_router = grounding_router

    def execute(
        self,
        operation: str = "resolve",
        key: Optional[str] = None,
        query: Optional[str] = None,
        claim: Optional[str] = None,
        content: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> ToolResult:
        """
        Execute a KVRM operation.

        Args:
            operation: One of "resolve", "search", "ground", "store"
            key: Key to resolve (for resolve/store operations)
            query: Search query (for search operation)
            claim: Claim to verify (for ground operation)
            content: Content to store (for store operation)
            source: Source of content (for store operation)
            metadata: Additional metadata
            limit: Maximum results for search

        Returns:
            ToolResult with operation outcome
        """
        try:
            if operation == "resolve":
                return self._resolve(key)
            elif operation == "search":
                return self._search(query, limit)
            elif operation == "ground":
                return self._ground(claim)
            elif operation == "store":
                return self._store(key, content, source, metadata or {})
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"Unknown operation: {operation}. Use: resolve, search, ground, store",
                )

        except Exception as e:
            logger.exception(f"KVRM operation '{operation}' failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"KVRM error: {str(e)}",
            )

    def _resolve(self, key: Optional[str]) -> ToolResult:
        """Resolve a key to verified content."""
        if not key:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Key required for resolve operation",
            )

        resolved = self.key_resolver.resolve(key)

        if resolved is None:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Key not found: {key}",
                metadata={"found": False, "key": key},
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=resolved.content,
            metadata={
                "found": True,
                "key": resolved.key,
                "key_type": resolved.key_type.value,
                "confidence": resolved.confidence,
                "citation": resolved.citation,
                "content_hash": resolved.content_hash,
                "source": resolved.source,
            },
        )

    def _search(self, query: Optional[str], limit: int) -> ToolResult:
        """Search for content by query."""
        if not query:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Query required for search operation",
            )

        results = self.key_resolver.search(query, limit=limit)

        if not results:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"No results found for: {query}",
                metadata={"results_count": 0, "query": query},
            )

        # Format results
        output_lines = [f"Found {len(results)} result(s) for: {query}\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"{i}. [{r.key_type.value}] {r.key}\n"
                f"   {r.content[:100]}{'...' if len(r.content) > 100 else ''}\n"
                f"   Confidence: {r.confidence:.2f}\n"
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_lines),
            metadata={
                "results_count": len(results),
                "query": query,
                "keys": [r.key for r in results],
            },
        )

    def _ground(self, claim: Optional[str]) -> ToolResult:
        """Verify a claim against known facts."""
        if not claim:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Claim required for ground operation",
            )

        if not self.grounding_router:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Grounding router not configured",
            )

        result = self.grounding_router.ground(claim)

        if result.is_verified:
            output = (
                f"VERIFIED: {claim}\n"
                f"Source: {result.resolved_content.citation if result.resolved_content else 'N/A'}\n"
                f"Content: {result.resolved_content.content if result.resolved_content else 'N/A'}"
            )
        elif result.grounded:
            output = (
                f"GROUNDED (confidence: {result.confidence:.2f}): {claim}\n"
                f"Key: {result.key_used}\n"
                f"Note: Partially verified"
            )
        elif result.claim_type in (ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.QUESTION):
            output = f"NOT APPLICABLE: {result.claim_type.value} claims don't require verification"
        else:
            output = (
                f"UNVERIFIED: {claim}\n"
                f"Reason: {result.reason}"
            )
            if result.suggestions:
                output += f"\nSuggestions: {', '.join(result.suggestions)}"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata={
                "grounded": result.grounded,
                "verified": result.is_verified,
                "claim_type": result.claim_type.value,
                "confidence": result.confidence,
                "key_used": result.key_used,
                "suggestions": result.suggestions,
            },
        )

    def _store(
        self,
        key: Optional[str],
        content: Optional[str],
        source: Optional[str],
        metadata: Dict[str, Any],
    ) -> ToolResult:
        """Store new verified content."""
        if not content:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Content required for store operation",
            )

        # Determine which store to use based on key prefix or metadata
        store_name = None
        if key:
            if key.startswith("mem:"):
                store_name = "memory"
            elif key.startswith("fact:"):
                store_name = "fact"

        if not store_name:
            store_name = metadata.get("store", "fact")

        store = self.key_resolver.stores.get(store_name)
        if not store:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Store not found: {store_name}",
            )

        if store.read_only:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Store '{store_name}' is read-only",
            )

        stored_key = store.store(
            key=key or "",
            content=content,
            source=source or "consciousness",
            metadata=metadata,
        )

        if stored_key:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Content stored successfully with key: {stored_key}",
                metadata={"key": stored_key, "store": store_name},
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Failed to store content",
            )


class GroundedThinkingTool(Tool):
    """
    Tool for grounded thinking - generates thoughts then verifies claims.

    This combines free-form thinking with automatic grounding of
    factual claims, enabling Claude Code-level reasoning with
    zero-hallucination on verifiable facts.
    """

    def __init__(
        self,
        grounding_router: GroundingRouter,
        inference_fn: Any,
    ):
        """
        Initialize grounded thinking tool.

        Args:
            grounding_router: Router for grounding claims
            inference_fn: LLM inference function for generating thoughts
        """
        super().__init__(
            name="grounded_think",
            description=(
                "Think about a topic with automatic fact-checking. "
                "Generates thoughts then verifies factual claims against "
                "known facts. Use for reasoning that needs accuracy."
            ),
            requires_confirmation=False,
        )
        self.grounding_router = grounding_router
        self.inference_fn = inference_fn

    def execute(
        self,
        prompt: str,
        ground_claims: bool = True,
    ) -> ToolResult:
        """
        Generate a grounded thought.

        Args:
            prompt: What to think about
            ground_claims: Whether to verify factual claims

        Returns:
            ToolResult with thought and grounding results
        """
        try:
            # Generate initial thought
            thought = self.inference_fn(prompt)

            if not ground_claims:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=thought,
                    metadata={"grounded": False},
                )

            # Ground the thought
            grounded_thought, results = self.grounding_router.ground_thought(thought)

            # Summarize grounding results
            verified_count = sum(1 for r in results if r.is_verified)
            grounded_count = sum(1 for r in results if r.grounded)
            total_claims = len([r for r in results if r.claim_type == ClaimType.FACTUAL])

            summary = (
                f"\n\n---\nGrounding: {verified_count}/{total_claims} claims verified, "
                f"{grounded_count}/{total_claims} grounded"
            )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=grounded_thought + summary,
                metadata={
                    "grounded": True,
                    "verified_claims": verified_count,
                    "grounded_claims": grounded_count,
                    "total_factual_claims": total_claims,
                    "original_thought": thought,
                    "grounding_results": [r.to_dict() for r in results],
                },
            )

        except Exception as e:
            logger.exception(f"Grounded thinking failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Grounded thinking error: {str(e)}",
            )


def create_kvrm_tools(
    memory_store: Any = None,
    facts_db_path: Optional[str] = None,
    inference_fn: Any = None,
) -> List[Tool]:
    """
    Factory function to create all KVRM tools.

    Args:
        memory_store: MindForge MemoryStore instance
        facts_db_path: Path to facts database
        inference_fn: LLM inference function

    Returns:
        List of configured KVRM tools
    """
    from mindforge.kvrm.grounding import create_grounding_router
    from mindforge.kvrm.resolver import (
        KeyResolver,
        MemoryKeyStore,
        FactKeyStore,
        ExternalKeyStore,
    )
    from pathlib import Path

    # Create key resolver
    resolver = KeyResolver()

    if memory_store:
        resolver.register_store("memory", MemoryKeyStore(memory_store))

    if facts_db_path:
        resolver.register_store("fact", FactKeyStore(Path(facts_db_path)))

    resolver.register_store("external", ExternalKeyStore())

    # Create grounding router
    grounding_router = create_grounding_router(
        memory_store=memory_store,
        facts_db_path=facts_db_path,
        inference_fn=inference_fn,
    )

    tools = [
        KVRMTool(resolver, grounding_router),
    ]

    # Add grounded thinking tool if inference function available
    if inference_fn:
        tools.append(GroundedThinkingTool(grounding_router, inference_fn))

    return tools

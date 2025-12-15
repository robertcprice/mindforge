"""
Grounding Router

Routes claims through verified key stores for zero-hallucination responses.
Integrates with the consciousness agent to ground factual statements.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from conch.kvrm.key_store import KeyType, ResolvedContent
from conch.kvrm.resolver import KeyResolver

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that can be grounded."""

    FACTUAL = "factual"           # Verifiable factual claim
    MEMORY = "memory"             # Reference to past experience
    OPINION = "opinion"           # Subjective opinion (not groundable)
    QUESTION = "question"         # Question (not a claim)
    CREATIVE = "creative"         # Creative/imaginative content
    ACTION = "action"             # Action statement
    UNKNOWN = "unknown"           # Cannot classify


@dataclass
class GroundingResult:
    """
    Result of grounding a claim.

    Contains the original claim, its type, and resolved verification.
    """

    original: str                               # Original claim text
    claim_type: ClaimType                       # Classified type
    grounded: bool                              # Whether claim was grounded
    confidence: float                           # Confidence in grounding

    # Verification
    resolved_content: Optional[ResolvedContent] = None  # Verified content
    key_used: Optional[str] = None                      # Key that resolved

    # For ungrounded claims
    reason: str = ""                            # Why grounding failed
    suggestions: List[str] = field(default_factory=list)  # Alternative keys

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """Check if claim is fully verified."""
        return self.grounded and self.confidence >= 0.9

    @property
    def status(self) -> str:
        """Human-readable status."""
        if self.is_verified:
            return "VERIFIED"
        elif self.grounded:
            return "GROUNDED"
        elif self.claim_type in (ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.QUESTION):
            return "NOT_APPLICABLE"
        else:
            return "UNVERIFIED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "claim_type": self.claim_type.value,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "key_used": self.key_used,
            "status": self.status,
            "reason": self.reason,
            "suggestions": self.suggestions,
            "resolved_content": self.resolved_content.to_dict() if self.resolved_content else None,
        }


class GroundingRouter:
    """
    Routes claims through verified key stores for grounding.

    The router:
    1. Classifies incoming claims by type
    2. Extracts potential keys from factual claims
    3. Routes to appropriate key stores
    4. Returns grounded or ungrounded results

    This enables the consciousness to think freely while grounding
    factual statements through verified data.
    """

    def __init__(
        self,
        key_resolver: KeyResolver,
        inference_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the grounding router.

        Args:
            key_resolver: KeyResolver for looking up verified content
            inference_fn: Optional LLM function for key extraction
        """
        self.key_resolver = key_resolver
        self.inference_fn = inference_fn

        # Patterns for extracting keys from text
        self._key_patterns = [
            # Memory keys: mem:thought:20241209:abc123
            re.compile(r"mem:[a-z]+:\d{8}:[a-f0-9]+"),
            # Fact keys: fact:domain:id
            re.compile(r"fact:[a-z_]+:[a-z0-9_]+"),
            # External keys: ext:source:...
            re.compile(r"ext:[a-z]+:[^\s]+"),
        ]

        # Claim type indicators
        self._factual_indicators = [
            "is", "are", "was", "were", "has", "have", "states", "says",
            "according to", "research shows", "studies indicate",
            "the fact that", "it is true that", "scientifically",
        ]
        self._opinion_indicators = [
            "i think", "i believe", "in my opinion", "i feel",
            "probably", "maybe", "perhaps", "could be",
        ]
        self._question_indicators = ["?", "what", "how", "why", "when", "where", "who"]

    def ground(
        self,
        text: str,
        force_type: Optional[ClaimType] = None,
    ) -> GroundingResult:
        """
        Ground a single claim or statement.

        Args:
            text: The text to ground
            force_type: Override claim type classification

        Returns:
            GroundingResult with verification details
        """
        # Classify claim type
        claim_type = force_type or self._classify_claim(text)

        # Not all claims need grounding
        if claim_type in (ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.QUESTION):
            return GroundingResult(
                original=text,
                claim_type=claim_type,
                grounded=False,
                confidence=1.0,
                reason=f"Claim type '{claim_type.value}' does not require grounding",
            )

        # Try to extract and resolve keys
        keys = self._extract_keys(text)

        if keys:
            # Try each extracted key
            for key in keys:
                resolved = self.key_resolver.resolve(key)
                if resolved:
                    return GroundingResult(
                        original=text,
                        claim_type=claim_type,
                        grounded=True,
                        confidence=resolved.confidence,
                        resolved_content=resolved,
                        key_used=key,
                    )

        # Try LLM-based key extraction if available
        if self.inference_fn and claim_type == ClaimType.FACTUAL:
            llm_result = self._llm_ground(text)
            if llm_result.grounded:
                return llm_result

        # Try semantic search as fallback
        try:
            search_results = self.key_resolver.search(text, limit=3)
        except Exception as e:
            logger.warning(f"Search failed during grounding: {e}")
            search_results = []

        if search_results:
            best = search_results[0]
            # Only use if reasonably confident match
            if best.confidence >= 0.7:
                return GroundingResult(
                    original=text,
                    claim_type=claim_type,
                    grounded=True,
                    confidence=best.confidence * 0.8,  # Reduce for semantic match
                    resolved_content=best,
                    key_used=best.key,
                    metadata={"match_type": "semantic"},
                )

            # Return as suggestions if not confident enough
            return GroundingResult(
                original=text,
                claim_type=claim_type,
                grounded=False,
                confidence=0.0,
                reason="No exact match found",
                suggestions=[r.key for r in search_results],
            )

        return GroundingResult(
            original=text,
            claim_type=claim_type,
            grounded=False,
            confidence=0.0,
            reason="No verified content found for this claim",
        )

    def ground_multiple(
        self,
        claims: List[str],
    ) -> List[GroundingResult]:
        """Ground multiple claims."""
        return [self.ground(claim) for claim in claims]

    def ground_thought(
        self,
        thought: str,
    ) -> Tuple[str, List[GroundingResult]]:
        """
        Ground a full thought, extracting and verifying claims.

        Args:
            thought: Full thought text

        Returns:
            (grounded_thought, list of GroundingResults)
        """
        # Split thought into sentences/claims
        claims = self._split_into_claims(thought)

        results = []
        grounded_parts = []

        for claim in claims:
            result = self.ground(claim)
            results.append(result)

            if result.is_verified and result.resolved_content:
                # Replace claim with verified content and citation
                grounded_parts.append(
                    f"{result.resolved_content.content} {result.resolved_content.citation}"
                )
            else:
                # Keep original claim
                grounded_parts.append(claim)

        grounded_thought = " ".join(grounded_parts)
        return grounded_thought, results

    def _classify_claim(self, text: str) -> ClaimType:
        """Classify the type of claim."""
        text_lower = text.lower().strip()

        # Check for questions
        if any(ind in text_lower for ind in self._question_indicators):
            return ClaimType.QUESTION

        # Check for opinions
        if any(ind in text_lower for ind in self._opinion_indicators):
            return ClaimType.OPINION

        # Check for factual indicators
        if any(ind in text_lower for ind in self._factual_indicators):
            return ClaimType.FACTUAL

        # Check for memory references
        if any(phrase in text_lower for phrase in ["i remember", "previously", "before", "last time"]):
            return ClaimType.MEMORY

        # Check for action statements
        if text_lower.startswith(("let me", "i will", "i'll", "going to")):
            return ClaimType.ACTION

        # Default to factual if it contains verifiable-looking content
        if any(self._key_patterns[i].search(text) for i in range(len(self._key_patterns))):
            return ClaimType.FACTUAL

        return ClaimType.UNKNOWN

    def _extract_keys(self, text: str) -> List[str]:
        """Extract potential keys from text."""
        keys = []
        for pattern in self._key_patterns:
            matches = pattern.findall(text)
            keys.extend(matches)
        return keys

    def _split_into_claims(self, text: str) -> List[str]:
        """Split text into individual claims/sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _llm_ground(self, text: str) -> GroundingResult:
        """Use LLM to extract key and ground claim."""
        if not self.inference_fn:
            return GroundingResult(
                original=text,
                claim_type=ClaimType.FACTUAL,
                grounded=False,
                confidence=0.0,
                reason="No inference function available",
            )

        prompt = f"""Extract a verification key for this claim. Output JSON only.

Claim: "{text}"

Available key formats:
- mem:{{type}}:{{date}}:{{hash}} - for memories
- fact:{{domain}}:{{id}} - for facts
- ext:{{source}}:{{path}} - for external sources
- UNKNOWN - if claim cannot be verified

Output format:
{{"key": "...", "confidence": 0.0-1.0, "reasoning": "..."}}

Your response:"""

        try:
            response = self.inference_fn(prompt)

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                key = data.get("key", "UNKNOWN")
                confidence = data.get("confidence", 0.5)

                if key and key != "UNKNOWN":
                    resolved = self.key_resolver.resolve(key)
                    if resolved:
                        return GroundingResult(
                            original=text,
                            claim_type=ClaimType.FACTUAL,
                            grounded=True,
                            confidence=min(confidence, resolved.confidence),
                            resolved_content=resolved,
                            key_used=key,
                            metadata={"extraction_method": "llm"},
                        )

        except Exception as e:
            logger.warning(f"LLM grounding failed: {e}")

        return GroundingResult(
            original=text,
            claim_type=ClaimType.FACTUAL,
            grounded=False,
            confidence=0.0,
            reason="LLM key extraction did not find verifiable content",
        )


def create_grounding_router(
    memory_store: Any = None,
    facts_db_path: Optional[str] = None,
    inference_fn: Optional[Callable[[str], str]] = None,
) -> GroundingRouter:
    """
    Factory function to create a configured grounding router.

    Args:
        memory_store: Conch MemoryStore instance
        facts_db_path: Path to facts database
        inference_fn: LLM inference function

    Returns:
        Configured GroundingRouter
    """
    from conch.kvrm.resolver import (
        KeyResolver,
        MemoryKeyStore,
        FactKeyStore,
        ExternalKeyStore,
    )
    from pathlib import Path

    resolver = KeyResolver()

    # Add memory store if provided
    if memory_store:
        resolver.register_store("memory", MemoryKeyStore(memory_store))

    # Add facts store if path provided
    if facts_db_path:
        resolver.register_store("fact", FactKeyStore(Path(facts_db_path)))

    # Add external store (backends can be added later)
    resolver.register_store("external", ExternalKeyStore())

    return GroundingRouter(resolver, inference_fn)

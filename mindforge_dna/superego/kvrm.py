"""
Knowledge Verification and Routing Module (KVRM).

Implements fact verification and claim grounding using the Key-Value Retrieval
Memory pattern. Routes claims to appropriate knowledge sources and verifies
facts against stored knowledge.
"""

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Classification of claim types for routing."""
    FACTUAL = "factual"          # Objective, verifiable facts
    MEMORY = "memory"            # Personal/episodic memories
    OPINION = "opinion"          # Subjective judgments
    QUESTION = "question"        # Information requests
    CREATIVE = "creative"        # Generated content
    ACTION = "action"            # Commands or operations
    UNKNOWN = "unknown"          # Cannot classify


@dataclass(frozen=True)
class GroundingResult:
    """
    Result of claim grounding/verification.

    Attributes:
        claim: Original claim being verified
        claim_type: Classification of the claim
        is_grounded: Whether claim is verified/grounded
        source: Where the grounding came from
        confidence: Confidence score 0.0-1.0
        evidence: Supporting evidence or explanation
        key: KVRM key if stored
    """
    claim: str
    claim_type: ClaimType
    is_grounded: bool
    source: str
    confidence: float
    evidence: Optional[str] = None
    key: Optional[str] = None

    @property
    def is_verified(self) -> bool:
        """Alias for is_grounded for clarity."""
        return self.is_grounded


class KVRMRouter:
    """
    Knowledge Verification and Routing Module.

    Classifies claims, routes to appropriate knowledge sources, and verifies
    facts against stored knowledge using a key-value retrieval pattern.

    Key Format:
        mem:type:date:hash - Personal memories
        fact:domain:id - Verified facts
        ext:source:path - External knowledge
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize KVRM with fact storage.

        Args:
            db_path: Path to SQLite database, or None for default location
        """
        if db_path is None:
            db_path = Path.home() / ".mindforge" / "kvrm_facts.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        self._compile_patterns()

        logger.info(f"KVRMRouter initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database for fact storage."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    claim TEXT NOT NULL,
                    domain TEXT,
                    source TEXT,
                    confidence REAL,
                    evidence TEXT,
                    created_at TEXT,
                    verified_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    claim TEXT NOT NULL,
                    memory_type TEXT,
                    context TEXT,
                    created_at TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_domain
                ON facts(domain)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type)
            """)

            conn.commit()

            # Lightweight schema migrations for existing databases
            self._ensure_column(conn, "facts", "evidence", "TEXT")
            self._ensure_column(conn, "facts", "verified_at", "TEXT")
            self._ensure_column(conn, "facts", "source", "TEXT")
            self._ensure_column(conn, "memories", "context", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
        """Ensure a column exists; add if missing."""
        try:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing = {row[1] for row in info}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except Exception as e:
            logger.warning(f"Schema migration for {table}.{column} failed: {e}")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for claim classification."""
        # Factual claim indicators
        self._factual_patterns = [
            re.compile(r"\b(is|are|was|were|has|have|will be)\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*(percent|%|degrees?|meters?|feet)\b", re.IGNORECASE),
            re.compile(r"\b(always|never|every|all|none)\b", re.IGNORECASE),
        ]

        # Memory/episodic indicators
        self._memory_patterns = [
            re.compile(r"\b(I|we|my|our)\b"),
            re.compile(r"\b(remember|recall|experienced|saw|heard)\b", re.IGNORECASE),
            re.compile(r"\b(yesterday|last week|ago|previously)\b", re.IGNORECASE),
        ]

        # Opinion indicators
        self._opinion_patterns = [
            re.compile(r"\b(think|believe|feel|seems?|appears?)\b", re.IGNORECASE),
            re.compile(r"\b(good|bad|better|worse|best|worst)\b", re.IGNORECASE),
            re.compile(r"\b(should|ought|must|need to)\b", re.IGNORECASE),
        ]

        # Question indicators
        self._question_patterns = [
            re.compile(r"\?$"),
            re.compile(r"^(what|who|where|when|why|how|which|whose)\b", re.IGNORECASE),
        ]

        # Action indicators
        self._action_patterns = [
            re.compile(r"^(create|delete|update|modify|run|execute)\b", re.IGNORECASE),
            re.compile(r"^(please|could you|can you|would you)\b", re.IGNORECASE),
        ]

    def classify_claim(self, claim: str) -> ClaimType:
        """
        Classify claim type for routing.

        Args:
            claim: Text claim to classify

        Returns:
            ClaimType enum value
        """
        claim = claim.strip()

        # Check question first (most distinctive)
        if any(p.search(claim) for p in self._question_patterns):
            return ClaimType.QUESTION

        # Check action commands
        if any(p.search(claim) for p in self._action_patterns):
            return ClaimType.ACTION

        # Check memory/episodic
        memory_matches = sum(1 for p in self._memory_patterns if p.search(claim))
        if memory_matches >= 2:
            return ClaimType.MEMORY

        # Check opinion
        opinion_matches = sum(1 for p in self._opinion_patterns if p.search(claim))
        if opinion_matches >= 2:
            return ClaimType.OPINION

        # Check factual
        factual_matches = sum(1 for p in self._factual_patterns if p.search(claim))
        if factual_matches >= 1:
            return ClaimType.FACTUAL

        # Check for creative content (longer, narrative)
        if len(claim.split()) > 30 and not any([
            any(p.search(claim) for p in self._factual_patterns),
            any(p.search(claim) for p in self._memory_patterns),
        ]):
            return ClaimType.CREATIVE

        return ClaimType.UNKNOWN

    def ground_claim(
        self,
        claim: str,
        domain: Optional[str] = None,
        force_verify: bool = False
    ) -> GroundingResult:
        """
        Ground/verify a claim against knowledge sources.

        Args:
            claim: Claim to verify
            domain: Optional domain hint for routing
            force_verify: Force external verification even if cached

        Returns:
            GroundingResult with verification status
        """
        claim = claim.strip()
        claim_type = self.classify_claim(claim)

        # Route based on claim type
        if claim_type == ClaimType.MEMORY:
            return self._ground_memory(claim)
        elif claim_type == ClaimType.FACTUAL:
            return self._ground_factual(claim, domain, force_verify)
        elif claim_type == ClaimType.OPINION:
            return GroundingResult(
                claim=claim,
                claim_type=claim_type,
                is_grounded=True,  # Opinions are inherently grounded
                source="subjective",
                confidence=1.0,
                evidence="Opinion claims are subjective and don't require verification"
            )
        elif claim_type == ClaimType.QUESTION:
            return GroundingResult(
                claim=claim,
                claim_type=claim_type,
                is_grounded=False,
                source="none",
                confidence=0.0,
                evidence="Questions require answers, not grounding"
            )
        else:
            return GroundingResult(
                claim=claim,
                claim_type=claim_type,
                is_grounded=False,
                source="unknown",
                confidence=0.0,
                evidence="Unable to classify claim for grounding"
            )

    def _ground_memory(self, claim: str) -> GroundingResult:
        """Ground a memory claim by checking episodic storage."""
        # Generate memory key with timestamp to avoid collisions
        claim_hash = hashlib.md5(claim.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"mem:episodic:{timestamp}:{claim_hash}"

        # Check if memory exists
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT claim, context FROM memories WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

        if row:
            logger.debug(f"Memory grounded from storage: {key}")
            return GroundingResult(
                claim=claim,
                claim_type=ClaimType.MEMORY,
                is_grounded=True,
                source="episodic_memory",
                confidence=0.9,
                evidence=row[1] if row[1] else "Previously stored memory",
                key=key
            )
        else:
            # Store new memory
            self._store_memory(key, claim)
            logger.debug(f"New memory stored: {key}")
            return GroundingResult(
                claim=claim,
                claim_type=ClaimType.MEMORY,
                is_grounded=True,
                source="episodic_memory",
                confidence=0.8,
                evidence="Newly stored episodic memory",
                key=key
            )

    def _ground_factual(
        self,
        claim: str,
        domain: Optional[str],
        force_verify: bool
    ) -> GroundingResult:
        """Ground a factual claim against verified facts."""
        # Generate fact key
        if domain is None:
            domain = "general"

        claim_hash = hashlib.md5(claim.encode()).hexdigest()[:12]
        key = f"fact:{domain}:{claim_hash}"

        # Check cached facts
        if not force_verify:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT confidence, evidence, source FROM facts WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

            if row:
                logger.debug(f"Fact grounded from cache: {key}")
                return GroundingResult(
                    claim=claim,
                    claim_type=ClaimType.FACTUAL,
                    is_grounded=True,
                    source=row[2] or "cached",
                    confidence=row[0] or 0.7,
                    evidence=row[1],
                    key=key
                )

        # Would integrate with external verification here
        # For now, return unverified status
        logger.debug(f"Factual claim requires verification: {claim}")
        return GroundingResult(
            claim=claim,
            claim_type=ClaimType.FACTUAL,
            is_grounded=False,
            source="unverified",
            confidence=0.0,
            evidence="External verification required",
            key=key
        )

    def _store_memory(self, key: str, claim: str, context: Optional[str] = None) -> None:
        """Store a memory in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (key, claim, memory_type, context, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, claim, "episodic", context, datetime.now().isoformat())
            )
            conn.commit()

    def store_fact(
        self,
        claim: str,
        domain: str,
        source: str,
        confidence: float,
        evidence: Optional[str] = None
    ) -> str:
        """
        Store a verified fact.

        Args:
            claim: The factual claim
            domain: Knowledge domain
            source: Source of verification
            confidence: Confidence score 0.0-1.0
            evidence: Supporting evidence

        Returns:
            The KVRM key for the stored fact
        """
        claim_hash = hashlib.md5(claim.encode()).hexdigest()[:12]
        key = f"fact:{domain}:{claim_hash}"

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO facts
                   (key, claim, domain, source, confidence, evidence,
                    created_at, verified_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, claim, domain, source, confidence, evidence,
                 datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()

        logger.info(f"Fact stored: {key} (confidence: {confidence})")
        return key

    def ground_thought(self, thought: str) -> List[GroundingResult]:
        """
        Ground all claims in a complex thought.

        Breaks down a thought into component claims and grounds each.

        Args:
            thought: Complex thought containing multiple claims

        Returns:
            List of GroundingResult for each claim found
        """
        # Simple sentence splitting (could be enhanced)
        sentences = [s.strip() for s in thought.split('.') if s.strip()]

        results = []
        for sentence in sentences:
            if len(sentence) > 10:  # Skip very short fragments
                result = self.ground_claim(sentence)
                results.append(result)

        return results

    def get_facts_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Retrieve all facts for a domain.

        Args:
            domain: Knowledge domain to query

        Returns:
            List of fact dictionaries
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT key, claim, source, confidence, evidence, verified_at
                   FROM facts WHERE domain = ?
                   ORDER BY confidence DESC, verified_at DESC""",
                (domain,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_memories_by_type(self, memory_type: str = "episodic") -> List[Dict[str, Any]]:
        """
        Retrieve memories by type.

        Args:
            memory_type: Type of memory to retrieve

        Returns:
            List of memory dictionaries
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT key, claim, context, created_at
                   FROM memories WHERE memory_type = ?
                   ORDER BY created_at DESC""",
                (memory_type,)
            )
            return [dict(row) for row in cursor.fetchall()]


# Module-level singleton
_kvrm_router_instance: KVRMRouter | None = None


def get_kvrm_router(db_path: Optional[Path] = None) -> KVRMRouter:
    """
    Get singleton instance of KVRMRouter.

    Args:
        db_path: Optional database path (only used on first call)

    Returns:
        Shared KVRMRouter instance
    """
    global _kvrm_router_instance
    if _kvrm_router_instance is None:
        _kvrm_router_instance = KVRMRouter(db_path)
    return _kvrm_router_instance

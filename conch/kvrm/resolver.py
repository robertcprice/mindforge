"""
Key Resolvers

Concrete implementations of KeyStore for different verified data sources.
"""

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from conch.kvrm.key_store import (
    KeyStore,
    KeyType,
    ResolvedContent,
)

logger = logging.getLogger(__name__)


class MemoryKeyStore(KeyStore):
    """
    Key store backed by Conch's memory system.

    Provides verified access to past thoughts, reflections, and experiences.
    Keys follow format: mem:{type}:{timestamp}:{hash}
    Examples:
        - mem:thought:20241209:a1b2c3
        - mem:reflection:20241209:d4e5f6
        - mem:learning:20241208:g7h8i9
    """

    KEY_PATTERN = re.compile(
        r"^mem:(thought|reflection|learning|interaction|pattern):(\d{8}):([a-f0-9]+)$"
    )

    def __init__(self, memory_store: Any):
        """
        Initialize with a Conch MemoryStore.

        Args:
            memory_store: Instance of conch.memory.store.MemoryStore
        """
        super().__init__("Conch Memory", KeyType.MEMORY, read_only=False)
        self.memory_store = memory_store
        self._key_cache: Dict[str, int] = {}  # key -> memory_id mapping

    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """Resolve a memory key to its content."""
        match = self.KEY_PATTERN.match(key)
        if not match:
            self._record_lookup(False)
            return None

        memory_type, date_str, content_hash = match.groups()

        # Try cache first
        if key in self._key_cache:
            memory_id = self._key_cache[key]
            memory = self.memory_store.get_by_id(memory_id)
            if memory:
                self._record_lookup(True)
                return self._memory_to_resolved(memory, key)

        # Search by content hash
        memories = self.memory_store.search(content_hash, limit=10)
        for memory in memories:
            if self._make_key(memory) == key:
                self._key_cache[key] = memory.id
                self._record_lookup(True)
                return self._memory_to_resolved(memory, key)

        self._record_lookup(False)
        return None

    def validate_key(self, key: str) -> bool:
        """Check if key matches pattern and exists."""
        if not self.KEY_PATTERN.match(key):
            return False
        return self.resolve(key) is not None

    def search(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[ResolvedContent]:
        """Search memories semantically.

        Note: Sanitizes query for FTS5 compatibility by removing special characters.
        """
        # Sanitize query for FTS5 - remove special characters that cause syntax errors
        sanitized_query = self._sanitize_fts_query(query)
        if not sanitized_query.strip():
            return []

        try:
            memories = self.memory_store.search(sanitized_query, limit=limit)
        except Exception as e:
            logger.warning(f"Memory search failed for query '{query}': {e}")
            return []

        results = []
        for memory in memories:
            key = self._make_key(memory)
            resolved = self._memory_to_resolved(memory, key)
            if resolved.confidence >= min_confidence:
                results.append(resolved)

        return results

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize a query string for FTS5 compatibility.

        FTS5 has special characters that cause syntax errors:
        - Commas, quotes, parentheses, operators
        - We extract just alphanumeric words for safe searching
        """
        import re
        # Extract alphanumeric words only, join with spaces
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query)
        return ' '.join(words)

    def _store_impl(
        self,
        key: str,
        content: str,
        source: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Store a new memory and return its key."""
        from conch.memory.store import Memory, MemoryType

        # Map key type to memory type
        type_map = {
            "thought": MemoryType.THOUGHT,
            "reflection": MemoryType.REFLECTION,
            "learning": MemoryType.LEARNING,
            "interaction": MemoryType.INTERACTION,
            "pattern": MemoryType.PATTERN,
        }

        # Parse key type from key or default to thought
        match = self.KEY_PATTERN.match(key) if key else None
        if match:
            memory_type = type_map.get(match.group(1), MemoryType.THOUGHT)
        else:
            memory_type = MemoryType.THOUGHT

        memory = Memory(
            content=content,
            memory_type=memory_type,
            source=source or "kvrm_store",
            importance=metadata.get("importance", 0.5),
            metadata=metadata,
        )

        memory_id = self.memory_store.store(memory)
        self._stats["stores"] += 1

        # Generate key for the stored memory
        stored_memory = self.memory_store.get_by_id(memory_id)
        if stored_memory:
            new_key = self._make_key(stored_memory)
            self._key_cache[new_key] = memory_id
            return new_key

        return None

    def _make_key(self, memory: Any) -> str:
        """Generate a key for a memory object."""
        # Get timestamp
        if hasattr(memory, 'timestamp'):
            date_str = memory.timestamp.strftime("%Y%m%d")
        else:
            date_str = datetime.now().strftime("%Y%m%d")

        # Get memory type
        if hasattr(memory, 'memory_type'):
            type_str = memory.memory_type.value
        else:
            type_str = "thought"

        # Content hash
        content = memory.content if hasattr(memory, 'content') else str(memory)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]

        return f"mem:{type_str}:{date_str}:{content_hash}"

    def _memory_to_resolved(self, memory: Any, key: str) -> ResolvedContent:
        """Convert a Memory object to ResolvedContent."""
        content = memory.content if hasattr(memory, 'content') else str(memory)

        return ResolvedContent(
            key=key,
            key_type=KeyType.MEMORY,
            content=content,
            source="conch_memory",
            confidence=memory.importance if hasattr(memory, 'importance') else 0.8,
            metadata={
                "memory_id": memory.id if hasattr(memory, 'id') else None,
                "memory_type": memory.memory_type.value if hasattr(memory, 'memory_type') else "unknown",
            },
        )


class FactKeyStore(KeyStore):
    """
    Key store for verified factual knowledge.

    Stores facts with verification metadata and provenance.
    Keys follow format: fact:{domain}:{id}
    Examples:
        - fact:science:gravity_001
        - fact:history:ww2_start
        - fact:math:pythagorean
    """

    KEY_PATTERN = re.compile(r"^fact:([a-z_]+):([a-z0-9_]+)$")

    def __init__(self, db_path: Path):
        """
        Initialize with SQLite database path.

        Args:
            db_path: Path to facts database
        """
        super().__init__("Verified Facts", KeyType.FACT, read_only=False)
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    verified_at TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)")

    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """Resolve a fact key to its content."""
        match = self.KEY_PATTERN.match(key)
        if not match:
            self._record_lookup(False)
            return None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content, content_hash, source, confidence, verified_at, metadata FROM facts WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

        if not row:
            self._record_lookup(False)
            return None

        content, content_hash, source, confidence, verified_at, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        self._record_lookup(True)
        return ResolvedContent(
            key=key,
            key_type=KeyType.FACT,
            content=content,
            content_hash=content_hash,
            source=source or "",
            confidence=confidence,
            verified_at=datetime.fromisoformat(verified_at) if verified_at else None,
            metadata=metadata,
        )

    def validate_key(self, key: str) -> bool:
        """Check if key exists in facts database."""
        if not self.KEY_PATTERN.match(key):
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM facts WHERE key = ?", (key,))
            return cursor.fetchone() is not None

    def search(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[ResolvedContent]:
        """Search facts by content (simple LIKE search)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT key, content, content_hash, source, confidence, verified_at, metadata
                FROM facts
                WHERE content LIKE ? AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (f"%{query}%", min_confidence, limit)
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            key, content, content_hash, source, confidence, verified_at, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            results.append(ResolvedContent(
                key=key,
                key_type=KeyType.FACT,
                content=content,
                content_hash=content_hash,
                source=source or "",
                confidence=confidence,
                verified_at=datetime.fromisoformat(verified_at) if verified_at else None,
                metadata=metadata,
            ))

        return results

    def _store_impl(
        self,
        key: str,
        content: str,
        source: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Store a new fact."""
        # Generate key if not provided
        if not key or not self.KEY_PATTERN.match(key):
            domain = metadata.get("domain", "general")
            fact_id = hashlib.sha256(content.encode()).hexdigest()[:12]
            key = f"fact:{domain}:{fact_id}"

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        verified_at = datetime.now().isoformat()
        confidence = metadata.pop("confidence", 1.0)
        metadata_json = json.dumps(metadata)

        # Extract domain from key
        match = self.KEY_PATTERN.match(key)
        domain = match.group(1) if match else "general"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts
                (key, domain, content, content_hash, source, confidence, verified_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key, domain, content, content_hash, source, confidence, verified_at, metadata_json)
            )

        self._stats["stores"] += 1
        return key


class ExternalKeyStore(KeyStore):
    """
    Key store for external verified sources.

    Supports pluggable backends for different data sources (Bible, docs, etc.).
    Keys follow format: ext:{source}:{domain}:{id}
    Examples:
        - ext:bible:nt:john:3:16
        - ext:docs:python:pathlib
        - ext:wiki:science:gravity
    """

    KEY_PATTERN = re.compile(r"^ext:([a-z]+):(.+)$")

    def __init__(self, name: str = "External Sources"):
        """Initialize external key store."""
        super().__init__(name, KeyType.EXTERNAL, read_only=True)
        self._backends: Dict[str, Any] = {}

    def register_backend(self, source: str, resolver: Any) -> None:
        """
        Register a backend resolver for a source.

        Args:
            source: Source name (e.g., "bible", "docs")
            resolver: Resolver object with resolve(key) method
        """
        self._backends[source] = resolver
        logger.info(f"Registered external backend: {source}")

    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """Resolve an external key through the appropriate backend."""
        match = self.KEY_PATTERN.match(key)
        if not match:
            self._record_lookup(False)
            return None

        source, sub_key = match.groups()

        if source not in self._backends:
            logger.warning(f"No backend registered for source: {source}")
            self._record_lookup(False)
            return None

        backend = self._backends[source]

        # Call backend resolver
        try:
            result = backend.resolve(sub_key)
            if result is None:
                self._record_lookup(False)
                return None

            # Convert backend result to ResolvedContent
            if isinstance(result, ResolvedContent):
                self._record_lookup(True)
                return result

            # Handle dict or object results
            content = result.text if hasattr(result, 'text') else str(result)
            citation = result.citation if hasattr(result, 'citation') else key

            self._record_lookup(True)
            return ResolvedContent(
                key=key,
                key_type=KeyType.EXTERNAL,
                content=content,
                source=source,
                confidence=1.0,
                metadata={"citation": citation},
            )

        except Exception as e:
            logger.error(f"Backend resolve error for {key}: {e}")
            self._record_lookup(False)
            return None

    def validate_key(self, key: str) -> bool:
        """Check if key can be resolved by a backend."""
        match = self.KEY_PATTERN.match(key)
        if not match:
            return False

        source = match.group(1)
        if source not in self._backends:
            return False

        return self.resolve(key) is not None

    def search(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[ResolvedContent]:
        """Search is not supported for external stores."""
        # External stores typically don't support semantic search
        # Individual backends could implement this
        return []


class KeyResolver:
    """
    Main resolver that coordinates multiple key stores.

    Routes keys to appropriate stores based on prefix and aggregates results.
    """

    def __init__(self):
        """Initialize the key resolver."""
        self.stores: Dict[str, KeyStore] = {}
        self._prefix_map: Dict[str, str] = {
            "mem:": "memory",
            "fact:": "fact",
            "ext:": "external",
        }

    def register_store(self, name: str, store: KeyStore) -> None:
        """
        Register a key store.

        Args:
            name: Store name for routing
            store: KeyStore instance
        """
        self.stores[name] = store
        logger.info(f"Registered key store: {name} ({store.key_type.value})")

    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """
        Resolve a key through the appropriate store.

        Args:
            key: Key to resolve

        Returns:
            ResolvedContent if found, None otherwise
        """
        # Handle special keys
        if key in ("UNKNOWN", "AMBIGUOUS"):
            return ResolvedContent.unknown(key)

        # Route to appropriate store
        for prefix, store_name in self._prefix_map.items():
            if key.startswith(prefix):
                store = self.stores.get(store_name)
                if store:
                    return store.resolve(key)
                break

        # Try all stores as fallback
        for store in self.stores.values():
            result = store.resolve(key)
            if result:
                return result

        return None

    def validate(self, key: str) -> bool:
        """Check if a key is valid in any store."""
        if key in ("UNKNOWN", "AMBIGUOUS"):
            return True

        for prefix, store_name in self._prefix_map.items():
            if key.startswith(prefix):
                store = self.stores.get(store_name)
                if store:
                    return store.validate_key(key)
                return False

        return any(store.validate_key(key) for store in self.stores.values())

    def search(
        self,
        query: str,
        limit: int = 5,
        store_names: Optional[List[str]] = None,
    ) -> List[ResolvedContent]:
        """
        Search across stores for relevant content.

        Args:
            query: Search query
            limit: Maximum results
            store_names: Specific stores to search (None = all)

        Returns:
            List of matching ResolvedContent items
        """
        results = []
        stores_to_search = (
            [self.stores[n] for n in store_names if n in self.stores]
            if store_names
            else list(self.stores.values())
        )

        for store in stores_to_search:
            store_results = store.search(query, limit=limit)
            results.extend(store_results)

        # Sort by confidence and deduplicate
        results.sort(key=lambda x: x.confidence, reverse=True)
        seen_keys = set()
        unique_results = []
        for r in results:
            if r.key not in seen_keys:
                seen_keys.add(r.key)
                unique_results.append(r)

        return unique_results[:limit]

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics from all stores."""
        return {name: store.get_stats() for name, store in self.stores.items()}

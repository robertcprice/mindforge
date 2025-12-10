"""
Key Store Base Classes

Abstract interfaces for verified data stores that support zero-hallucination
retrieval through key-based routing.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of keys in the KVRM system."""

    # Internal stores
    MEMORY = "memory"           # Past thoughts, reflections, experiences
    FACT = "fact"               # Verified factual knowledge
    PATTERN = "pattern"         # Recognized behavioral patterns

    # External stores
    EXTERNAL = "external"       # External verified sources (Bible, docs, etc.)
    TOOL_RESULT = "tool_result" # Cached tool execution results

    # Special keys
    UNKNOWN = "unknown"         # Key not found / invalid query
    AMBIGUOUS = "ambiguous"     # Multiple valid answers exist


@dataclass
class ResolvedContent:
    """
    Resolved content from a key store.

    Includes verification metadata for provable accuracy.
    """

    key: str
    key_type: KeyType
    content: str

    # Verification
    content_hash: str = ""          # SHA256 of content for integrity
    source: str = ""                # Where this content came from
    verified_at: Optional[datetime] = None

    # Metadata
    confidence: float = 1.0         # 1.0 = fully verified, <1.0 = uncertain
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(
                self.content.encode('utf-8')
            ).hexdigest()[:16]
        if self.verified_at is None:
            self.verified_at = datetime.now()

    @property
    def is_verified(self) -> bool:
        """Check if content is fully verified."""
        return self.confidence >= 1.0

    @property
    def citation(self) -> str:
        """Human-readable citation."""
        return f"[{self.key_type.value}:{self.key}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "key_type": self.key_type.value,
            "content": self.content,
            "content_hash": self.content_hash,
            "source": self.source,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "related_keys": self.related_keys,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedContent":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            key_type=KeyType(data["key_type"]),
            content=data["content"],
            content_hash=data.get("content_hash", ""),
            source=data.get("source", ""),
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            related_keys=data.get("related_keys", []),
        )

    @classmethod
    def unknown(cls, query: str) -> "ResolvedContent":
        """Create an UNKNOWN response for invalid queries."""
        return cls(
            key="UNKNOWN",
            key_type=KeyType.UNKNOWN,
            content=f"No verified content found for: {query}",
            source="system",
            confidence=1.0,
            metadata={"original_query": query},
        )

    @classmethod
    def ambiguous(cls, query: str, candidates: List[str]) -> "ResolvedContent":
        """Create an AMBIGUOUS response when multiple answers exist."""
        return cls(
            key="AMBIGUOUS",
            key_type=KeyType.AMBIGUOUS,
            content=f"Multiple valid answers exist for: {query}",
            source="system",
            confidence=1.0,
            metadata={"original_query": query, "candidates": candidates},
            related_keys=candidates,
        )


class KeyStore(ABC):
    """
    Abstract base class for verified key-value stores.

    Implementations must provide:
    - resolve(): Look up a key and return verified content
    - validate_key(): Check if a key exists
    - store(): Add new verified content (if writable)
    - search(): Semantic search for relevant keys
    """

    def __init__(self, name: str, key_type: KeyType, read_only: bool = False):
        """
        Initialize key store.

        Args:
            name: Human-readable store name
            key_type: Type of keys this store handles
            read_only: If True, store() operations are disabled
        """
        self.name = name
        self.key_type = key_type
        self.read_only = read_only
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
        }

    @abstractmethod
    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """
        Resolve a key to verified content.

        Args:
            key: The key to look up

        Returns:
            ResolvedContent if found, None if not found
        """
        pass

    @abstractmethod
    def validate_key(self, key: str) -> bool:
        """
        Check if a key is valid in this store.

        Args:
            key: The key to validate

        Returns:
            True if key exists and is valid
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[ResolvedContent]:
        """
        Search for relevant content by semantic query.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching ResolvedContent items
        """
        pass

    def store(
        self,
        key: str,
        content: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store new verified content.

        Args:
            key: Key to store under (or None to auto-generate)
            content: The verified content
            source: Source of the content
            metadata: Additional metadata

        Returns:
            The key used for storage, or None if read-only
        """
        if self.read_only:
            logger.warning(f"Attempted to store in read-only store: {self.name}")
            return None

        return self._store_impl(key, content, source, metadata or {})

    def _store_impl(
        self,
        key: str,
        content: str,
        source: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Implementation of store - override in subclasses."""
        raise NotImplementedError("Writable stores must implement _store_impl")

    def get_stats(self) -> Dict[str, int]:
        """Get store statistics."""
        return self._stats.copy()

    def _record_lookup(self, hit: bool) -> None:
        """Record a lookup for statistics."""
        self._stats["lookups"] += 1
        if hit:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.key_type.value})"


class CompositeKeyStore(KeyStore):
    """
    A key store that delegates to multiple underlying stores.

    Useful for creating a unified interface over multiple data sources.
    """

    def __init__(
        self,
        name: str,
        stores: List[KeyStore],
        key_type: KeyType = KeyType.EXTERNAL,
    ):
        """
        Initialize composite store.

        Args:
            name: Name for this composite store
            stores: List of underlying key stores
            key_type: Default key type for this store
        """
        super().__init__(name, key_type, read_only=True)
        self.stores = stores
        self._store_map: Dict[str, KeyStore] = {}

        # Build prefix-to-store mapping
        for store in stores:
            prefix = store.key_type.value
            self._store_map[prefix] = store

    def resolve(self, key: str) -> Optional[ResolvedContent]:
        """Resolve by trying each store in order."""
        for store in self.stores:
            result = store.resolve(key)
            if result:
                self._record_lookup(True)
                return result

        self._record_lookup(False)
        return None

    def validate_key(self, key: str) -> bool:
        """Check if key is valid in any store."""
        return any(store.validate_key(key) for store in self.stores)

    def search(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[ResolvedContent]:
        """Search across all stores and merge results."""
        all_results = []
        per_store_limit = max(1, limit // len(self.stores))

        for store in self.stores:
            results = store.search(query, limit=per_store_limit, min_confidence=min_confidence)
            all_results.extend(results)

        # Sort by confidence and return top results
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results[:limit]

    def add_store(self, store: KeyStore) -> None:
        """Add a new store to the composite."""
        self.stores.append(store)
        self._store_map[store.key_type.value] = store

    def get_store(self, key_type: KeyType) -> Optional[KeyStore]:
        """Get a specific store by key type."""
        return self._store_map.get(key_type.value)

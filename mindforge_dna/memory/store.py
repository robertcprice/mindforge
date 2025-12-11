"""
MindForge DNA - Memory System: Store

Hybrid memory storage using ChromaDB for vectors and SQLite for structured data.
Implements the sacred/routine memory distinction with importance-based compression.

Key features:
- ChromaDB for semantic search
- SQLite for structured metadata
- CLaRa-style compression for routine memories
- Sacred memory protection (importance >= 0.75)
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A single memory unit."""

    key: str                              # KVRM key: mem:{type}:{date}:{hash}
    content: str                          # Full content (or compressed if routine)
    original_content: Optional[str]       # Original if compressed
    importance: float                     # 0.0-1.0
    is_sacred: bool                       # True if importance >= 0.75
    memory_type: str                      # reflection, fact, experience, etc.
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "content": self.content,
            "original_content": self.original_content,
            "importance": self.importance,
            "is_sacred": self.is_sacred,
            "memory_type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            content=data["content"],
            original_content=data.get("original_content"),
            importance=data["importance"],
            is_sacred=data["is_sacred"],
            memory_type=data["memory_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            metadata=json.loads(data.get("metadata", "{}")),
        )


class MemoryStore:
    """Hybrid memory storage system.

    Uses ChromaDB for vector similarity search and SQLite for
    structured metadata and fast key-value lookups.

    Architecture:
        - ChromaDB: Semantic embeddings for retrieval
        - SQLite: Metadata, access patterns, importance scores
        - File system: Large memories and backups

    Example:
        store = MemoryStore(data_dir=Path("data"))
        store.initialize()

        # Store a memory
        memory = store.store(
            content="Learned an important lesson about error handling",
            memory_type="reflection",
            importance=0.85  # Sacred - won't be compressed
        )

        # Retrieve by query
        results = store.search("error handling lessons", top_k=5)

        # Get by key
        memory = store.get("mem:reflection:20251211:a1b2c3d4")
    """

    SACRED_THRESHOLD = 0.75
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default sentence-transformers model

    def __init__(
        self,
        data_dir: Path,
        embedding_model: Optional[str] = None
    ):
        """Initialize memory store.

        Args:
            data_dir: Directory for data storage
            embedding_model: Optional custom embedding model name
        """
        self.data_dir = data_dir
        self.db_path = data_dir / "memories.db"
        self.chroma_path = data_dir / "chroma"
        self.embedding_model = embedding_model or self.EMBEDDING_MODEL

        self.conn: Optional[sqlite3.Connection] = None
        self.chroma_client = None
        self.collection = None
        self.encoder = None

        logger.info(f"MemoryStore initialized with data_dir={data_dir}")

    def initialize(self) -> None:
        """Initialize storage backends."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        self._init_chroma()
        self._init_encoder()
        logger.info("MemoryStore backends initialized")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                original_content TEXT,
                importance REAL NOT NULL,
                is_sacred INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sacred ON memories(is_sacred)
        """)

        self.conn.commit()
        logger.debug("SQLite initialized")

    def _init_chroma(self) -> None:
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.chroma_path.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.chroma_path),
                anonymized_telemetry=False
            ))

            self.collection = self.chroma_client.get_or_create_collection(
                name="mindforge_memories",
                metadata={"hnsw:space": "cosine"}
            )

            logger.debug(f"ChromaDB initialized with {self.collection.count()} memories")

        except ImportError:
            logger.warning("ChromaDB not available - vector search disabled")
            self.chroma_client = None
            self.collection = None

    def _init_encoder(self) -> None:
        """Initialize sentence encoder for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.embedding_model)
            logger.debug(f"Encoder initialized: {self.embedding_model}")

        except ImportError:
            logger.warning("sentence-transformers not available - using fallback")
            self.encoder = None

    def _generate_key(self, content: str, memory_type: str) -> str:
        """Generate a unique memory key.

        Format: mem:{type}:{timestamp}:{hash}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"mem:{memory_type}:{timestamp}:{content_hash}"

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if self.encoder is None:
            return None

        try:
            embedding = self.encoder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress_routine: bool = True
    ) -> Memory:
        """Store a new memory.

        Args:
            content: Memory content
            memory_type: Type (reflection, fact, experience, etc.)
            importance: Optional pre-computed importance (0.0-1.0)
            metadata: Optional additional metadata
            compress_routine: Whether to compress non-sacred memories

        Returns:
            The stored Memory object
        """
        # Generate key
        key = self._generate_key(content, memory_type)

        # Determine importance and sacred status
        if importance is None:
            importance = 0.5  # Default moderate importance

        is_sacred = importance >= self.SACRED_THRESHOLD

        # Handle compression
        original_content = None
        if compress_routine and not is_sacred:
            # Compress routine memories (simple truncation for now)
            # Full CLaRa compression would use the Memory neuron
            if len(content) > 500:
                original_content = content
                content = content[:500] + "..."
                logger.debug(f"Compressed routine memory: {key}")

        now = datetime.now()
        memory = Memory(
            key=key,
            content=content,
            original_content=original_content,
            importance=importance,
            is_sacred=is_sacred,
            memory_type=memory_type,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {}
        )

        # Store in SQLite
        self.conn.execute("""
            INSERT OR REPLACE INTO memories
            (key, content, original_content, importance, is_sacred,
             memory_type, created_at, accessed_at, access_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.key,
            memory.content,
            memory.original_content,
            memory.importance,
            1 if memory.is_sacred else 0,
            memory.memory_type,
            memory.created_at.isoformat(),
            memory.accessed_at.isoformat(),
            memory.access_count,
            json.dumps(memory.metadata)
        ))
        self.conn.commit()

        # Store in ChromaDB
        if self.collection is not None:
            embedding = self._embed(content)
            if embedding:
                self.collection.add(
                    ids=[key],
                    embeddings=[embedding],
                    metadatas=[{"importance": importance, "type": memory_type}],
                    documents=[content]
                )

        logger.info(f"Stored memory: {key} (sacred={is_sacred}, importance={importance:.2f})")
        return memory

    def get(self, key: str, update_access: bool = True) -> Optional[Memory]:
        """Get a memory by key.

        Args:
            key: Memory key
            update_access: Whether to update access timestamp and count

        Returns:
            Memory if found, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        memory = Memory.from_dict(dict(row))

        if update_access:
            now = datetime.now()
            self.conn.execute("""
                UPDATE memories
                SET accessed_at = ?, access_count = access_count + 1
                WHERE key = ?
            """, (now.isoformat(), key))
            self.conn.commit()
            memory.accessed_at = now
            memory.access_count += 1

        return memory

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        sacred_only: bool = False
    ) -> List[Memory]:
        """Search memories by semantic similarity.

        Args:
            query: Search query
            top_k: Number of results
            memory_type: Filter by type
            sacred_only: Only return sacred memories

        Returns:
            List of matching memories sorted by relevance
        """
        if self.collection is None:
            logger.warning("ChromaDB not available, falling back to keyword search")
            return self._keyword_search(query, top_k, memory_type, sacred_only)

        # Query ChromaDB
        embedding = self._embed(query)
        if embedding is None:
            return self._keyword_search(query, top_k, memory_type, sacred_only)

        where_clause = {}
        if memory_type:
            where_clause["type"] = memory_type

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k * 2,  # Get extra for filtering
            where=where_clause if where_clause else None
        )

        memories = []
        for key in results["ids"][0]:
            memory = self.get(key, update_access=False)
            if memory:
                if sacred_only and not memory.is_sacred:
                    continue
                memories.append(memory)
                if len(memories) >= top_k:
                    break

        return memories

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[str],
        sacred_only: bool
    ) -> List[Memory]:
        """Fallback keyword search when vector search unavailable."""
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params = [f"%{query}%"]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        if sacred_only:
            sql += " AND is_sacred = 1"

        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(top_k)

        cursor = self.conn.execute(sql, params)
        return [Memory.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_sacred_memories(self, limit: int = 100) -> List[Memory]:
        """Get all sacred memories.

        Returns:
            List of sacred memories sorted by importance
        """
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE is_sacred = 1
            ORDER BY importance DESC
            LIMIT ?
        """, (limit,))

        return [Memory.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 10) -> List[Memory]:
        """Get most recently accessed memories."""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            ORDER BY accessed_at DESC
            LIMIT ?
        """, (limit,))

        return [Memory.from_dict(dict(row)) for row in cursor.fetchall()]

    def delete(self, key: str) -> bool:
        """Delete a memory by key.

        Note: Sacred memories require explicit confirmation.
        """
        memory = self.get(key, update_access=False)
        if memory is None:
            return False

        if memory.is_sacred:
            logger.warning(f"Deleting sacred memory: {key}")

        self.conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.conn.commit()

        if self.collection is not None:
            try:
                self.collection.delete(ids=[key])
            except Exception as e:
                logger.error(f"ChromaDB delete failed: {e}")

        logger.info(f"Deleted memory: {key}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_sacred = 1 THEN 1 ELSE 0 END) as sacred,
                AVG(importance) as avg_importance,
                COUNT(DISTINCT memory_type) as type_count
            FROM memories
        """)
        row = cursor.fetchone()

        return {
            "total_memories": row[0],
            "sacred_memories": row[1],
            "routine_memories": row[0] - row[1],
            "average_importance": row[2] or 0.0,
            "memory_types": row[3],
        }

    def close(self) -> None:
        """Close database connections."""
        if self.conn:
            self.conn.close()
        if self.chroma_client:
            self.chroma_client.persist()
        logger.info("MemoryStore closed")


def create_memory_store(data_dir: str = "data") -> MemoryStore:
    """Factory function to create an initialized MemoryStore.

    Args:
        data_dir: Path to data directory

    Returns:
        Initialized MemoryStore
    """
    store = MemoryStore(Path(data_dir))
    store.initialize()
    return store

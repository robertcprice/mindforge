"""
Conch Memory Store

SQLite-based structured memory storage for Conch.
Stores memories with metadata, timestamps, and relationships.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories stored."""
    INTERACTION = "interaction"      # User interaction
    THOUGHT = "thought"              # Generated thought
    REFLECTION = "reflection"        # Post-interaction reflection
    LEARNING = "learning"            # Something learned
    FACT = "fact"                    # Factual information
    PREFERENCE = "preference"        # User preference
    PATTERN = "pattern"              # Detected pattern
    SYSTEM = "system"                # System event


@dataclass
class Memory:
    """Represents a single memory item."""
    id: Optional[int] = None
    content: str = ""
    memory_type: MemoryType = MemoryType.INTERACTION
    timestamp: datetime = field(default_factory=datetime.now)

    # Metadata
    source: str = ""                 # Where this came from
    importance: float = 0.5          # 0-1, affects retention
    access_count: int = 0            # How often accessed
    last_accessed: Optional[datetime] = None

    # Relationships
    related_to: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Additional metadata as JSON
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "related_to": self.related_to,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            related_to=data.get("related_to", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        """Create from SQLite row."""
        return cls(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            source=row["source"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            related_to=json.loads(row["related_to"]) if row["related_to"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class MemoryStore:
    """SQLite-based memory storage.

    Provides CRUD operations for memories with:
    - Full-text search
    - Filtering by type, tags, time range
    - Automatic importance decay
    - Memory consolidation
    """

    def __init__(self, db_path: Path):
        """Initialize memory store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"MemoryStore initialized at {db_path}")

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Main memories table
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    related_to TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

                -- Full-text search
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id'
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;

                -- Settings table for memory system config
                CREATE TABLE IF NOT EXISTS memory_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                -- Statistics table
                CREATE TABLE IF NOT EXISTS memory_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store(self, memory: Memory) -> int:
        """Store a new memory.

        Args:
            memory: Memory to store

        Returns:
            ID of stored memory
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    content, memory_type, timestamp, source, importance,
                    access_count, last_accessed, related_to, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.content,
                    memory.memory_type.value,
                    memory.timestamp.isoformat(),
                    memory.source,
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    json.dumps(memory.related_to),
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                )
            )
            memory_id = cursor.lastrowid

        logger.debug(f"Stored memory {memory_id}: {memory.content[:50]}...")
        return memory_id

    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a memory by ID.

        Args:
            memory_id: ID of memory to retrieve

        Returns:
            Memory if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()

            if row:
                # Update access count
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                    """,
                    (datetime.now().isoformat(), memory_id)
                )
                return Memory.from_row(row)

        return None

    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        min_importance: float = 0.0,
        limit: int = 20,
    ) -> list[Memory]:
        """Search memories using full-text search and filters.

        Args:
            query: Search query (uses FTS5 syntax)
            memory_type: Filter by type
            tags: Filter by tags (any match)
            since: Only memories after this time
            until: Only memories before this time
            min_importance: Minimum importance threshold
            limit: Maximum results

        Returns:
            List of matching memories
        """
        conditions = []
        params = []

        # Full-text search
        if query:
            conditions.append(
                "id IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?)"
            )
            params.append(query)

        # Type filter
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        # Time filters
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        # Importance filter
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        # Build query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            memories = [Memory.from_row(row) for row in rows]

        # Filter by tags in Python (JSON field)
        if tags:
            memories = [
                m for m in memories
                if any(t in m.tags for t in tags)
            ]

        return memories

    def get_recent(
        self,
        count: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> list[Memory]:
        """Get most recent memories.

        Args:
            count: Number of memories to retrieve
            memory_type: Optional type filter

        Returns:
            List of recent memories
        """
        sql = "SELECT * FROM memories"
        params = []

        if memory_type:
            sql += " WHERE memory_type = ?"
            params.append(memory_type.value)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(count)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [Memory.from_row(row) for row in rows]

    def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[Memory]:
        """Get memories of a specific type.

        Args:
            memory_type: Type to filter by
            limit: Maximum results

        Returns:
            List of memories
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE memory_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (memory_type.value, limit)
            ).fetchall()
            return [Memory.from_row(row) for row in rows]

    def update_importance(
        self,
        memory_id: int,
        new_importance: float,
    ) -> None:
        """Update memory importance.

        Args:
            memory_id: ID of memory
            new_importance: New importance value (0-1)
        """
        new_importance = max(0.0, min(1.0, new_importance))

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (new_importance, memory_id)
            )

    def add_tags(self, memory_id: int, tags: list[str]) -> None:
        """Add tags to a memory.

        Args:
            memory_id: ID of memory
            tags: Tags to add
        """
        memory = self.get(memory_id)
        if memory:
            new_tags = list(set(memory.tags + tags))
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE id = ?",
                    (json.dumps(new_tags), memory_id)
                )

    def link_memories(self, id1: int, id2: int) -> None:
        """Create a bidirectional link between two memories.

        Args:
            id1: First memory ID
            id2: Second memory ID
        """
        mem1 = self.get(id1)
        mem2 = self.get(id2)

        if mem1 and mem2:
            # Add links
            if id2 not in mem1.related_to:
                mem1.related_to.append(id2)
            if id1 not in mem2.related_to:
                mem2.related_to.append(id1)

            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET related_to = ? WHERE id = ?",
                    (json.dumps(mem1.related_to), id1)
                )
                conn.execute(
                    "UPDATE memories SET related_to = ? WHERE id = ?",
                    (json.dumps(mem2.related_to), id2)
                )

    def delete(self, memory_id: int) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )
            return cursor.rowcount > 0

    def decay_importance(
        self,
        decay_factor: float = 0.99,
        min_importance: float = 0.1,
    ) -> int:
        """Apply decay to all memory importance values.

        Called periodically to simulate forgetting.

        Args:
            decay_factor: Multiplier for importance (0.99 = 1% decay)
            min_importance: Floor value for importance

        Returns:
            Number of memories affected
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET importance = MAX(?, importance * ?)
                WHERE importance > ?
                """,
                (min_importance, decay_factor, min_importance)
            )
            return cursor.rowcount

    def cleanup_old(
        self,
        max_age_days: int = 30,
        min_importance: float = 0.2,
    ) -> int:
        """Remove old, low-importance memories.

        Args:
            max_age_days: Delete memories older than this
            min_importance: Only delete if importance below this

        Returns:
            Number of memories deleted
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE timestamp < ? AND importance < ?
                """,
                (cutoff, min_importance)
            )
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old memories")

        return deleted

    def get_statistics(self) -> dict:
        """Get memory statistics.

        Returns:
            Dictionary with stats
        """
        with self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as count FROM memories"
            ).fetchone()["count"]

            by_type = {}
            for row in conn.execute(
                "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
            ).fetchall():
                by_type[row["memory_type"]] = row["count"]

            avg_importance = conn.execute(
                "SELECT AVG(importance) as avg FROM memories"
            ).fetchone()["avg"] or 0

            recent_count = conn.execute(
                """
                SELECT COUNT(*) as count FROM memories
                WHERE timestamp > datetime('now', '-1 day')
                """
            ).fetchone()["count"]

        return {
            "total_memories": total,
            "by_type": by_type,
            "average_importance": avg_importance,
            "memories_last_24h": recent_count,
        }

    def export_all(self) -> list[dict]:
        """Export all memories as JSON-serializable list.

        Returns:
            List of memory dictionaries
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY timestamp"
            ).fetchall()
            return [Memory.from_row(row).to_dict() for row in rows]

    def import_memories(self, memories: list[dict]) -> int:
        """Import memories from list of dictionaries.

        Args:
            memories: List of memory dicts

        Returns:
            Number imported
        """
        count = 0
        for data in memories:
            try:
                memory = Memory.from_dict(data)
                memory.id = None  # Force new ID
                self.store(memory)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")

        logger.info(f"Imported {count} memories")
        return count

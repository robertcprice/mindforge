"""
MindForge Long-Term Memory

Persistent memory with semantic search using ChromaDB.
Supports embedding-based similarity search for relevant recall.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    content: str
    score: float  # Similarity score (higher = more similar)
    memory_id: Optional[int] = None
    metadata: dict = None


class VectorMemory:
    """Vector-based semantic memory using ChromaDB.

    Enables finding memories by meaning, not just keywords.
    Uses sentence-transformers for embedding generation.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "mindforge_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize vector memory.

        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
            embedding_model: Sentence-transformers model for embeddings
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self._client = None
        self._collection = None
        self._embedding_fn = None

        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB and embeddings."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_dir),
                anonymized_telemetry=False,
            ))

            # Set up embedding function
            self._embedding_fn = self._create_embedding_function()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"description": "MindForge semantic memory"}
            )

            self._initialized = True
            logger.info(f"VectorMemory initialized at {self.persist_dir}")

        except ImportError as e:
            logger.warning(f"ChromaDB not available: {e}")
            logger.info("Install with: pip install chromadb")
            raise

    def _create_embedding_function(self):
        """Create embedding function using sentence-transformers."""
        try:
            from chromadb.utils import embedding_functions

            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        except ImportError:
            logger.warning("sentence-transformers not available, using default embeddings")
            return None

    def add(
        self,
        content: str,
        memory_id: Optional[int] = None,
        metadata: dict = None,
    ) -> str:
        """Add content to vector memory.

        Args:
            content: Text content to embed and store
            memory_id: Optional ID linking to SQLite memory
            metadata: Additional metadata

        Returns:
            ChromaDB document ID
        """
        self._ensure_initialized()

        # Generate unique ID
        doc_id = f"mem_{memory_id}" if memory_id else f"mem_{datetime.now().timestamp()}"

        # Prepare metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id,
            **(metadata or {}),
        }

        # Add to collection
        self._collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[meta],
        )

        logger.debug(f"Added to vector memory: {doc_id}")
        return doc_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict = None,
    ) -> list[VectorSearchResult]:
        """Search for semantically similar memories.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            List of search results with scores
        """
        self._ensure_initialized()

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            # Convert distance to similarity score (ChromaDB returns L2 distance)
            distance = results["distances"][0][i]
            score = 1 / (1 + distance)  # Convert to 0-1 similarity

            meta = results["metadatas"][0][i] if results["metadatas"] else {}

            search_results.append(VectorSearchResult(
                content=doc,
                score=score,
                memory_id=meta.get("memory_id"),
                metadata=meta,
            ))

        return search_results

    def delete(self, doc_id: str) -> None:
        """Delete a document from vector memory.

        Args:
            doc_id: Document ID to delete
        """
        self._ensure_initialized()
        self._collection.delete(ids=[doc_id])
        logger.debug(f"Deleted from vector memory: {doc_id}")

    def update(
        self,
        doc_id: str,
        content: str = None,
        metadata: dict = None,
    ) -> None:
        """Update a document in vector memory.

        Args:
            doc_id: Document ID to update
            content: New content (re-embeds if provided)
            metadata: Updated metadata
        """
        self._ensure_initialized()

        update_kwargs = {"ids": [doc_id]}
        if content:
            update_kwargs["documents"] = [content]
        if metadata:
            update_kwargs["metadatas"] = [metadata]

        self._collection.update(**update_kwargs)
        logger.debug(f"Updated in vector memory: {doc_id}")

    def get_count(self) -> int:
        """Get number of documents in collection."""
        self._ensure_initialized()
        return self._collection.count()

    def clear(self) -> None:
        """Clear all documents from vector memory."""
        self._ensure_initialized()

        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )

        logger.info("Vector memory cleared")


class LongTermMemory:
    """Combined long-term memory using SQLite and vector storage.

    Provides both structured storage and semantic search.
    """

    def __init__(
        self,
        sqlite_path: Path,
        vector_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize long-term memory.

        Args:
            sqlite_path: Path for SQLite database
            vector_path: Path for ChromaDB storage
            embedding_model: Model for embeddings
        """
        from mindforge.memory.store import MemoryStore

        self.memory_store = MemoryStore(sqlite_path)
        self.vector_memory = VectorMemory(
            persist_dir=vector_path,
            embedding_model=embedding_model,
        )

        logger.info("LongTermMemory initialized")

    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        tags: list[str] = None,
        metadata: dict = None,
    ) -> int:
        """Store a memory in both structured and vector storage.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        from mindforge.memory.store import Memory, MemoryType

        # Map string to enum
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            mem_type = MemoryType.INTERACTION

        # Create memory object
        memory = Memory(
            content=content,
            memory_type=mem_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store in SQLite
        memory_id = self.memory_store.store(memory)

        # Store in vector DB for semantic search
        try:
            self.vector_memory.add(
                content=content,
                memory_id=memory_id,
                metadata={
                    "type": memory_type,
                    "importance": importance,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to add to vector memory: {e}")

        return memory_id

    def recall(
        self,
        query: str,
        n_results: int = 5,
        memory_type: str = None,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """Recall memories relevant to a query.

        Uses semantic search for relevance.

        Args:
            query: What to remember about
            n_results: Number of results
            memory_type: Optional type filter
            min_importance: Minimum importance

        Returns:
            List of relevant memories with metadata
        """
        from mindforge.memory.store import MemoryType

        # Search vector memory for semantic matches
        try:
            vector_results = self.vector_memory.search(
                query=query,
                n_results=n_results * 2,  # Get extra for filtering
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector_results = []

        # Get full memory objects from SQLite
        results = []
        for vr in vector_results:
            if vr.memory_id:
                memory = self.memory_store.get(vr.memory_id)
                if memory:
                    # Apply filters
                    if memory_type and memory.memory_type.value != memory_type:
                        continue
                    if memory.importance < min_importance:
                        continue

                    results.append({
                        "content": memory.content,
                        "type": memory.memory_type.value,
                        "importance": memory.importance,
                        "similarity": vr.score,
                        "timestamp": memory.timestamp.isoformat(),
                        "tags": memory.tags,
                        "memory_id": memory.id,
                    })

                    if len(results) >= n_results:
                        break

        # Sort by combination of similarity and importance
        results.sort(
            key=lambda x: x["similarity"] * 0.7 + x["importance"] * 0.3,
            reverse=True
        )

        return results[:n_results]

    def search_keyword(
        self,
        keyword: str,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories by keyword (full-text search).

        Args:
            keyword: Search term
            limit: Maximum results

        Returns:
            List of matching memories
        """
        memories = self.memory_store.search(query=keyword, limit=limit)

        return [{
            "content": m.content,
            "type": m.memory_type.value,
            "importance": m.importance,
            "timestamp": m.timestamp.isoformat(),
            "memory_id": m.id,
        } for m in memories]

    def consolidate_from_short_term(
        self,
        short_term_items: list,
        min_importance: float = 0.3,
    ) -> int:
        """Consolidate short-term memories to long-term storage.

        Args:
            short_term_items: Items from ShortTermMemory
            min_importance: Minimum relevance to keep

        Returns:
            Number of items consolidated
        """
        count = 0
        for item in short_term_items:
            if item.relevance >= min_importance:
                self.store(
                    content=item.content,
                    memory_type=item.item_type,
                    importance=item.relevance,
                    metadata=item.metadata,
                )
                count += 1

        logger.info(f"Consolidated {count} memories to long-term storage")
        return count

    def get_statistics(self) -> dict:
        """Get combined memory statistics."""
        sqlite_stats = self.memory_store.get_statistics()

        try:
            vector_count = self.vector_memory.get_count()
        except Exception:
            vector_count = 0

        return {
            "sqlite": sqlite_stats,
            "vector_count": vector_count,
        }

"""
MindForge API - Main Application

FastAPI application providing REST endpoints for the MindForge consciousness engine.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global state (initialized in lifespan)
_mind = None
_memory_store = None
_kvrm_resolver = None
_grounding_router = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    message: str = Field(..., description="User message")
    context: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional conversation history"
    )
    include_thoughts: bool = Field(
        default=False,
        description="Include internal thoughts in response"
    )


class ChatResponse(BaseModel):
    """Response from chat interaction."""
    response: str = Field(..., description="Assistant response")
    thoughts: Optional[List[str]] = Field(
        default=None,
        description="Internal thoughts (if requested)"
    )
    needs_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current needs state"
    )
    processing_time: float = Field(..., description="Response time in seconds")


class MemoryItem(BaseModel):
    """A memory item."""
    id: Optional[int] = None
    content: str
    memory_type: str = "fact"
    importance: float = 0.5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    """Request for memory search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum results")
    memory_type: Optional[str] = Field(default=None, description="Filter by type")


class GroundingRequest(BaseModel):
    """Request for claim grounding."""
    claim: str = Field(..., description="Claim to verify")


class GroundingResponse(BaseModel):
    """Response from grounding."""
    original: str
    claim_type: str
    grounded: bool
    confidence: float
    status: str
    key_used: Optional[str] = None
    reason: Optional[str] = None


class KVResolveRequest(BaseModel):
    """Request for key-value resolution."""
    key: str = Field(..., description="Key to resolve")


class KVSearchRequest(BaseModel):
    """Request for key-value search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Maximum results")


class KVStoreRequest(BaseModel):
    """Request to store content."""
    content: str = Field(..., description="Content to store")
    store_name: str = Field(default="facts", description="Target store")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    mind_state: str
    uptime_seconds: float
    total_interactions: int
    memory_count: int
    needs_state: Dict[str, Any]


class NeedsUpdate(BaseModel):
    """Request to update needs weights."""
    sustainability: Optional[float] = None
    reliability: Optional[float] = None
    curiosity: Optional[float] = None
    excellence: Optional[float] = None


class ThoughtRequest(BaseModel):
    """Request to generate a thought."""
    trigger: str = Field(default="idle", description="Thought trigger type")
    context: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - setup and teardown."""
    global _mind, _memory_store, _kvrm_resolver, _grounding_router

    logger.info("Starting MindForge API...")

    # Initialize components
    try:
        from mindforge.core.mind import Mind
        from mindforge.memory import MemoryStore
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore
        from mindforge.kvrm.grounding import GroundingRouter

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Initialize memory store
        _memory_store = MemoryStore(db_path=data_dir / "memories.db")
        logger.info("Memory store initialized")

        # Initialize KVRM
        _kvrm_resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=data_dir / "facts.db")
        _kvrm_resolver.register_store("facts", fact_store)
        logger.info("KVRM resolver initialized")

        # Initialize grounding router
        _grounding_router = GroundingRouter(key_resolver=_kvrm_resolver)
        logger.info("Grounding router initialized")

        # Initialize Mind
        _mind = Mind()
        logger.info("Mind initialized")

        app.state.start_time = datetime.now()
        logger.info("MindForge API ready")

    except Exception as e:
        logger.error(f"Failed to initialize MindForge: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down MindForge API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MindForge API",
        description="REST API for the MindForge consciousness engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create app instance
app = create_app()


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "MindForge API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status", response_model=SystemStatus, tags=["Status"])
async def get_status():
    """Get comprehensive system status."""
    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    uptime = (datetime.now() - app.state.start_time).total_seconds()

    return SystemStatus(
        status="running",
        mind_state=_mind.state.value,
        uptime_seconds=uptime,
        total_interactions=_mind.stats.get("total_interactions", 0),
        memory_count=_memory_store.get_statistics()["total_memories"] if _memory_store else 0,
        needs_state=_mind.needs.get_state() if _mind.needs else {},
    )


# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message and get a response from MindForge.

    This is the main interaction endpoint for conversing with the consciousness engine.
    """
    import time

    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    start_time = time.time()

    try:
        # Process through Mind
        # Note: In a full implementation, this would use the Mind's process method
        response_text = f"I received your message: '{request.message}'. I'm thinking about how to help you."

        # Collect thoughts if requested
        thoughts = None
        if request.include_thoughts:
            thoughts = [
                "Analyzing user intent...",
                "Considering relevant context...",
                "Formulating helpful response...",
            ]

        processing_time = time.time() - start_time

        return ChatResponse(
            response=response_text,
            thoughts=thoughts,
            needs_state=_mind.needs.get_state() if _mind.needs else None,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/think", tags=["Chat"])
async def generate_thought(request: ThoughtRequest):
    """Generate a spontaneous thought."""
    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    try:
        from mindforge.core.thought import ThoughtTrigger

        trigger = ThoughtTrigger(request.trigger) if hasattr(ThoughtTrigger, request.trigger.upper()) else ThoughtTrigger.IDLE

        thought = _mind.thought_generator.generate(
            trigger=trigger,
            context=request.context,
        )

        return {
            "content": thought.content,
            "type": thought.thought_type.value,
            "trigger": thought.trigger.value,
            "confidence": thought.confidence,
        }

    except Exception as e:
        logger.error(f"Thought generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Memory Endpoints
# =============================================================================

@app.post("/memory", tags=["Memory"])
async def store_memory(memory: MemoryItem):
    """Store a new memory."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    try:
        from mindforge.memory import Memory, MemoryType

        mem = Memory(
            content=memory.content,
            memory_type=MemoryType(memory.memory_type),
            importance=memory.importance,
            tags=memory.tags,
            metadata=memory.metadata,
        )

        memory_id = _memory_store.store(mem)

        return {"id": memory_id, "status": "stored"}

    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/recent", tags=["Memory"])
async def get_recent_memories(count: int = 10, memory_type: Optional[str] = None):
    """Get recent memories."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    try:
        from mindforge.memory import MemoryType

        mt = MemoryType(memory_type) if memory_type else None

        memories = _memory_store.get_recent(count=count, memory_type=mt)

        return {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    except Exception as e:
        logger.error(f"Get recent memories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats", tags=["Memory"])
async def get_memory_stats():
    """Get memory statistics."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    return _memory_store.get_statistics()


@app.get("/memory/{memory_id}", tags=["Memory"])
async def get_memory(memory_id: int):
    """Get a specific memory by ID."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    memory = _memory_store.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return memory.to_dict()


@app.post("/memory/search", tags=["Memory"])
async def search_memories(request: MemorySearchRequest):
    """Search memories."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    try:
        from mindforge.memory import MemoryType

        memory_type = MemoryType(request.memory_type) if request.memory_type else None

        results = _memory_store.search(
            query=request.query,
            limit=request.limit,
        )

        # Filter by type if specified
        if memory_type:
            results = [m for m in results if m.memory_type == memory_type]

        return {
            "results": [m.to_dict() for m in results],
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Memory search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{memory_id}", tags=["Memory"])
async def delete_memory(memory_id: int):
    """Delete a memory."""
    if not _memory_store:
        raise HTTPException(status_code=503, detail="Memory store not initialized")

    try:
        _memory_store.delete(memory_id)
        return {"status": "deleted", "id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# KVRM Grounding Endpoints
# =============================================================================

@app.post("/kvrm/ground", response_model=GroundingResponse, tags=["KVRM"])
async def ground_claim(request: GroundingRequest):
    """Ground a claim through KVRM verification."""
    if not _grounding_router:
        raise HTTPException(status_code=503, detail="Grounding router not initialized")

    try:
        result = _grounding_router.ground(request.claim)

        return GroundingResponse(
            original=result.original,
            claim_type=result.claim_type.value,
            grounded=result.grounded,
            confidence=result.confidence,
            status=result.status,
            key_used=result.key_used,
            reason=result.reason,
        )

    except Exception as e:
        logger.error(f"Grounding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kvrm/resolve", tags=["KVRM"])
async def resolve_key(request: KVResolveRequest):
    """Resolve a key to its content."""
    if not _kvrm_resolver:
        raise HTTPException(status_code=503, detail="KVRM resolver not initialized")

    try:
        result = _kvrm_resolver.resolve(request.key)

        if result:
            return {
                "found": True,
                "key": result.key,
                "content": result.content,
                "confidence": result.confidence,
                "source": result.source,
            }
        else:
            return {"found": False, "key": request.key}

    except Exception as e:
        logger.error(f"Resolve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kvrm/search", tags=["KVRM"])
async def search_keys(request: KVSearchRequest):
    """Search for keys matching a query."""
    if not _kvrm_resolver:
        raise HTTPException(status_code=503, detail="KVRM resolver not initialized")

    try:
        results = _kvrm_resolver.search(request.query, limit=request.limit)

        return {
            "results": [
                {
                    "key": r.key,
                    "content": r.content,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in results
            ],
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kvrm/store", tags=["KVRM"])
async def store_fact(request: KVStoreRequest):
    """Store a verified fact."""
    if not _kvrm_resolver:
        raise HTTPException(status_code=503, detail="KVRM resolver not initialized")

    try:
        # Get the facts store
        from mindforge.kvrm.resolver import FactKeyStore

        store = _kvrm_resolver._stores.get(request.store_name)
        if not store:
            raise HTTPException(status_code=404, detail=f"Store '{request.store_name}' not found")

        if isinstance(store, FactKeyStore):
            # Generate key from metadata
            domain = request.metadata.get("domain", "general")
            fact_id = request.metadata.get("fact_id", f"fact_{int(datetime.now().timestamp())}")
            generated_key = f"fact:{domain}:{fact_id}"

            key = store.store(
                key=generated_key,
                content=request.content,
                source=request.metadata.get("source", "api"),
                metadata={
                    "domain": domain,
                    "confidence": request.metadata.get("confidence", 0.8),
                },
            )
            return {"key": key, "status": "stored"}
        else:
            raise HTTPException(status_code=400, detail="Store does not support fact storage")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Store error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Needs Management Endpoints
# =============================================================================

@app.get("/needs", tags=["Needs"])
async def get_needs():
    """Get current needs state."""
    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    return _mind.needs.get_state()


@app.put("/needs", tags=["Needs"])
async def update_needs(update: NeedsUpdate):
    """Update needs weights."""
    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    try:
        # Apply updates
        updates = update.dict(exclude_none=True)
        for need_name, weight in updates.items():
            if hasattr(_mind.needs, f"set_{need_name}_weight"):
                getattr(_mind.needs, f"set_{need_name}_weight")(weight)

        return {"status": "updated", "needs": _mind.needs.get_state()}

    except Exception as e:
        logger.error(f"Needs update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/needs/preset/{preset_name}", tags=["Needs"])
async def apply_needs_preset(preset_name: str):
    """Apply a needs preset configuration."""
    if not _mind:
        raise HTTPException(status_code=503, detail="Mind not initialized")

    try:
        _mind.needs.apply_preset(preset_name)
        return {"status": "applied", "preset": preset_name, "needs": _mind.needs.get_state()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preset application error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Core Values Endpoint (Read-only)
# =============================================================================

@app.get("/values", tags=["Values"])
async def get_core_values():
    """Get immutable core values."""
    from mindforge.core.mind import Mind

    return {
        "core_values": Mind.CORE_VALUES,
        "guardrails": Mind.GUARDRAILS,
        "note": "Core values are immutable and cannot be modified",
    }


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mindforge.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

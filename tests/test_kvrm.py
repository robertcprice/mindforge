"""
Comprehensive tests for MindForge KVRM (Key-Value Response Mapping) module.

Tests cover:
- KeyStore base class and implementations
- KeyResolver routing logic
- GroundingRouter claim classification
- KVRMTool execution
- GroundedThinkingTool
"""

import pytest
from pathlib import Path
from datetime import datetime


class TestKeyStore:
    """Tests for KeyStore base class and implementations."""

    def test_key_store_stats(self):
        """Test KeyStore tracks lookup stats."""
        from mindforge.kvrm.key_store import KeyStore, KeyType

        # Create a mock store that always resolves
        class MockStore(KeyStore):
            def resolve(self, key):
                self._record_lookup(True)
                return None

            def validate_key(self, key):
                return True

            def search(self, query, limit=5, min_confidence=0.5):
                return []

        store = MockStore("test", KeyType.MEMORY, read_only=True)

        # Make some lookups
        for _ in range(5):
            store.resolve("test_key")

        stats = store.get_stats()
        assert stats["lookups"] == 5
        assert stats["hits"] == 5

    def test_key_store_read_only(self):
        """Test read-only store cannot store content."""
        from mindforge.kvrm.key_store import KeyStore, KeyType

        class ReadOnlyStore(KeyStore):
            def resolve(self, key):
                return None

            def validate_key(self, key):
                return True

            def search(self, query, limit=5, min_confidence=0.5):
                return []

        store = ReadOnlyStore("test", KeyType.FACT, read_only=True)
        assert store.read_only is True


class TestMemoryKeyStore:
    """Tests for MemoryKeyStore."""

    def test_memory_key_store_initialization(self, tmp_path):
        """Test MemoryKeyStore initializes with memory store."""
        from mindforge.kvrm.resolver import MemoryKeyStore
        from mindforge.memory import MemoryStore

        db_path = tmp_path / "test_mem.db"
        mem_store = MemoryStore(db_path=db_path)
        key_store = MemoryKeyStore(memory_store=mem_store)

        assert key_store.name == "MindForge Memory"
        assert key_store.read_only is False

    def test_memory_key_pattern_validation(self, tmp_path):
        """Test memory key pattern validation."""
        from mindforge.kvrm.resolver import MemoryKeyStore
        from mindforge.memory import MemoryStore

        db_path = tmp_path / "test_mem.db"
        mem_store = MemoryStore(db_path=db_path)
        key_store = MemoryKeyStore(memory_store=mem_store)

        # Valid patterns
        assert key_store.KEY_PATTERN.match("mem:thought:20241209:abc123")
        assert key_store.KEY_PATTERN.match("mem:reflection:20241208:def456")
        assert key_store.KEY_PATTERN.match("mem:learning:20241207:789abc")

        # Invalid patterns
        assert not key_store.KEY_PATTERN.match("mem:invalid:20241209:abc123")
        assert not key_store.KEY_PATTERN.match("fact:thought:20241209:abc123")
        assert not key_store.KEY_PATTERN.match("mem:thought:invalid:abc123")


class TestFactKeyStore:
    """Tests for FactKeyStore."""

    def test_fact_key_store_initialization(self, tmp_path):
        """Test FactKeyStore initializes correctly."""
        from mindforge.kvrm.resolver import FactKeyStore

        db_path = tmp_path / "facts.db"
        fact_store = FactKeyStore(db_path=db_path)

        assert fact_store.name == "Verified Facts"
        assert fact_store.read_only is False

    def test_fact_store_add_and_resolve(self, tmp_path):
        """Test adding and resolving facts."""
        from mindforge.kvrm.resolver import FactKeyStore

        db_path = tmp_path / "facts.db"
        fact_store = FactKeyStore(db_path=db_path)

        # Add a fact using store() method
        key = fact_store.store(
            key="fact:science:earth_round",
            content="The Earth is roughly spherical in shape.",
            source="NASA",
            metadata={"domain": "science", "confidence": 1.0},
        )

        assert key == "fact:science:earth_round"

        # Resolve the fact
        resolved = fact_store.resolve(key)
        assert resolved is not None
        assert "Earth" in resolved.content
        assert resolved.confidence == 1.0

    def test_fact_store_search(self, tmp_path):
        """Test searching facts."""
        from mindforge.kvrm.resolver import FactKeyStore

        db_path = tmp_path / "facts.db"
        fact_store = FactKeyStore(db_path=db_path)

        # Add multiple facts using store() method
        fact_store.store(
            key="fact:science:water_h2o",
            content="Water is composed of H2O molecules.",
            metadata={"domain": "science"}
        )
        fact_store.store(
            key="fact:science:water_freezes",
            content="Water freezes at 0 degrees Celsius.",
            metadata={"domain": "science"}
        )
        fact_store.store(
            key="fact:history:moon_landing",
            content="Apollo 11 landed on the moon in 1969.",
            metadata={"domain": "history"}
        )

        # Search for water
        results = fact_store.search("water")
        assert len(results) >= 2


class TestKeyResolver:
    """Tests for KeyResolver routing."""

    def test_resolver_register_store(self, tmp_path):
        """Test registering stores with resolver."""
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore

        resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=tmp_path / "facts.db")

        resolver.register_store("facts", fact_store)

        assert "facts" in resolver.stores

    def test_resolver_resolve_with_prefix(self, tmp_path):
        """Test resolver routes to correct store by prefix."""
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore

        resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=tmp_path / "facts.db")
        resolver.register_store("facts", fact_store)

        # Add a fact using store() method
        fact_store.store(
            key="fact:test:example",
            content="Test content",
            metadata={"domain": "test"}
        )

        # Resolve via resolver
        resolved = resolver.resolve("fact:test:example")
        assert resolved is not None
        assert "Test content" in resolved.content


class TestGroundingResult:
    """Tests for GroundingResult dataclass."""

    def test_grounding_result_creation(self):
        """Test creating GroundingResult."""
        from mindforge.kvrm.grounding import GroundingResult, ClaimType

        result = GroundingResult(
            original="The sky is blue",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.95,
        )

        assert result.original == "The sky is blue"
        assert result.claim_type == ClaimType.FACTUAL
        assert result.grounded is True
        assert result.confidence == 0.95

    def test_grounding_result_is_verified(self):
        """Test is_verified property."""
        from mindforge.kvrm.grounding import GroundingResult, ClaimType

        # Verified (grounded + high confidence)
        verified = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.95,
        )
        assert verified.is_verified is True

        # Grounded but not verified (low confidence)
        grounded = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.7,
        )
        assert grounded.is_verified is False

        # Not grounded
        ungrounded = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=False,
            confidence=0.0,
        )
        assert ungrounded.is_verified is False

    def test_grounding_result_status(self):
        """Test status property."""
        from mindforge.kvrm.grounding import GroundingResult, ClaimType

        # Verified
        verified = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.95,
        )
        assert verified.status == "VERIFIED"

        # Grounded
        grounded = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.7,
        )
        assert grounded.status == "GROUNDED"

        # Not applicable (opinion)
        opinion = GroundingResult(
            original="test",
            claim_type=ClaimType.OPINION,
            grounded=False,
            confidence=0.0,
        )
        assert opinion.status == "NOT_APPLICABLE"

        # Unverified
        unverified = GroundingResult(
            original="test",
            claim_type=ClaimType.FACTUAL,
            grounded=False,
            confidence=0.0,
        )
        assert unverified.status == "UNVERIFIED"

    def test_grounding_result_to_dict(self):
        """Test to_dict serialization."""
        from mindforge.kvrm.grounding import GroundingResult, ClaimType

        result = GroundingResult(
            original="The sky is blue",
            claim_type=ClaimType.FACTUAL,
            grounded=True,
            confidence=0.95,
        )

        d = result.to_dict()
        assert d["original"] == "The sky is blue"
        assert d["claim_type"] == "factual"
        assert d["grounded"] is True
        assert d["confidence"] == 0.95


class TestClaimType:
    """Tests for ClaimType enum."""

    def test_claim_types_exist(self):
        """Test all expected claim types exist."""
        from mindforge.kvrm.grounding import ClaimType

        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.MEMORY.value == "memory"
        assert ClaimType.OPINION.value == "opinion"
        assert ClaimType.QUESTION.value == "question"
        assert ClaimType.CREATIVE.value == "creative"
        assert ClaimType.ACTION.value == "action"
        assert ClaimType.UNKNOWN.value == "unknown"


class TestGroundingRouter:
    """Tests for GroundingRouter."""

    def test_grounding_router_initialization(self):
        """Test GroundingRouter initializes correctly."""
        from mindforge.kvrm.grounding import GroundingRouter
        from mindforge.kvrm.resolver import KeyResolver

        resolver = KeyResolver()
        router = GroundingRouter(key_resolver=resolver)

        assert router.key_resolver is resolver

    def test_grounding_router_classify_factual(self):
        """Test claim classification for factual claims."""
        from mindforge.kvrm.grounding import GroundingRouter, ClaimType
        from mindforge.kvrm.resolver import KeyResolver

        resolver = KeyResolver()
        router = GroundingRouter(key_resolver=resolver)

        # Factual claim with "is" indicator (matches _factual_indicators)
        result = router._classify_claim("The Earth is a planet.")
        assert result == ClaimType.FACTUAL

    def test_grounding_router_classify_opinion(self):
        """Test claim classification for opinions."""
        from mindforge.kvrm.grounding import GroundingRouter, ClaimType
        from mindforge.kvrm.resolver import KeyResolver

        resolver = KeyResolver()
        router = GroundingRouter(key_resolver=resolver)

        # Opinion indicators
        result = router._classify_claim("I think this is a good approach.")
        assert result == ClaimType.OPINION

    def test_grounding_router_classify_question(self):
        """Test claim classification for questions."""
        from mindforge.kvrm.grounding import GroundingRouter, ClaimType
        from mindforge.kvrm.resolver import KeyResolver

        resolver = KeyResolver()
        router = GroundingRouter(key_resolver=resolver)

        # Questions
        result = router._classify_claim("What is the capital of France?")
        assert result == ClaimType.QUESTION


class TestKVRMTool:
    """Tests for KVRMTool."""

    def test_kvrm_tool_initialization(self):
        """Test KVRMTool initializes correctly."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        assert tool.name == "kvrm"
        assert "resolve" in tool.description.lower() or "ground" in tool.description.lower()

    def test_kvrm_tool_unknown_operation(self):
        """Test KVRMTool returns error for unknown operation."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        result = tool.execute(operation="invalid_operation")
        assert result.status == ToolStatus.ERROR
        assert "Unknown operation" in result.error

    def test_kvrm_tool_resolve_missing_key(self):
        """Test KVRMTool resolve operation with missing key."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        result = tool.execute(operation="resolve", key="")
        assert result.status == ToolStatus.ERROR
        assert "Key required" in result.error

    def test_kvrm_tool_search_missing_query(self):
        """Test KVRMTool search operation with missing query."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        result = tool.execute(operation="search", query="")
        assert result.status == ToolStatus.ERROR
        assert "Query required" in result.error

    def test_kvrm_tool_ground_missing_claim(self):
        """Test KVRMTool ground operation with missing claim."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        result = tool.execute(operation="ground", claim="")
        assert result.status == ToolStatus.ERROR
        assert "Claim required" in result.error

    def test_kvrm_tool_store_missing_content(self):
        """Test KVRMTool store operation with missing content."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        tool = KVRMTool(key_resolver=resolver)

        result = tool.execute(operation="store", content="")
        assert result.status == ToolStatus.ERROR
        assert "Content required" in result.error

    def test_kvrm_tool_resolve_success(self, tmp_path):
        """Test successful resolve operation."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=tmp_path / "facts.db")
        resolver.register_store("facts", fact_store)

        # Add a fact using store() method
        fact_store.store(
            key="fact:test:hello",
            content="Hello, World!",
            metadata={"domain": "test"}
        )

        tool = KVRMTool(key_resolver=resolver)
        result = tool.execute(operation="resolve", key="fact:test:hello")

        assert result.status == ToolStatus.SUCCESS
        assert "Hello, World!" in result.output

    def test_kvrm_tool_search_success(self, tmp_path):
        """Test successful search operation."""
        from mindforge.kvrm.tool import KVRMTool
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore
        from mindforge.tools import ToolStatus

        resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=tmp_path / "facts.db")
        resolver.register_store("facts", fact_store)

        # Add facts using store() method
        fact_store.store(
            key="fact:science:water1",
            content="Water is H2O",
            metadata={"domain": "science"}
        )
        fact_store.store(
            key="fact:science:water2",
            content="Water freezes at 0C",
            metadata={"domain": "science"}
        )

        tool = KVRMTool(key_resolver=resolver)
        result = tool.execute(operation="search", query="water")

        assert result.status == ToolStatus.SUCCESS


class TestKVRMIntegration:
    """Integration tests for the full KVRM system."""

    def test_full_grounding_workflow(self, tmp_path):
        """Test complete grounding workflow."""
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore
        from mindforge.kvrm.grounding import GroundingRouter
        from mindforge.kvrm.tool import KVRMTool

        # Set up resolver with fact store
        resolver = KeyResolver()
        fact_store = FactKeyStore(db_path=tmp_path / "facts.db")
        resolver.register_store("facts", fact_store)

        # Add verified facts using store() method
        fact_store.store(
            key="fact:geography:paris_capital",
            content="Paris is the capital of France.",
            metadata={"domain": "geography", "confidence": 1.0},
        )

        # Create grounding router
        router = GroundingRouter(key_resolver=resolver)

        # Create tool
        tool = KVRMTool(key_resolver=resolver, grounding_router=router)

        # Test resolve
        resolve_result = tool.execute(
            operation="resolve",
            key="fact:geography:paris_capital"
        )
        assert "Paris" in resolve_result.output

        # Test search
        search_result = tool.execute(
            operation="search",
            query="capital France"
        )
        assert search_result.status.value == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

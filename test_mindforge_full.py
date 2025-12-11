#!/usr/bin/env python3
"""
MindForge Full System Test

Tests all MindForge functionality:
1. MLX Inference Backend (with LoRA adapter)
2. Mind/Consciousness Core
3. Memory System
4. KVRM Grounding
5. Training Integration
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def test_mlx_inference():
    """Test 1: MLX Inference Backend with LoRA adapter."""
    print_section("TEST 1: MLX INFERENCE BACKEND")

    try:
        from mindforge.inference.mlx_backend import MLXBackend, is_mlx_available
        from mindforge.inference.base import InferenceConfig

        if not is_mlx_available():
            print("MLX not available on this system")
            return False

        print("MLX is available")

        # Test configuration
        config = InferenceConfig(
            model_path="mlx-community/Qwen2.5-7B-Instruct-8bit",
            adapter_path="./models/mindforge_lora_mlx",
            max_tokens=100,
            temperature=0.7,
        )

        print(f"Model: {config.model_path}")
        print(f"Adapter: {config.adapter_path}")

        # Check adapter exists
        adapter_path = Path(config.adapter_path)
        if adapter_path.exists():
            print(f"Adapter found at {adapter_path}")
            adapter_files = list(adapter_path.glob("*"))
            print(f"  Files: {[f.name for f in adapter_files]}")
        else:
            print(f"No adapter at {adapter_path}, using base model only")
            config.adapter_path = ""

        # Create backend
        backend = MLXBackend(config)

        print("\nLoading model...")
        backend.load()
        print("Model loaded successfully!")

        # Get memory usage
        mem_info = backend.get_memory_usage()
        if mem_info:
            print(f"Memory usage: {mem_info.get('active_memory_gb', 0):.2f} GB")

        # Test generation
        print("\nTesting generation...")
        test_prompt = "What are your core values?"

        messages = [
            {"role": "system", "content": "You are MindForge, a benevolent AI assistant."},
            {"role": "user", "content": test_prompt}
        ]
        formatted = backend.format_chat(messages)

        result = backend.generate(
            prompt=formatted,
            max_tokens=100,
            temperature=0.7,
            stop_sequences=["<|im_end|>"]
        )

        print(f"Prompt: {test_prompt}")
        print(f"Response: {result.text[:200]}...")
        print(f"Tokens: {result.tokens_generated}, Time: {result.generation_time:.2f}s")
        print(f"Speed: {result.tokens_per_second:.1f} tok/s")

        # Test streaming
        print("\nTesting streaming generation...")
        stream_prompt = "Explain briefly why honesty is important."
        messages = [
            {"role": "system", "content": "You are MindForge, a benevolent AI."},
            {"role": "user", "content": stream_prompt}
        ]
        formatted = backend.format_chat(messages)

        print(f"Streaming: ", end="", flush=True)
        tokens = []
        for token in backend.generate_stream(
            prompt=formatted,
            max_tokens=50,
            stop_sequences=["<|im_end|>"]
        ):
            print(token, end="", flush=True)
            tokens.append(token)
        print(f"\n({len(tokens)} tokens streamed)")

        # Unload
        backend.unload()
        print("\nModel unloaded successfully")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_manager():
    """Test 2: Model Manager (unified inference interface)."""
    print_section("TEST 2: MODEL MANAGER")

    try:
        from mindforge.inference.model_manager import ModelManager, MINDFORGE_SYSTEM_PROMPT
        from mindforge.inference.base import InferenceConfig

        config = InferenceConfig(
            model_path="mlx-community/Qwen2.5-7B-Instruct-8bit",
            max_tokens=100,
            temperature=0.7,
        )

        # Check for adapter
        adapter_path = Path("./models/mindforge_lora_mlx")
        if adapter_path.exists():
            config.adapter_path = str(adapter_path)
            print(f"Using adapter: {adapter_path}")

        manager = ModelManager(config=config)

        print("Loading model via ModelManager...")
        manager.load()
        print(f"Backend: {manager.backend_type.value if manager.backend_type else 'unknown'}")
        print(f"Loaded: {manager.is_loaded}")

        # Test chat
        print("\nTesting chat interface...")
        messages = [
            {"role": "system", "content": MINDFORGE_SYSTEM_PROMPT},
            {"role": "user", "content": "Hello, who are you?"}
        ]

        response = manager.chat(messages, max_tokens=100)
        print(f"User: Hello, who are you?")
        print(f"MindForge: {response[:200]}...")

        # Get inference function
        print("\nTesting inference function...")
        inference_fn = manager.get_inference_function()
        result = inference_fn("What is 2+2? Answer briefly.")
        print(f"Inference fn result: {result[:100]}...")

        manager.unload()
        print("\nModel Manager test complete")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system():
    """Test 3: Memory System."""
    print_section("TEST 3: MEMORY SYSTEM")

    try:
        from mindforge.memory.store import MemoryStore, Memory, MemoryType

        # Use temp file for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        print(f"Creating memory store at: {db_path}")
        store = MemoryStore(db_path)

        # Store memories
        print("\nStoring test memories...")

        mem1 = Memory(
            content="User asked about machine learning basics",
            memory_type=MemoryType.INTERACTION,
            importance=0.8,
            tags=["ml", "tutorial"],
            source="test"
        )
        id1 = store.store(mem1)
        print(f"  Stored interaction memory (id={id1})")

        mem2 = Memory(
            content="I should explain concepts more simply for beginners",
            memory_type=MemoryType.REFLECTION,
            importance=0.7,
            tags=["learning", "improvement"],
            source="test"
        )
        id2 = store.store(mem2)
        print(f"  Stored reflection memory (id={id2})")

        mem3 = Memory(
            content="Python is a programming language created by Guido van Rossum",
            memory_type=MemoryType.FACT,
            importance=0.9,
            tags=["python", "fact"],
            source="test"
        )
        id3 = store.store(mem3)
        print(f"  Stored fact memory (id={id3})")

        # Retrieve
        print("\nRetrieving memory...")
        retrieved = store.get(id1)
        if retrieved:
            print(f"  Retrieved: {retrieved.content}")
            print(f"  Type: {retrieved.memory_type.value}")
            print(f"  Tags: {retrieved.tags}")

        # Search
        print("\nSearching memories...")
        results = store.search("Python programming")
        print(f"  Found {len(results)} results for 'Python programming'")
        for r in results:
            print(f"    - {r.content[:50]}... (type={r.memory_type.value})")

        # Get by type
        print("\nGetting memories by type...")
        reflections = store.get_by_type(MemoryType.REFLECTION)
        print(f"  Found {len(reflections)} reflection memories")

        # Link memories
        print("\nLinking related memories...")
        store.link_memories(id1, id2)
        linked = store.get(id1)
        print(f"  Memory {id1} now linked to: {linked.related_to}")

        # Statistics
        print("\nMemory statistics:")
        stats = store.get_statistics()
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  By type: {stats['by_type']}")
        print(f"  Avg importance: {stats['average_importance']:.2f}")

        # Cleanup test db
        db_path.unlink()
        print("\nMemory system test complete")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kvrm_grounding():
    """Test 4: KVRM Grounding System."""
    print_section("TEST 4: KVRM GROUNDING SYSTEM")

    try:
        from mindforge.kvrm.grounding import GroundingRouter, ClaimType
        from mindforge.kvrm.resolver import KeyResolver

        print("Creating grounding router...")
        resolver = KeyResolver()
        router = GroundingRouter(resolver)

        # Test claim classification
        test_claims = [
            ("Python was created by Guido van Rossum", ClaimType.FACTUAL),
            ("What is the weather today?", ClaimType.QUESTION),
            ("I think this is a good solution", ClaimType.OPINION),
            ("I remember discussing this yesterday", ClaimType.MEMORY),
            ("Let me help you with that", ClaimType.ACTION),
        ]

        print("\nTesting claim classification:")
        for claim, expected_type in test_claims:
            result = router.ground(claim)
            match = "" if result.claim_type == expected_type else f" (expected {expected_type.value})"
            print(f"  [{result.claim_type.value:10}] {claim[:40]}...{match}")

        # Test grounding
        print("\nTesting claim grounding:")
        claims_to_ground = [
            "Python is a programming language",
            "I believe AI can help people",
            "What time is it?",
        ]

        for claim in claims_to_ground:
            result = router.ground(claim)
            status = result.status
            print(f"  [{status:15}] {claim[:40]}...")
            if result.reason:
                print(f"                  Reason: {result.reason}")

        # Test thought grounding
        print("\nTesting thought grounding:")
        thought = "Python is a great language. I think it's easy to learn. What makes it popular?"
        grounded_thought, results = router.ground_thought(thought)

        print(f"  Original: {thought}")
        print(f"  Grounded: {grounded_thought[:100]}...")
        print(f"  Results: {len(results)} claims analyzed")
        for r in results:
            print(f"    - [{r.status}] {r.original[:30]}...")

        print("\nKVRM grounding test complete")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mind_core():
    """Test 5: Mind/Consciousness Core."""
    print_section("TEST 5: MIND/CONSCIOUSNESS CORE")

    try:
        from mindforge.core.mind import Mind, MindState, create_mind
        from mindforge.core.needs import NeedType

        # Create a simple mock inference function
        def mock_inference(prompt: str) -> str:
            return "I understand your request. I'm here to help with thoughtful guidance."

        print("Creating Mind instance...")
        mind = create_mind(needs_preset="balanced", inference_fn=mock_inference)

        print(f"Mind created: {mind}")
        print(f"State: {mind.state.value}")
        print(f"Core values: {list(mind.CORE_VALUES.keys())}")

        # Check needs
        print("\nNeeds state:")
        needs = mind.get_needs_state()
        for need, value in needs.items():
            print(f"  {need}: {value}")

        # Process input
        print("\nProcessing user input...")

        async def run_interaction():
            response = await mind.process_input(
                "I'm feeling stressed about work",
                context={}
            )
            return response

        response = asyncio.run(run_interaction())
        print(f"User: I'm feeling stressed about work")
        print(f"Mind: {response[:150]}...")

        # Check statistics
        print("\nMind statistics:")
        stats = mind.get_statistics()
        print(f"  Interactions: {stats['total_interactions']}")
        print(f"  Thoughts generated: {stats['thoughts_generated']}")
        print(f"  Current state: {stats['current_state']}")

        # Get recent thoughts
        print("\nRecent thoughts:")
        thoughts = mind.get_thought_history(5)
        for t in thoughts[:3]:
            print(f"  [{t.thought_type.value}] {t.content[:50]}...")

        # Test needs presets
        print("\nTesting needs presets:")
        for preset in ["balanced", "learning", "production", "creative"]:
            mind.set_needs_preset(preset)
            state = mind.get_needs_state()
            sus = state.get('sustainability', {})
            cur = state.get('curiosity', {})
            sus_val = sus.get('level', 0) if isinstance(sus, dict) else sus
            cur_val = cur.get('level', 0) if isinstance(cur, dict) else cur
            print(f"  {preset}: sustainability={sus_val:.2f}, curiosity={cur_val:.2f}")

        print("\nMind core test complete")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_integration():
    """Test 6: Training Pipeline Integration."""
    print_section("TEST 6: TRAINING PIPELINE")

    try:
        from mindforge.training.data import TrainingExample, ExampleType
        from mindforge.training.lora import LoRAConfig

        print("Testing LoRA configuration...")

        # Test LoRA configs
        configs = {
            "consciousness": LoRAConfig.for_consciousness(),
            "grounding": LoRAConfig.for_grounding(),
            "combined": LoRAConfig.for_combined(),
        }

        for name, config in configs.items():
            print(f"\n  {name} config:")
            print(f"    rank={config.rank}, alpha={config.alpha}")
            print(f"    targets: {config.target_modules[:3]}...")

        # Test training example creation
        print("\nTesting training example creation...")

        example = TrainingExample(
            prompt="What are your core values?",
            completion="My core values are benevolence, honesty, humility, and growth for service.",
            example_type=ExampleType.THOUGHT_GENERATION,
            quality_score=0.9,
            metadata={"source": "test"}
        )

        print(f"  Created example: {example.example_type.value}")
        print(f"  Quality score: {example.quality_score}")

        # Check training data exists
        train_path = Path("./data/training/train.jsonl")
        valid_path = Path("./data/training/valid.jsonl")

        print("\nChecking training data:")
        if train_path.exists():
            with open(train_path) as f:
                train_count = sum(1 for _ in f)
            print(f"  train.jsonl: {train_count} examples")
        else:
            print(f"  train.jsonl: NOT FOUND")

        if valid_path.exists():
            with open(valid_path) as f:
                valid_count = sum(1 for _ in f)
            print(f"  valid.jsonl: {valid_count} examples")
        else:
            print(f"  valid.jsonl: NOT FOUND")

        # Check adapters
        print("\nChecking trained adapters:")
        adapter_dirs = [
            Path("./models/mindforge_lora_mlx"),
            Path("./models/mindforge_trained"),
        ]

        for adapter_dir in adapter_dirs:
            if adapter_dir.exists():
                files = list(adapter_dir.glob("*"))
                print(f"  {adapter_dir.name}:")
                for f in files:
                    size = f.stat().st_size / 1024 / 1024
                    print(f"    - {f.name} ({size:.1f} MB)")
            else:
                print(f"  {adapter_dir.name}: NOT FOUND")

        print("\nTraining integration test complete")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_consciousness_loop():
    """Test 7: Full Consciousness Loop."""
    print_section("TEST 7: FULL CONSCIOUSNESS LOOP")

    try:
        from mindforge.core.mind import create_mind
        from mindforge.inference.model_manager import ModelManager, MINDFORGE_SYSTEM_PROMPT
        from mindforge.inference.base import InferenceConfig
        from mindforge.memory.store import MemoryStore, Memory, MemoryType
        import tempfile

        print("Setting up full consciousness loop...")

        # 1. Setup inference
        print("\n1. Setting up inference...")
        config = InferenceConfig(
            model_path="mlx-community/Qwen2.5-7B-Instruct-8bit",
            max_tokens=150,
            temperature=0.7,
        )

        adapter_path = Path("./models/mindforge_lora_mlx")
        if adapter_path.exists():
            config.adapter_path = str(adapter_path)
            print(f"   Using MindForge adapter")

        manager = ModelManager(config=config)
        manager.load()
        print(f"   Model loaded: {manager.backend_type.value}")

        # 2. Create inference function for Mind
        def mindforge_inference(prompt: str) -> str:
            messages = [
                {"role": "system", "content": MINDFORGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            return manager.chat(messages, max_tokens=150)

        # 3. Setup memory
        print("\n2. Setting up memory system...")
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            memory_path = Path(f.name)
        memory_store = MemoryStore(memory_path)
        print(f"   Memory store ready")

        # 4. Create Mind
        print("\n3. Creating Mind with all components...")
        mind = create_mind(
            needs_preset="balanced",
            inference_fn=mindforge_inference
        )
        mind.set_memory_store(memory_store)
        print(f"   Mind created and connected")

        # 5. Run consciousness loop
        print("\n4. Running consciousness interactions...")

        test_inputs = [
            "Hello! Tell me about your core values.",
            "How do you decide what's the right thing to do?",
            "What limitations do you have?",
        ]

        async def run_loop():
            for user_input in test_inputs:
                print(f"\n   User: {user_input}")
                response = await mind.process_input(user_input)
                print(f"   MindForge: {response[:150]}...")

                # Store interaction in memory
                memory_store.store(Memory(
                    content=f"User: {user_input}\nResponse: {response[:200]}",
                    memory_type=MemoryType.INTERACTION,
                    importance=0.7,
                    tags=["consciousness_test"]
                ))

        asyncio.run(run_loop())

        # 6. Check results
        print("\n5. Checking consciousness loop results...")
        stats = mind.get_statistics()
        print(f"   Total interactions: {stats['total_interactions']}")
        print(f"   Thoughts generated: {stats['thoughts_generated']}")

        mem_stats = memory_store.get_statistics()
        print(f"   Memories stored: {mem_stats['total_memories']}")

        # 7. Test spontaneous thought
        print("\n6. Testing spontaneous thought generation...")

        async def generate_spontaneous():
            thought = await mind.generate_spontaneous_thought(
                context={"time_since_last_interaction": 60}
            )
            return thought

        thought = asyncio.run(generate_spontaneous())
        if thought:
            print(f"   Generated: [{thought.thought_type.value}] {thought.content[:100]}...")
        else:
            print("   No spontaneous thought generated (normal)")

        # Cleanup
        manager.unload()
        memory_path.unlink()

        print("\n" + "=" * 60)
        print(" FULL CONSCIOUSNESS LOOP TEST COMPLETE")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" MINDFORGE FULL SYSTEM TEST SUITE")
    print("=" * 60)
    print(f" Started: {datetime.now().isoformat()}")
    print("=" * 60)

    tests = [
        ("MLX Inference Backend", test_mlx_inference),
        ("Model Manager", test_model_manager),
        ("Memory System", test_memory_system),
        ("KVRM Grounding", test_kvrm_grounding),
        ("Mind Core", test_mind_core),
        ("Training Integration", test_training_integration),
        ("Full Consciousness Loop", test_full_consciousness_loop),
    ]

    results = {}

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASSED" if passed else "FAILED"
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            results[name] = "INTERRUPTED"
            break
        except Exception as e:
            results[name] = f"ERROR: {e}"

    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)

    for name, result in results.items():
        status = "" if result == "PASSED" else ""
        print(f"  {status} {name}: {result}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

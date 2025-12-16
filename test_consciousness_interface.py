#!/usr/bin/env python3
"""
Test script for the Consciousness Interface.

Verifies that all components work correctly without requiring interactive input.
"""

import sys
import time

# Test imports
print("=" * 60)
print("CONSCIOUSNESS INTERFACE TEST SUITE")
print("=" * 60)
print()

# Test 1: Import all components
print("[1] Testing imports...")
try:
    from consciousness_interface import (
        AGENTS, AgentRole, Agent,
        call_ollama, ChatSession,
        OLLAMA_URL, MODEL
    )
    from rich.console import Console
    from rich.panel import Panel
    print("    ✓ All imports successful")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Agent definitions
print("\n[2] Testing agent definitions...")
assert len(AGENTS) >= 6, f"Expected 6+ agents, got {len(AGENTS)}"
print(f"    ✓ Loaded {len(AGENTS)} agents")

for name, agent in AGENTS.items():
    assert isinstance(agent, Agent), f"{name} is not an Agent"
    assert agent.name, f"{name} has no name"
    assert isinstance(agent.role, AgentRole), f"{name} has invalid role"
    assert agent.personality, f"{name} has no personality"
    print(f"    ✓ Agent '{name}' ({agent.role.value})")

# Test 3: System prompts
print("\n[3] Testing system prompts...")
for name, agent in AGENTS.items():
    prompt = agent.get_system_prompt()
    assert agent.name in prompt, f"Agent name not in system prompt"
    assert agent.role.value in prompt, f"Agent role not in system prompt"
    assert len(prompt) > 50, f"System prompt too short"
print(f"    ✓ All {len(AGENTS)} agents have valid system prompts")

# Test 4: Ollama API connection (non-streaming)
print("\n[4] Testing Ollama API (non-streaming)...")
start = time.time()
try:
    response, elapsed = call_ollama(
        prompt="Say 'test successful' in exactly two words.",
        system_prompt="You are a test assistant. Be extremely brief.",
        temperature=0.1
    )
    assert response and len(response) > 0, "Empty response"
    assert elapsed > 0, "Invalid elapsed time"
    print(f"    ✓ API responded in {elapsed:.2f}s")
    print(f"    ✓ Response: {response[:100]}...")
except Exception as e:
    print(f"    ✗ API call failed: {e}")
    sys.exit(1)

# Test 5: Ollama API with streaming
print("\n[5] Testing Ollama API (streaming)...")
tokens_received = []
def stream_callback(token):
    tokens_received.append(token)

try:
    response, elapsed = call_ollama(
        prompt="Count: 1, 2, 3",
        system_prompt="Complete the count to 5. Be very brief.",
        temperature=0.1,
        callback=stream_callback
    )
    assert len(tokens_received) > 0, "No tokens streamed"
    print(f"    ✓ Streamed {len(tokens_received)} tokens in {elapsed:.2f}s")
except Exception as e:
    print(f"    ✗ Streaming failed: {e}")
    sys.exit(1)

# Test 6: ChatSession
print("\n[6] Testing ChatSession...")
try:
    conch = AGENTS["conch"]
    session = ChatSession(conch)

    # Add some messages
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there!")

    assert len(session.history) == 2, "History not recording"

    context = session.get_context()
    assert "Hello" in context, "Context missing user message"
    assert "Hi there!" in context, "Context missing assistant message"

    print(f"    ✓ ChatSession initialized with {conch.name}")
    print(f"    ✓ Message history: {len(session.history)} messages")
    print(f"    ✓ Context generation working")
except Exception as e:
    print(f"    ✗ ChatSession test failed: {e}")
    sys.exit(1)

# Test 7: Agent interaction simulation
print("\n[7] Testing agent interaction...")
try:
    sophia = AGENTS["sophia"]

    # Get a short response from sophia
    response, elapsed = call_ollama(
        prompt="What is consciousness? Answer in one sentence.",
        system_prompt=sophia.get_system_prompt(),
        temperature=0.7
    )

    assert response and len(response) > 10, "Response too short"
    print(f"    ✓ {sophia.name} responded in {elapsed:.2f}s")
    print(f"    ✓ Response length: {len(response)} chars")
except Exception as e:
    print(f"    ✗ Agent interaction failed: {e}")
    sys.exit(1)

# Test 8: Rich console output
print("\n[8] Testing Rich console output...")
try:
    console = Console()
    # Test that we can create panels and tables
    panel = Panel("Test content", title="Test Panel")
    from rich.table import Table
    table = Table()
    table.add_column("Test")
    table.add_row("Value")
    print("    ✓ Rich Panel creation works")
    print("    ✓ Rich Table creation works")
except Exception as e:
    print(f"    ✗ Rich output failed: {e}")
    sys.exit(1)

# Summary
print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print()
print("The consciousness interface is ready to use:")
print("  ./venv/bin/python consciousness_interface.py")
print()
print("Available modes:")
print("  /chat       - Chat with consciousness agents")
print("  /agents     - Watch dual-agent conversations")
print("  /experiment - Run multi-agent experiments")
print("  /dashboard  - View system status")
print("  /reflect    - Agent self-reflection")
print()

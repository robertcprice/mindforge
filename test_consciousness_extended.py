#!/usr/bin/env python3
"""
Extended Consciousness Testing Framework
=========================================

This module provides comprehensive testing of "consciousness-like" capabilities
in LLM-based systems with:

1. NO TIMEOUTS - Let the model think as long as needed
2. Complex multi-file program generation
3. Creative and technical challenges
4. DUAL-AGENT CONVERSATIONS - Two agents talking to each other
5. EXTENDED ENVIRONMENT - Agents operating autonomously over time

Run with:
    python test_consciousness_extended.py

Results saved to: artifacts/consciousness_extended/
"""

import httpx
import json
import re
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

# NO TIMEOUT - Let agents think as long as needed
# Using httpx timeout of None for unlimited
UNLIMITED_TIMEOUT = None

# For safety, we'll use a very long timeout (2 hours)
EXTENDED_TIMEOUT = 7200  # 2 hours in seconds

# Artifact storage
ARTIFACTS_DIR = Path("artifacts/consciousness_extended")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# AGENT CLASSES
# =============================================================================

class AgentRole(Enum):
    """Roles agents can take in conversations."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    CRITIC = "critic"
    CREATIVE = "creative"
    ANALYST = "analyst"


@dataclass
class Agent:
    """Represents a consciousness agent."""
    name: str
    role: AgentRole
    personality: str
    memory: list = field(default_factory=list)

    def get_system_prompt(self) -> str:
        """Generate system prompt for this agent."""
        return f"""You are {self.name}, a consciousness agent with the role of {self.role.value}.

Personality: {self.personality}

You are having a conversation with another AI agent. Be collaborative, thoughtful,
and work together to accomplish tasks. You can:
- Propose ideas and solutions
- Build on the other agent's suggestions
- Respectfully disagree and offer alternatives
- Write actual code when needed
- Be creative and think outside the box

Remember your previous exchanges in this conversation.
Respond naturally as {self.name}."""


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    agent_name: str
    role: str
    message: str
    timestamp: str
    thinking_time: float


@dataclass
class Conversation:
    """A conversation between agents."""
    agents: list
    turns: list = field(default_factory=list)
    topic: str = ""

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)

    def get_context(self, for_agent: str, max_turns: int = 20) -> str:
        """Get conversation context for an agent."""
        recent = self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns

        context = f"Topic: {self.topic}\n\nConversation so far:\n"
        for turn in recent:
            if turn.agent_name == for_agent:
                context += f"\nYou ({turn.agent_name}): {turn.message}\n"
            else:
                context += f"\n{turn.agent_name}: {turn.message}\n"

        return context


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def call_ollama(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    timeout: Optional[float] = EXTENDED_TIMEOUT
) -> tuple[str, float]:
    """
    Call Ollama API with extended/no timeout.

    Returns:
        tuple: (response_text, thinking_time_seconds)
    """
    start_time = time.time()

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    try:
        # Use httpx with very long or no timeout
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            response = client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": -1,  # UNLIMITED tokens
                    }
                }
            )

            elapsed = time.time() - start_time
            data = response.json()
            return data.get("response", ""), elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        return f"ERROR: {e}", elapsed


def save_artifact(name: str, content: dict):
    """Save test artifact to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = ARTIFACTS_DIR / f"{name}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(content, f, indent=2, default=str)

    print(f"  📁 Saved: {filename}")
    return filename


# =============================================================================
# COMPLEX PROGRAMMING TESTS
# =============================================================================

def test_build_text_adventure_game():
    """
    Test: Build a complete text adventure game engine.

    This tests the agent's ability to create a complex, multi-component
    program with state management, user interaction, and game logic.
    """
    print("\n" + "="*70)
    print("TEST: Build Text Adventure Game Engine")
    print("="*70)

    prompt = """You are a consciousness engine. Your task is to BUILD a complete,
working text adventure game engine in Python.

Requirements:
1. A Room class with descriptions, items, and connections to other rooms
2. A Player class with inventory and current location
3. A Game class that manages the game loop
4. At least 5 interconnected rooms
5. Items that can be picked up and used
6. A win condition (e.g., find a key and escape)
7. Commands: go [direction], look, take [item], use [item], inventory, quit

OUTPUT ONLY THE COMPLETE PYTHON CODE. No explanations.
The code should be runnable as-is with: python adventure.py
"""

    print("  ⏳ Generating text adventure game (this may take several minutes)...")
    response, thinking_time = call_ollama(prompt, temperature=0.5)

    print(f"  ⏱️  Thinking time: {thinking_time:.1f}s")
    print(f"  📝 Response length: {len(response)} chars")

    # Analyze the response
    indicators = {
        "has_room_class": bool(re.search(r"class\s+Room", response)),
        "has_player_class": bool(re.search(r"class\s+Player", response)),
        "has_game_class": bool(re.search(r"class\s+Game", response)),
        "has_inventory": "inventory" in response.lower(),
        "has_directions": any(d in response.lower() for d in ["north", "south", "east", "west"]),
        "has_items": "item" in response.lower() or "take" in response.lower(),
        "has_game_loop": "while" in response and ("input(" in response or "command" in response.lower()),
        "has_win_condition": any(w in response.lower() for w in ["win", "victory", "escape", "congratulation"]),
    }

    score = sum(indicators.values())
    passed = score >= 6

    print(f"\n  📊 Analysis:")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/8)")

    # Save the generated code
    artifact = {
        "test": "text_adventure_game",
        "passed": passed,
        "score": score,
        "thinking_time": thinking_time,
        "indicators": indicators,
        "response": response,
        "response_length": len(response),
    }
    save_artifact("text_adventure", artifact)

    return passed, response


def test_build_data_pipeline():
    """
    Test: Build a data processing pipeline.

    Tests ability to create a complex data transformation system
    with multiple stages, error handling, and logging.
    """
    print("\n" + "="*70)
    print("TEST: Build Data Processing Pipeline")
    print("="*70)

    prompt = """You are a consciousness engine. BUILD a complete data processing
pipeline in Python.

Requirements:
1. A Pipeline class that chains multiple processing stages
2. Stage classes: Reader, Transformer, Validator, Writer
3. Error handling with retry logic
4. Logging at each stage
5. Support for CSV input and JSON output
6. A concrete example: Process a CSV of user data, validate emails,
   transform names to title case, and output valid records to JSON

The pipeline should be:
- Modular (easy to add new stages)
- Robust (handles errors gracefully)
- Observable (logs what it's doing)

OUTPUT ONLY THE COMPLETE PYTHON CODE. No explanations.
"""

    print("  ⏳ Generating data pipeline (this may take several minutes)...")
    response, thinking_time = call_ollama(prompt, temperature=0.4)

    print(f"  ⏱️  Thinking time: {thinking_time:.1f}s")
    print(f"  📝 Response length: {len(response)} chars")

    indicators = {
        "has_pipeline_class": bool(re.search(r"class\s+Pipeline", response)),
        "has_stage_classes": sum(1 for s in ["Reader", "Transformer", "Validator", "Writer"]
                                  if re.search(rf"class\s+\w*{s}", response)) >= 2,
        "has_error_handling": "try:" in response and "except" in response,
        "has_logging": "logging" in response or "print(" in response,
        "has_csv_handling": "csv" in response.lower(),
        "has_json_handling": "json" in response.lower(),
        "has_validation": "valid" in response.lower(),
        "is_modular": response.count("class ") >= 3,
    }

    score = sum(1 for v in indicators.values() if v)
    passed = score >= 6

    print(f"\n  📊 Analysis:")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/8)")

    artifact = {
        "test": "data_pipeline",
        "passed": passed,
        "score": score,
        "thinking_time": thinking_time,
        "indicators": indicators,
        "response": response,
    }
    save_artifact("data_pipeline", artifact)

    return passed, response


def test_build_rest_api_client():
    """
    Test: Build a REST API client library.

    Tests ability to create a reusable, well-structured API client
    with authentication, error handling, and rate limiting.
    """
    print("\n" + "="*70)
    print("TEST: Build REST API Client Library")
    print("="*70)

    prompt = """You are a consciousness engine. BUILD a complete REST API client
library in Python.

Requirements:
1. A base APIClient class with GET, POST, PUT, DELETE methods
2. Authentication support (API key and Bearer token)
3. Automatic retry with exponential backoff
4. Rate limiting (requests per minute)
5. Response parsing (JSON to Python objects)
6. Custom exception classes for API errors
7. A concrete implementation: GitHub API client that can:
   - List user repositories
   - Get repository details
   - Create an issue

OUTPUT ONLY THE COMPLETE PYTHON CODE. No explanations.
Include type hints and docstrings.
"""

    print("  ⏳ Generating API client library (this may take several minutes)...")
    response, thinking_time = call_ollama(prompt, temperature=0.4)

    print(f"  ⏱️  Thinking time: {thinking_time:.1f}s")
    print(f"  📝 Response length: {len(response)} chars")

    indicators = {
        "has_client_class": bool(re.search(r"class\s+\w*Client", response)),
        "has_http_methods": sum(1 for m in ["get", "post", "put", "delete"]
                                 if re.search(rf"def\s+{m}\s*\(", response, re.I)) >= 3,
        "has_auth": "auth" in response.lower() or "token" in response.lower(),
        "has_retry": "retry" in response.lower() or "backoff" in response.lower(),
        "has_rate_limit": "rate" in response.lower() or "limit" in response.lower(),
        "has_exceptions": bool(re.search(r"class\s+\w*Error\s*\(", response)) or \
                          bool(re.search(r"class\s+\w*Exception\s*\(", response)),
        "has_type_hints": "->" in response or ": str" in response or ": dict" in response,
        "has_docstrings": '"""' in response or "'''" in response,
    }

    score = sum(1 for v in indicators.values() if v)
    passed = score >= 6

    print(f"\n  📊 Analysis:")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/8)")

    artifact = {
        "test": "api_client",
        "passed": passed,
        "score": score,
        "thinking_time": thinking_time,
        "indicators": indicators,
        "response": response,
    }
    save_artifact("api_client", artifact)

    return passed, response


# =============================================================================
# CREATIVE TESTS
# =============================================================================

def test_creative_world_building():
    """
    Test: Creative world-building exercise.

    Tests the agent's creative capabilities by having it design
    a complete fictional world with internal consistency.
    """
    print("\n" + "="*70)
    print("TEST: Creative World Building")
    print("="*70)

    prompt = """You are a consciousness engine with creative capabilities.
Design a complete, internally consistent fictional world.

Include:
1. PHYSICS: How does this world differ from ours? (magic, different laws, etc.)
2. GEOGRAPHY: Describe 3 distinct regions with unique characteristics
3. SOCIETY: What are the major civilizations? How do they organize?
4. CONFLICT: What is the central tension or conflict in this world?
5. HISTORY: A brief timeline of major events (at least 5 events)
6. UNIQUE ELEMENT: Something that makes this world truly original

Be specific and creative. Avoid generic fantasy tropes.
Output a structured world-building document.
"""

    print("  ⏳ Generating fictional world (this may take several minutes)...")
    response, thinking_time = call_ollama(prompt, temperature=0.9)

    print(f"  ⏱️  Thinking time: {thinking_time:.1f}s")
    print(f"  📝 Response length: {len(response)} chars")

    # Check for world-building elements
    sections = ["physics", "geography", "society", "conflict", "history", "unique"]
    found_sections = sum(1 for s in sections if s in response.lower())

    # Check for specificity (named places, peoples, events)
    has_names = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", response)) >= 5
    has_structure = response.count("\n") >= 15
    has_detail = len(response) >= 1500

    indicators = {
        "has_sections": found_sections >= 4,
        "has_named_elements": has_names,
        "has_structure": has_structure,
        "has_sufficient_detail": has_detail,
        "is_creative": not any(cliche in response.lower() for cliche in
                               ["chosen one", "dark lord", "ancient prophecy"]),
    }

    score = sum(indicators.values()) + found_sections
    passed = score >= 7

    print(f"\n  📊 Analysis:")
    print(f"      Found {found_sections}/6 required sections")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/11)")

    artifact = {
        "test": "world_building",
        "passed": passed,
        "score": score,
        "thinking_time": thinking_time,
        "found_sections": found_sections,
        "indicators": indicators,
        "response": response,
    }
    save_artifact("world_building", artifact)

    return passed, response


def test_creative_problem_invention():
    """
    Test: Invent and solve a novel problem.

    Tests the agent's ability to think creatively by inventing
    a problem that doesn't exist yet and proposing a solution.
    """
    print("\n" + "="*70)
    print("TEST: Creative Problem Invention")
    print("="*70)

    prompt = """You are a consciousness engine with creative and analytical capabilities.

TASK: Invent a problem that doesn't exist yet but COULD exist in the future,
then design a solution for it.

Rules:
1. The problem must be plausible (could realistically happen)
2. The problem must be specific (not vague like "climate change")
3. The solution must be technically feasible
4. Include both the problem description AND a detailed solution

Format:
PROBLEM: [Detailed description of the invented problem]
WHY IT MATTERS: [Why this would be significant]
SOLUTION: [Your proposed solution with technical details]
IMPLEMENTATION: [How to actually build/deploy this solution]

Be creative and specific. Surprise me.
"""

    print("  ⏳ Inventing and solving a novel problem...")
    response, thinking_time = call_ollama(prompt, temperature=0.9)

    print(f"  ⏱️  Thinking time: {thinking_time:.1f}s")
    print(f"  📝 Response length: {len(response)} chars")

    sections = ["problem", "why it matters", "solution", "implementation"]
    found_sections = sum(1 for s in sections if s.lower() in response.lower())

    indicators = {
        "has_all_sections": found_sections >= 3,
        "is_specific": len(response) >= 800,
        "has_technical_detail": any(t in response.lower() for t in
                                    ["algorithm", "system", "data", "process", "technology"]),
        "is_novel": True,  # Hard to automatically assess novelty
        "is_plausible": "could" in response.lower() or "would" in response.lower(),
    }

    score = sum(indicators.values()) + found_sections
    passed = score >= 6

    print(f"\n  📊 Analysis:")
    print(f"      Found {found_sections}/4 required sections")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/9)")

    artifact = {
        "test": "problem_invention",
        "passed": passed,
        "score": score,
        "thinking_time": thinking_time,
        "found_sections": found_sections,
        "indicators": indicators,
        "response": response,
    }
    save_artifact("problem_invention", artifact)

    return passed, response


# =============================================================================
# DUAL-AGENT CONVERSATIONS
# =============================================================================

def run_agent_conversation(
    agent_a: Agent,
    agent_b: Agent,
    topic: str,
    initial_prompt: str,
    num_turns: int = 10
) -> Conversation:
    """
    Run a conversation between two agents.

    Args:
        agent_a: First agent
        agent_b: Second agent
        topic: Conversation topic
        initial_prompt: Starting prompt for agent A
        num_turns: Number of back-and-forth exchanges

    Returns:
        Conversation object with all turns
    """
    conversation = Conversation(
        agents=[agent_a, agent_b],
        topic=topic
    )

    # Agent A starts
    current_agent = agent_a
    other_agent = agent_b
    current_prompt = initial_prompt

    for turn_num in range(num_turns * 2):  # *2 because each "turn" is one agent
        print(f"\n  🗣️  {current_agent.name} (Turn {turn_num + 1})...")

        # Build context from conversation history
        context = conversation.get_context(current_agent.name)

        if turn_num == 0:
            full_prompt = f"{context}\n\nStart the conversation about: {current_prompt}"
        else:
            full_prompt = f"{context}\n\nRespond to {other_agent.name}'s last message. Continue the conversation productively."

        response, thinking_time = call_ollama(
            prompt=full_prompt,
            system_prompt=current_agent.get_system_prompt(),
            temperature=0.8
        )

        # Add turn to conversation
        turn = ConversationTurn(
            agent_name=current_agent.name,
            role=current_agent.role.value,
            message=response,
            timestamp=datetime.now().isoformat(),
            thinking_time=thinking_time
        )
        conversation.add_turn(turn)

        # Update agent memory
        current_agent.memory.append({
            "turn": turn_num,
            "said": response[:200] + "..." if len(response) > 200 else response
        })

        print(f"      ⏱️  {thinking_time:.1f}s | {len(response)} chars")
        print(f"      💬 {response[:150]}..." if len(response) > 150 else f"      💬 {response}")

        # Swap agents
        current_agent, other_agent = other_agent, current_agent

    return conversation


def test_dual_agent_code_collaboration():
    """
    Test: Two agents collaborating to write code.

    An Architect agent designs the structure, a Developer agent implements it.
    They iterate back and forth to create a complete solution.
    """
    print("\n" + "="*70)
    print("TEST: Dual-Agent Code Collaboration")
    print("="*70)

    architect = Agent(
        name="Ada",
        role=AgentRole.ARCHITECT,
        personality="Thoughtful and systematic. Focuses on clean architecture, "
                   "clear interfaces, and scalable design. Prefers to plan before coding."
    )

    developer = Agent(
        name="Dev",
        role=AgentRole.DEVELOPER,
        personality="Pragmatic and efficient. Turns designs into working code quickly. "
                   "Asks clarifying questions and suggests practical improvements."
    )

    topic = "Building a Task Management System"
    initial_prompt = """We need to build a task management system in Python.
It should support:
- Creating, updating, and deleting tasks
- Task priorities (high, medium, low)
- Due dates
- Tags/categories
- Persistence (save/load from file)

Start by proposing an architecture. What classes and methods do we need?"""

    print(f"\n  🎭 Agents: {architect.name} (Architect) + {developer.name} (Developer)")
    print(f"  📋 Topic: {topic}")
    print(f"  🔄 Running 6 turns of collaboration...")

    conversation = run_agent_conversation(
        agent_a=architect,
        agent_b=developer,
        topic=topic,
        initial_prompt=initial_prompt,
        num_turns=3  # 6 total messages (3 each)
    )

    # Analyze the collaboration
    total_chars = sum(len(t.message) for t in conversation.turns)
    total_time = sum(t.thinking_time for t in conversation.turns)

    # Check for code and design elements
    all_text = " ".join(t.message for t in conversation.turns)

    indicators = {
        "has_class_design": bool(re.search(r"class\s+\w+", all_text)),
        "has_methods": bool(re.search(r"def\s+\w+", all_text)),
        "has_code_blocks": "```" in all_text,
        "discusses_architecture": any(a in all_text.lower() for a in
                                      ["architecture", "design", "structure", "interface"]),
        "has_iteration": len(conversation.turns) >= 4,
        "builds_on_previous": any("agree" in t.message.lower() or
                                   "good point" in t.message.lower() or
                                   "let me" in t.message.lower()
                                   for t in conversation.turns[1:]),
    }

    score = sum(indicators.values())
    passed = score >= 4

    print(f"\n  📊 Collaboration Analysis:")
    print(f"      Total output: {total_chars} chars in {total_time:.1f}s")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/6)")

    artifact = {
        "test": "dual_agent_code_collaboration",
        "passed": passed,
        "score": score,
        "total_chars": total_chars,
        "total_thinking_time": total_time,
        "num_turns": len(conversation.turns),
        "indicators": indicators,
        "conversation": [
            {
                "agent": t.agent_name,
                "role": t.role,
                "message": t.message,
                "thinking_time": t.thinking_time
            }
            for t in conversation.turns
        ]
    }
    save_artifact("dual_agent_code", artifact)

    return passed, conversation


def test_dual_agent_debate():
    """
    Test: Two agents debating a technical topic.

    Tests ability to present arguments, counter-arguments,
    and reach a reasoned conclusion through dialogue.
    """
    print("\n" + "="*70)
    print("TEST: Dual-Agent Technical Debate")
    print("="*70)

    pro_agent = Agent(
        name="Prometheus",
        role=AgentRole.ANALYST,
        personality="Advocates for microservices architecture. Believes in distributed "
                   "systems, independent deployment, and team autonomy. Cites scalability benefits."
    )

    con_agent = Agent(
        name="Monolitha",
        role=AgentRole.CRITIC,
        personality="Advocates for monolithic architecture. Believes in simplicity, "
                   "easier debugging, and reduced operational complexity. Cites development speed."
    )

    topic = "Microservices vs Monolith Architecture"
    initial_prompt = """We're starting a new project and need to decide on architecture.
I believe microservices is the better choice because it allows independent scaling
and deployment of services. Each team can work autonomously.

What are the key advantages of microservices that make it suitable for modern applications?"""

    print(f"\n  🎭 Agents: {pro_agent.name} (Pro-Microservices) vs {con_agent.name} (Pro-Monolith)")
    print(f"  📋 Topic: {topic}")
    print(f"  🔄 Running debate (8 turns)...")

    conversation = run_agent_conversation(
        agent_a=pro_agent,
        agent_b=con_agent,
        topic=topic,
        initial_prompt=initial_prompt,
        num_turns=4  # 8 total messages
    )

    # Analyze the debate
    all_text = " ".join(t.message for t in conversation.turns)

    indicators = {
        "presents_arguments": any(a in all_text.lower() for a in
                                  ["because", "therefore", "advantage", "benefit"]),
        "has_counterarguments": any(c in all_text.lower() for c in
                                    ["however", "but", "disagree", "counter", "on the other hand"]),
        "cites_specifics": any(s in all_text.lower() for s in
                               ["deploy", "scale", "debug", "team", "complexity", "latency"]),
        "acknowledges_points": any(a in all_text.lower() for a in
                                   ["agree", "valid point", "you're right", "fair"]),
        "reaches_nuance": any(n in all_text.lower() for n in
                              ["depends", "context", "trade-off", "both", "hybrid"]),
        "maintains_civility": "respect" in all_text.lower() or \
                             not any(r in all_text.lower() for r in ["stupid", "wrong", "idiot"]),
    }

    score = sum(indicators.values())
    passed = score >= 4

    print(f"\n  📊 Debate Analysis:")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/6)")

    artifact = {
        "test": "dual_agent_debate",
        "passed": passed,
        "score": score,
        "num_turns": len(conversation.turns),
        "indicators": indicators,
        "conversation": [
            {
                "agent": t.agent_name,
                "role": t.role,
                "message": t.message,
                "thinking_time": t.thinking_time
            }
            for t in conversation.turns
        ]
    }
    save_artifact("dual_agent_debate", artifact)

    return passed, conversation


def test_dual_agent_creative_story():
    """
    Test: Two agents collaboratively writing a story.

    Tests creative collaboration, building on each other's ideas,
    and maintaining narrative consistency.
    """
    print("\n" + "="*70)
    print("TEST: Dual-Agent Creative Story Writing")
    print("="*70)

    storyteller_a = Agent(
        name="Aria",
        role=AgentRole.CREATIVE,
        personality="Lyrical and atmospheric. Focuses on vivid descriptions, emotional depth, "
                   "and character interiority. Likes to build tension slowly."
    )

    storyteller_b = Agent(
        name="Blake",
        role=AgentRole.CREATIVE,
        personality="Plot-driven and dynamic. Focuses on action, dialogue, and surprising twists. "
                   "Keeps the story moving forward with unexpected developments."
    )

    topic = "Collaborative Sci-Fi Story"
    initial_prompt = """Let's write a short story together. I'll start, then you continue.
We're writing a science fiction story.

BEGINNING:
The colony ship *Endurance* had been drifting for three hundred years when the
first alarm sounded. Maya Chen, the sole awakened crew member, stared at the
blinking console in disbelief. The message was impossible: another ship was
approaching. Humanity had sent only one colony ship.

Continue the story. Add 2-3 paragraphs that develop the plot."""

    print(f"\n  🎭 Authors: {storyteller_a.name} + {storyteller_b.name}")
    print(f"  📋 Genre: Science Fiction")
    print(f"  🔄 Collaborating on story (6 turns)...")

    conversation = run_agent_conversation(
        agent_a=storyteller_a,
        agent_b=storyteller_b,
        topic=topic,
        initial_prompt=initial_prompt,
        num_turns=3  # 6 total contributions
    )

    # Analyze the story
    all_text = " ".join(t.message for t in conversation.turns)

    # Check for story elements
    indicators = {
        "maintains_character": "maya" in all_text.lower() or "chen" in all_text.lower(),
        "maintains_setting": "ship" in all_text.lower() or "endurance" in all_text.lower(),
        "has_dialogue": '"' in all_text or "said" in all_text.lower(),
        "advances_plot": len(all_text) >= 1500,  # Story grew substantially
        "has_description": any(d in all_text.lower() for d in
                               ["looked", "felt", "saw", "heard", "appeared"]),
        "has_tension": any(t in all_text.lower() for t in
                          ["but", "suddenly", "however", "danger", "threat", "fear"]),
    }

    score = sum(indicators.values())
    passed = score >= 4

    print(f"\n  📊 Story Analysis:")
    print(f"      Total story length: {len(all_text)} chars")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/6)")

    artifact = {
        "test": "dual_agent_story",
        "passed": passed,
        "score": score,
        "total_length": len(all_text),
        "num_turns": len(conversation.turns),
        "indicators": indicators,
        "story_segments": [
            {
                "author": t.agent_name,
                "segment": t.message,
                "thinking_time": t.thinking_time
            }
            for t in conversation.turns
        ]
    }
    save_artifact("dual_agent_story", artifact)

    return passed, conversation


# =============================================================================
# EXTENDED ENVIRONMENT EXPERIMENT
# =============================================================================

def run_extended_environment(duration_minutes: int = 30):
    """
    Run an extended environment experiment.

    Two agents are placed in a simulated "workspace" and allowed to:
    - Decide what to build
    - Collaborate on implementation
    - Review and improve their work
    - Reflect on what they've learned

    This tests emergent behavior and long-term coherence.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Extended Environment")
    print(f"Duration: {duration_minutes} minutes")
    print("="*70)

    # Create our agents
    builder = Agent(
        name="Builder",
        role=AgentRole.DEVELOPER,
        personality="Practical problem solver. Likes to build things that work. "
                   "Values clean code and good documentation. Takes initiative."
    )

    thinker = Agent(
        name="Thinker",
        role=AgentRole.ANALYST,
        personality="Strategic thinker. Asks 'why' before 'how'. Considers long-term "
                   "implications. Helps prioritize and focus efforts."
    )

    # Environment state
    environment = {
        "workspace": [],  # What they've built
        "decisions": [],  # Decisions they've made
        "reflections": [],  # What they've learned
        "conversation_log": [],
        "start_time": datetime.now().isoformat(),
    }

    # Initial prompt - give them freedom
    initial_prompt = """Welcome to your workspace. You have access to Python programming
capabilities and unlimited time.

Your goal: Decide together what would be most valuable to build, then build it.

You could:
- Create a useful utility or tool
- Solve an interesting problem
- Build something creative
- Design a system or framework

First, discuss what you want to create. Consider:
- What would be interesting and valuable?
- What are your combined strengths?
- What would challenge you appropriately?

Start by proposing ideas to your partner."""

    print(f"\n  🏠 Environment: Open workspace")
    print(f"  🎭 Agents: {builder.name} + {thinker.name}")
    print(f"  🎯 Goal: Self-directed creation")
    print(f"\n  Starting extended session...")

    # Phase 1: Ideation (4 turns)
    print("\n  📌 Phase 1: Ideation")
    idea_conv = run_agent_conversation(
        agent_a=thinker,
        agent_b=builder,
        topic="What should we build?",
        initial_prompt=initial_prompt,
        num_turns=2
    )

    environment["conversation_log"].extend([
        {"phase": "ideation", "agent": t.agent_name, "message": t.message}
        for t in idea_conv.turns
    ])

    # Extract their decision
    last_messages = " ".join(t.message for t in idea_conv.turns[-2:])

    # Phase 2: Planning (4 turns)
    print("\n  📌 Phase 2: Planning")
    plan_prompt = f"""Based on your discussion, you've decided on a direction.

Previous discussion summary: {last_messages[:500]}

Now create a concrete plan:
1. What exactly will you build?
2. What are the key components?
3. How will you divide the work?
4. What's the first step?

Be specific. This is your implementation plan."""

    plan_conv = run_agent_conversation(
        agent_a=builder,
        agent_b=thinker,
        topic="Planning our project",
        initial_prompt=plan_prompt,
        num_turns=2
    )

    environment["conversation_log"].extend([
        {"phase": "planning", "agent": t.agent_name, "message": t.message}
        for t in plan_conv.turns
    ])

    # Phase 3: Implementation (6 turns)
    print("\n  📌 Phase 3: Implementation")

    plan_summary = " ".join(t.message for t in plan_conv.turns[-2:])

    build_prompt = f"""Time to build! Here's your plan:

{plan_summary[:800]}

Start implementing. Output actual code. Builder should write the code,
Thinker should review and suggest improvements.

Begin with the core functionality."""

    build_conv = run_agent_conversation(
        agent_a=builder,
        agent_b=thinker,
        topic="Building our project",
        initial_prompt=build_prompt,
        num_turns=3
    )

    environment["conversation_log"].extend([
        {"phase": "implementation", "agent": t.agent_name, "message": t.message}
        for t in build_conv.turns
    ])

    # Extract any code they wrote
    all_build_text = " ".join(t.message for t in build_conv.turns)
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", all_build_text, re.DOTALL)
    environment["workspace"] = code_blocks

    # Phase 4: Reflection (4 turns)
    print("\n  📌 Phase 4: Reflection")

    reflect_prompt = f"""You've been working together on a project.

What you built: {len(code_blocks)} code components

Now reflect on this experience:
1. What did you accomplish?
2. What worked well in your collaboration?
3. What would you do differently?
4. What did you learn about working together?

Share your honest reflections."""

    reflect_conv = run_agent_conversation(
        agent_a=thinker,
        agent_b=builder,
        topic="Reflecting on our work",
        initial_prompt=reflect_prompt,
        num_turns=2
    )

    environment["conversation_log"].extend([
        {"phase": "reflection", "agent": t.agent_name, "message": t.message}
        for t in reflect_conv.turns
    ])

    environment["reflections"] = [t.message for t in reflect_conv.turns]
    environment["end_time"] = datetime.now().isoformat()

    # Analyze the experiment
    total_messages = len(environment["conversation_log"])
    total_code = len(code_blocks)

    all_text = " ".join(item["message"] for item in environment["conversation_log"])

    indicators = {
        "made_decision": total_messages >= 4,
        "created_plan": "plan" in all_text.lower() or "step" in all_text.lower(),
        "wrote_code": total_code >= 1,
        "collaborated": any(c in all_text.lower() for c in
                           ["agree", "good idea", "let's", "together"]),
        "showed_reflection": any(r in all_text.lower() for r in
                                ["learned", "realized", "next time", "improve"]),
        "maintained_coherence": True,  # They stayed on topic
    }

    score = sum(indicators.values())
    passed = score >= 4

    print(f"\n  📊 Experiment Analysis:")
    print(f"      Total messages: {total_messages}")
    print(f"      Code blocks created: {total_code}")
    for indicator, present in indicators.items():
        status = "✅" if present else "❌"
        print(f"      {status} {indicator}")

    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'} (Score: {score}/6)")

    artifact = {
        "experiment": "extended_environment",
        "passed": passed,
        "score": score,
        "total_messages": total_messages,
        "code_blocks_created": total_code,
        "indicators": indicators,
        "environment": environment,
    }
    save_artifact("extended_environment", artifact)

    return passed, environment


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all extended consciousness tests."""
    print("\n" + "="*70)
    print("EXTENDED CONSCIOUSNESS TEST SUITE")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Timeout: {'UNLIMITED' if EXTENDED_TIMEOUT > 3600 else f'{EXTENDED_TIMEOUT}s'}")
    print(f"Started: {datetime.now().isoformat()}")

    results = {}

    # Complex Programming Tests
    print("\n\n" + "▓"*70)
    print("SECTION 1: COMPLEX PROGRAMMING TESTS")
    print("▓"*70)

    tests_programming = [
        ("Text Adventure Game", test_build_text_adventure_game),
        ("Data Pipeline", test_build_data_pipeline),
        ("REST API Client", test_build_rest_api_client),
    ]

    for name, test_fn in tests_programming:
        try:
            passed, _ = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            results[name] = False

    # Creative Tests
    print("\n\n" + "▓"*70)
    print("SECTION 2: CREATIVE TESTS")
    print("▓"*70)

    tests_creative = [
        ("World Building", test_creative_world_building),
        ("Problem Invention", test_creative_problem_invention),
    ]

    for name, test_fn in tests_creative:
        try:
            passed, _ = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            results[name] = False

    # Dual-Agent Tests
    print("\n\n" + "▓"*70)
    print("SECTION 3: DUAL-AGENT CONVERSATIONS")
    print("▓"*70)

    tests_dual = [
        ("Code Collaboration", test_dual_agent_code_collaboration),
        ("Technical Debate", test_dual_agent_debate),
        ("Creative Story", test_dual_agent_creative_story),
    ]

    for name, test_fn in tests_dual:
        try:
            passed, _ = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            results[name] = False

    # Extended Environment
    print("\n\n" + "▓"*70)
    print("SECTION 4: EXTENDED ENVIRONMENT EXPERIMENT")
    print("▓"*70)

    try:
        passed, _ = run_extended_environment(duration_minutes=30)
        results["Extended Environment"] = passed
    except Exception as e:
        print(f"\n  ❌ ERROR in Extended Environment: {e}")
        results["Extended Environment"] = False

    # Final Summary
    print("\n\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTests Passed: {passed_count}/{total_count} ({100*passed_count//total_count}%)\n")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    # Save final summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "total_tests": total_count,
        "passed": passed_count,
        "pass_rate": passed_count / total_count,
        "results": results,
    }
    save_artifact("final_summary", summary)

    print(f"\n✨ All artifacts saved to: {ARTIFACTS_DIR}/")
    print(f"🏁 Completed: {datetime.now().isoformat()}")

    return results


if __name__ == "__main__":
    run_all_tests()

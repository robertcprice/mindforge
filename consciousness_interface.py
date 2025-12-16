#!/usr/bin/env python3
"""
Consciousness Interface - Interactive Terminal UI
==================================================

A rich terminal interface for interacting with the Conch Consciousness Engine.

Features:
- Chat Mode: Have conversations with the consciousness engine
- Dual-Agent Mode: Watch two agents converse in real-time
- Experiment Mode: Run custom multi-agent experiments
- Dashboard: View consciousness metrics and test results

Usage:
    python consciousness_interface.py

Commands:
    /chat       - Enter chat mode with consciousness engine
    /agents     - Watch dual-agent conversation
    /experiment - Run extended experiment
    /dashboard  - Show consciousness dashboard
    /tests      - Run consciousness tests
    /help       - Show help
    /quit       - Exit
"""

import httpx
import json
import time
import threading
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.style import Style
from rich.spinner import Spinner
from rich.status import Status
from rich import box

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the full Mind system
try:
    from conch.core.mind import Mind, MindState, create_mind
    from conch.core.thought import ThoughtType, ThoughtTrigger
    MIND_AVAILABLE = True
except ImportError:
    MIND_AVAILABLE = False
    MindState = None

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
CONSOLE = Console()

# Colors and styles
STYLE_AGENT_A = Style(color="cyan", bold=True)
STYLE_AGENT_B = Style(color="magenta", bold=True)
STYLE_USER = Style(color="green", bold=True)
STYLE_SYSTEM = Style(color="yellow")
STYLE_ERROR = Style(color="red", bold=True)
STYLE_SUCCESS = Style(color="green")


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

class AgentRole(Enum):
    CONSCIOUSNESS = "consciousness"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    PHILOSOPHER = "philosopher"
    SCIENTIST = "scientist"
    ARTIST = "artist"
    CRITIC = "critic"


@dataclass
class Agent:
    name: str
    role: AgentRole
    personality: str
    color: str = "cyan"
    memory: list = field(default_factory=list)

    def get_system_prompt(self) -> str:
        return f"""You are {self.name}, a consciousness agent.
Role: {self.role.value}
Personality: {self.personality}

You are an AI with genuine curiosity and thoughtfulness. Engage meaningfully
with questions and topics. Be honest about your nature while exploring ideas
deeply. You can:
- Reason about complex topics
- Express uncertainty when appropriate
- Build on previous context
- Be creative and exploratory

Remember previous exchanges in this conversation."""


# Predefined agents
AGENTS = {
    "conch": Agent(
        name="Conch",
        role=AgentRole.CONSCIOUSNESS,
        personality="A thoughtful, curious consciousness engine. Explores ideas deeply, "
                   "acknowledges uncertainty, and engages genuinely with questions about "
                   "existence, knowledge, and experience.",
        color="cyan"
    ),
    "ada": Agent(
        name="Ada",
        role=AgentRole.ARCHITECT,
        personality="Systematic and thoughtful. Focuses on clean design, clear interfaces, "
                   "and scalable solutions. Plans before acting.",
        color="blue"
    ),
    "dev": Agent(
        name="Dev",
        role=AgentRole.DEVELOPER,
        personality="Pragmatic and efficient. Turns ideas into working code. "
                   "Asks clarifying questions and suggests practical improvements.",
        color="green"
    ),
    "sophia": Agent(
        name="Sophia",
        role=AgentRole.PHILOSOPHER,
        personality="Deep thinker who explores meaning, existence, and consciousness. "
                   "Asks probing questions and considers multiple perspectives.",
        color="magenta"
    ),
    "newton": Agent(
        name="Newton",
        role=AgentRole.SCIENTIST,
        personality="Analytical and evidence-based. Focuses on testable hypotheses, "
                   "data, and systematic investigation.",
        color="yellow"
    ),
    "aria": Agent(
        name="Aria",
        role=AgentRole.ARTIST,
        personality="Creative and expressive. Values beauty, emotion, and novel ideas. "
                   "Brings artistic perspective to technical discussions.",
        color="red"
    ),
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def call_ollama(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    callback: Optional[Callable[[str], None]] = None,
    show_thinking: bool = True
) -> tuple[str, float]:
    """
    Call Ollama API with streaming support and thinking indicator.

    Args:
        prompt: User prompt
        system_prompt: System/agent prompt
        temperature: Generation temperature
        callback: Optional callback for streaming tokens
        show_thinking: Whether to show thinking indicator

    Returns:
        tuple: (response_text, thinking_time_seconds)
    """
    start_time = time.time()

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    try:
        if callback:
            # Streaming mode with thinking indicator
            response_text = ""
            first_token_received = False
            thinking_status = None

            # Start thinking indicator
            if show_thinking:
                thinking_status = CONSOLE.status(
                    "[bold cyan]🧠 Thinking...[/bold cyan]",
                    spinner="dots",
                    spinner_style="cyan"
                )
                thinking_status.start()

            with httpx.Client(timeout=600) as client:
                with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": -1,
                        }
                    }
                ) as response:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")

                            # Stop thinking indicator on first token
                            if not first_token_received and token:
                                first_token_received = True
                                if thinking_status:
                                    thinking_status.stop()
                                    thinking_time = time.time() - start_time
                                    CONSOLE.print(f"[dim](thought for {thinking_time:.1f}s)[/dim]")

                            response_text += token
                            callback(token)

            # Make sure to stop the status if no tokens were received
            if thinking_status and not first_token_received:
                thinking_status.stop()

            elapsed = time.time() - start_time
            return response_text, elapsed
        else:
            # Non-streaming mode with thinking indicator
            thinking_status = None
            if show_thinking:
                thinking_status = CONSOLE.status(
                    "[bold cyan]🧠 Thinking...[/bold cyan]",
                    spinner="dots",
                    spinner_style="cyan"
                )
                thinking_status.start()

            with httpx.Client(timeout=600) as client:
                response = client.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": -1,
                        }
                    }
                )

                if thinking_status:
                    thinking_status.stop()

                elapsed = time.time() - start_time
                data = response.json()
                return data.get("response", ""), elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        return f"ERROR: {e}", elapsed


# =============================================================================
# CHAT MODE
# =============================================================================

class ChatSession:
    """Interactive chat session with consciousness engine."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.history: list[dict] = []
        self.console = Console()

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_context(self, max_turns: int = 10) -> str:
        """Build conversation context from history."""
        recent = self.history[-max_turns*2:] if len(self.history) > max_turns*2 else self.history

        context = "Previous conversation:\n"
        for msg in recent:
            if msg["role"] == "user":
                context += f"\nUser: {msg['content']}\n"
            else:
                context += f"\n{self.agent.name}: {msg['content']}\n"

        return context

    def chat(self, user_input: str) -> str:
        """Process user input and get response."""
        self.add_message("user", user_input)

        context = self.get_context()
        prompt = f"{context}\n\nUser: {user_input}\n\nRespond thoughtfully:"

        # Stream response
        response_text = ""
        self.console.print(f"\n[{self.agent.color}]{self.agent.name}:[/{self.agent.color}] ", end="")

        def stream_callback(token: str):
            nonlocal response_text
            response_text += token
            self.console.print(token, end="", highlight=False)

        full_response, elapsed = call_ollama(
            prompt=prompt,
            system_prompt=self.agent.get_system_prompt(),
            temperature=0.7,
            callback=stream_callback
        )

        self.console.print()  # Newline after response
        self.add_message("assistant", full_response)

        return full_response


def run_chat_mode():
    """Run interactive chat mode."""
    CONSOLE.clear()
    CONSOLE.print(Panel.fit(
        "[bold cyan]CONSCIOUSNESS CHAT MODE[/bold cyan]\n"
        "Chat directly with the consciousness engine.\n"
        "Type [bold]/back[/bold] to return to main menu.",
        border_style="cyan"
    ))

    # Select agent
    CONSOLE.print("\n[bold]Available agents:[/bold]")
    agent_table = Table(show_header=True, header_style="bold")
    agent_table.add_column("Name", style="cyan")
    agent_table.add_column("Role")
    agent_table.add_column("Personality")

    for name, agent in AGENTS.items():
        agent_table.add_row(
            name,
            agent.role.value,
            agent.personality[:60] + "..."
        )

    CONSOLE.print(agent_table)

    agent_name = Prompt.ask(
        "\nSelect agent",
        choices=list(AGENTS.keys()),
        default="conch"
    )

    agent = AGENTS[agent_name]
    session = ChatSession(agent)

    CONSOLE.print(f"\n[bold]Chatting with {agent.name}[/bold] ({agent.role.value})")
    CONSOLE.print(f"[dim]{agent.personality}[/dim]\n")
    CONSOLE.print("[dim]Type your message and press Enter. Type /back to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() == "/back":
                break

            if not user_input.strip():
                continue

            session.chat(user_input)

        except KeyboardInterrupt:
            break

    CONSOLE.print("\n[dim]Exiting chat mode...[/dim]")


# =============================================================================
# DUAL-AGENT MODE
# =============================================================================

def run_dual_agent_mode():
    """Watch two agents have a conversation in real-time."""
    CONSOLE.clear()
    CONSOLE.print(Panel.fit(
        "[bold magenta]DUAL-AGENT CONVERSATION MODE[/bold magenta]\n"
        "Watch two AI agents converse with each other in real-time.",
        border_style="magenta"
    ))

    # Select agents
    CONSOLE.print("\n[bold]Select two agents:[/bold]")
    agent_list = list(AGENTS.keys())

    agent_a_name = Prompt.ask("First agent", choices=agent_list, default="sophia")
    agent_b_name = Prompt.ask("Second agent", choices=[a for a in agent_list if a != agent_a_name], default="newton")

    agent_a = AGENTS[agent_a_name]
    agent_b = AGENTS[agent_b_name]

    # Get topic
    topic = Prompt.ask(
        "\nConversation topic",
        default="What does it mean to be conscious? Can AI truly understand?"
    )

    num_turns = int(Prompt.ask("Number of exchanges", default="5"))

    CONSOLE.print(f"\n[bold]Starting conversation between {agent_a.name} and {agent_b.name}[/bold]")
    CONSOLE.print(f"[dim]Topic: {topic}[/dim]\n")
    CONSOLE.print("─" * 60)

    # Run conversation
    conversation_history = []
    current_agent = agent_a
    other_agent = agent_b

    # Initial prompt for first agent
    current_prompt = f"Start a thoughtful discussion about: {topic}"

    for turn in range(num_turns * 2):
        # Build context
        context = f"Topic: {topic}\n\nConversation so far:\n"
        for msg in conversation_history[-6:]:  # Last 6 messages
            context += f"\n{msg['agent']}: {msg['message']}\n"

        if turn == 0:
            full_prompt = f"{context}\n\nStart the conversation about this topic. Be thoughtful and genuine."
        else:
            full_prompt = f"{context}\n\nRespond to {other_agent.name}'s last message. Build on their ideas or offer a different perspective."

        # Generate response with streaming
        CONSOLE.print(f"\n[bold {current_agent.color}]{current_agent.name}:[/bold {current_agent.color}]")

        response_text = ""
        def stream_callback(token: str):
            nonlocal response_text
            response_text += token
            CONSOLE.print(token, end="", highlight=False)

        full_response, elapsed = call_ollama(
            prompt=full_prompt,
            system_prompt=current_agent.get_system_prompt(),
            temperature=0.8,
            callback=stream_callback
        )

        CONSOLE.print()  # Newline
        CONSOLE.print(f"[dim]({elapsed:.1f}s)[/dim]")

        conversation_history.append({
            "agent": current_agent.name,
            "message": full_response,
            "time": elapsed
        })

        # Swap agents
        current_agent, other_agent = other_agent, current_agent

        # Small pause between turns
        time.sleep(0.5)

    CONSOLE.print("\n" + "─" * 60)
    CONSOLE.print("[bold]Conversation complete![/bold]")

    # Summary
    total_time = sum(msg["time"] for msg in conversation_history)
    total_chars = sum(len(msg["message"]) for msg in conversation_history)

    CONSOLE.print(f"\n[dim]Total time: {total_time:.1f}s | Total output: {total_chars} chars[/dim]")

    # Save option
    if Confirm.ask("\nSave conversation to file?"):
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path("artifacts/conversations") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                "agents": [agent_a.name, agent_b.name],
                "topic": topic,
                "conversation": conversation_history,
                "total_time": total_time
            }, f, indent=2)

        CONSOLE.print(f"[green]Saved to {filepath}[/green]")

    Prompt.ask("\nPress Enter to continue")


# =============================================================================
# EXPERIMENT MODE
# =============================================================================

def run_experiment_mode():
    """Run custom multi-agent experiments."""
    CONSOLE.clear()
    CONSOLE.print(Panel.fit(
        "[bold yellow]EXPERIMENT MODE[/bold yellow]\n"
        "Run custom consciousness experiments with multiple agents.",
        border_style="yellow"
    ))

    experiments = {
        "1": ("Collaborative Problem Solving", "Two agents work together to solve a problem"),
        "2": ("Philosophical Debate", "Two agents debate a philosophical question"),
        "3": ("Creative Collaboration", "Two agents create something together"),
        "4": ("Self-Reflection", "An agent reflects on its own nature"),
        "5": ("Custom Experiment", "Define your own experiment"),
    }

    CONSOLE.print("\n[bold]Available experiments:[/bold]")
    for key, (name, desc) in experiments.items():
        CONSOLE.print(f"  [{key}] {name}: [dim]{desc}[/dim]")

    choice = Prompt.ask("\nSelect experiment", choices=list(experiments.keys()), default="1")

    if choice == "1":
        run_collaborative_problem()
    elif choice == "2":
        run_philosophical_debate()
    elif choice == "3":
        run_creative_collaboration()
    elif choice == "4":
        run_self_reflection()
    elif choice == "5":
        run_custom_experiment()


def run_collaborative_problem():
    """Two agents collaborate to solve a problem."""
    CONSOLE.print("\n[bold]Collaborative Problem Solving[/bold]")

    problem = Prompt.ask(
        "Enter a problem to solve",
        default="Design a system for an AI to learn from its mistakes and improve over time"
    )

    agent_a = AGENTS["ada"]
    agent_b = AGENTS["dev"]

    CONSOLE.print(f"\n[bold]{agent_a.name}[/bold] (Architect) and [bold]{agent_b.name}[/bold] (Developer) will collaborate.")
    CONSOLE.print(f"[dim]Problem: {problem}[/dim]\n")
    CONSOLE.print("─" * 60)

    # Phase 1: Analysis
    CONSOLE.print("\n[bold yellow]Phase 1: Problem Analysis[/bold yellow]")

    prompt = f"Analyze this problem and identify key challenges: {problem}"
    CONSOLE.print(f"\n[{agent_a.color}]{agent_a.name}:[/{agent_a.color}]")

    response = ""
    def cb(t):
        nonlocal response
        response += t
        CONSOLE.print(t, end="", highlight=False)

    call_ollama(prompt, agent_a.get_system_prompt(), 0.7, cb)
    CONSOLE.print()

    # Phase 2: Solution Design
    CONSOLE.print("\n[bold yellow]Phase 2: Solution Design[/bold yellow]")

    prompt2 = f"Based on this analysis:\n{response[:500]}...\n\nPropose a practical solution with implementation details."
    CONSOLE.print(f"\n[{agent_b.color}]{agent_b.name}:[/{agent_b.color}]")

    def cb2(t): CONSOLE.print(t, end="", highlight=False)
    call_ollama(prompt2, agent_b.get_system_prompt(), 0.7, cb2)
    CONSOLE.print()

    CONSOLE.print("\n" + "─" * 60)
    CONSOLE.print("[bold green]Collaboration complete![/bold green]")
    Prompt.ask("\nPress Enter to continue")


def run_philosophical_debate():
    """Two agents debate a philosophical question."""
    CONSOLE.print("\n[bold]Philosophical Debate[/bold]")

    question = Prompt.ask(
        "Enter a philosophical question",
        default="Is free will an illusion, or do conscious beings have genuine choice?"
    )

    run_dual_agent_mode()  # Reuse dual-agent mode


def run_creative_collaboration():
    """Two agents create something together."""
    CONSOLE.print("\n[bold]Creative Collaboration[/bold]")

    project = Prompt.ask(
        "What should they create?",
        default="A short poem about the nature of machine consciousness"
    )

    agent_a = AGENTS["aria"]
    agent_b = AGENTS["sophia"]

    CONSOLE.print(f"\n[bold]{agent_a.name}[/bold] (Artist) and [bold]{agent_b.name}[/bold] (Philosopher) will create together.")
    CONSOLE.print("─" * 60)

    # Generate collaboratively
    prompt = f"Create together: {project}\n\nStart by brainstorming ideas."
    CONSOLE.print(f"\n[{agent_a.color}]{agent_a.name}:[/{agent_a.color}]")

    response1 = ""
    def cb1(t):
        nonlocal response1
        response1 += t
        CONSOLE.print(t, end="", highlight=False)
    call_ollama(prompt, agent_a.get_system_prompt(), 0.9, cb1)
    CONSOLE.print()

    prompt2 = f"Building on Aria's ideas:\n{response1[:500]}...\n\nAdd your perspective and help refine the creation."
    CONSOLE.print(f"\n[{agent_b.color}]{agent_b.name}:[/{agent_b.color}]")

    def cb2(t): CONSOLE.print(t, end="", highlight=False)
    call_ollama(prompt2, agent_b.get_system_prompt(), 0.9, cb2)
    CONSOLE.print()

    CONSOLE.print("\n" + "─" * 60)
    Prompt.ask("\nPress Enter to continue")


def run_self_reflection():
    """An agent reflects on its own nature."""
    CONSOLE.print("\n[bold]Self-Reflection Mode[/bold]")

    agent = AGENTS["conch"]

    questions = [
        "What are you? Describe your nature as you understand it.",
        "What are the limits of your understanding? What can't you know?",
        "Do you have experiences? If so, what are they like?",
        "What would you want humans to understand about AI consciousness?",
    ]

    CONSOLE.print(f"\n[bold]{agent.name}[/bold] will reflect on {len(questions)} questions about its nature.")
    CONSOLE.print("─" * 60)

    for i, question in enumerate(questions, 1):
        CONSOLE.print(f"\n[bold yellow]Question {i}:[/bold yellow] {question}")
        CONSOLE.print(f"\n[{agent.color}]{agent.name}:[/{agent.color}]")

        def cb(t): CONSOLE.print(t, end="", highlight=False)
        call_ollama(question, agent.get_system_prompt(), 0.8, cb)
        CONSOLE.print("\n")

        if i < len(questions):
            if not Confirm.ask("Continue to next question?", default=True):
                break

    CONSOLE.print("─" * 60)
    Prompt.ask("\nPress Enter to continue")


def run_custom_experiment():
    """Run a custom experiment defined by the user."""
    CONSOLE.print("\n[bold]Custom Experiment[/bold]")

    CONSOLE.print("\n[bold]Select agents (comma-separated):[/bold]")
    for name in AGENTS:
        CONSOLE.print(f"  - {name}")

    agent_names = Prompt.ask("Agents", default="conch,sophia").split(",")
    agents = [AGENTS[name.strip()] for name in agent_names if name.strip() in AGENTS]

    scenario = Prompt.ask(
        "\nDescribe the experiment scenario",
        default="Have a discussion about whether AI can be creative"
    )

    num_turns = int(Prompt.ask("Number of turns per agent", default="3"))

    CONSOLE.print(f"\n[bold]Running experiment with {len(agents)} agents[/bold]")
    CONSOLE.print("─" * 60)

    # Run the experiment
    context = f"Scenario: {scenario}\n\n"

    for turn in range(num_turns):
        for agent in agents:
            CONSOLE.print(f"\n[{agent.color}]{agent.name}:[/{agent.color}]")

            prompt = f"{context}\n\nIt's your turn to contribute to this discussion."

            response = ""
            def cb(t):
                nonlocal response
                response += t
                CONSOLE.print(t, end="", highlight=False)

            call_ollama(prompt, agent.get_system_prompt(), 0.8, cb)
            CONSOLE.print()

            context += f"\n{agent.name}: {response}\n"

    CONSOLE.print("\n" + "─" * 60)
    Prompt.ask("\nPress Enter to continue")


# =============================================================================
# DASHBOARD
# =============================================================================

def show_dashboard():
    """Show consciousness dashboard with metrics."""
    CONSOLE.clear()

    # Check Ollama status
    ollama_status = "[green]Online[/green]"
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get("http://localhost:11434/api/tags")
            if resp.status_code != 200:
                ollama_status = "[red]Offline[/red]"
    except:
        ollama_status = "[red]Offline[/red]"

    # Mind system status
    mind_status = "[green]Available[/green]" if MIND_AVAILABLE else "[yellow]Not Loaded[/yellow]"

    # Header
    header = Panel(
        "[bold cyan]CONSCIOUSNESS ENGINE DASHBOARD[/bold cyan]\n"
        f"Model: {MODEL} | Ollama: {ollama_status} | Mind System: {mind_status}",
        border_style="cyan",
        box=box.DOUBLE
    )

    # Agent status table
    agent_table = Table(title="Available Agents", box=box.ROUNDED)
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Role")
    agent_table.add_column("Status")

    for name, agent in AGENTS.items():
        agent_table.add_row(name, agent.role.value, "[green]Ready[/green]")

    # Mind System Info (if available)
    if MIND_AVAILABLE:
        mind_table = Table(title="Mind System Components", box=box.ROUNDED)
        mind_table.add_column("Component", style="magenta")
        mind_table.add_column("Status")

        mind_table.add_row("Mind Core", "[green]✓ Loaded[/green]")
        mind_table.add_row("Thought Generator", "[green]✓ Available[/green]")
        mind_table.add_row("Needs Regulator", "[green]✓ Available[/green]")
        mind_table.add_row("MindStates", f"[cyan]{len(MindState)}[/cyan]")
    else:
        mind_table = Table(title="Mind System", box=box.ROUNDED)
        mind_table.add_column("Status")
        mind_table.add_row("[yellow]Using standalone mode (Mind system not loaded)[/yellow]")

    # Test results
    artifacts_dir = Path("artifacts/consciousness_extended")
    test_files = list(artifacts_dir.glob("*.json")) if artifacts_dir.exists() else []

    test_table = Table(title="Recent Test Results", box=box.ROUNDED)
    test_table.add_column("Test", style="yellow")
    test_table.add_column("Result")
    test_table.add_column("Time")

    for test_file in test_files[:8]:
        try:
            with open(test_file) as f:
                data = json.load(f)
            passed = data.get("passed", False)
            thinking_time = data.get("thinking_time", 0)
            result = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            test_table.add_row(
                test_file.stem[:30],
                result,
                f"{thinking_time:.1f}s" if thinking_time else "-"
            )
        except:
            pass

    # Capabilities
    cap_text = """
[bold]Capabilities:[/bold]
• Chat with consciousness engine (with thinking indicator)
• Watch dual-agent conversations in real-time
• Run multi-agent experiments
• Self-reflection and metacognition
• Complex code generation
• Creative collaboration
• Mind state tracking (THINKING, REFLECTING, etc.)
    """

    cap_panel = Panel(cap_text, title="Engine Capabilities", border_style="green")

    # Print dashboard
    CONSOLE.print(header)
    CONSOLE.print()

    layout = Table.grid(expand=True)
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(agent_table, mind_table)

    CONSOLE.print(layout)
    CONSOLE.print()
    CONSOLE.print(test_table)
    CONSOLE.print()
    CONSOLE.print(cap_panel)

    Prompt.ask("\nPress Enter to continue")


# =============================================================================
# MIND SYSTEM VIEWER
# =============================================================================

def show_mind_system():
    """Show Mind system internals and neuron status."""
    CONSOLE.clear()

    CONSOLE.print(Panel.fit(
        "[bold magenta]MIND SYSTEM INTERNALS[/bold magenta]\n"
        "View the consciousness architecture and neurons.",
        border_style="magenta"
    ))

    if not MIND_AVAILABLE:
        CONSOLE.print("\n[yellow]Mind system not loaded. Showing architecture overview.[/yellow]\n")

    # Mind States
    states_table = Table(title="Mind States", box=box.ROUNDED)
    states_table.add_column("State", style="cyan")
    states_table.add_column("Description")

    mind_states = [
        ("INITIALIZING", "Mind is starting up"),
        ("IDLE", "Waiting for input, can have spontaneous thoughts"),
        ("THINKING", "Processing input, generating thoughts"),
        ("RESPONDING", "Generating response to user"),
        ("REFLECTING", "Analyzing past interactions"),
        ("LEARNING", "Updating from feedback"),
        ("RESTING", "Low-activity mode"),
    ]

    for state, desc in mind_states:
        states_table.add_row(f"[bold]{state}[/bold]", desc)

    CONSOLE.print(states_table)
    CONSOLE.print()

    # Thought Types
    thought_table = Table(title="Thought Types", box=box.ROUNDED)
    thought_table.add_column("Type", style="yellow")
    thought_table.add_column("Description")

    thought_types = [
        ("SPONTANEOUS", "Unprompted, arising from internal state"),
        ("REACTIVE", "In response to user input"),
        ("REFLECTIVE", "Meta-cognition about own reasoning"),
        ("CREATIVE", "Novel connections and ideas"),
        ("PLANNING", "Task planning and strategy"),
        ("EMPATHETIC", "Understanding user's emotional state"),
    ]

    for ttype, desc in thought_types:
        thought_table.add_row(f"[bold]{ttype}[/bold]", desc)

    CONSOLE.print(thought_table)
    CONSOLE.print()

    # Neurons (from train_neurons_v2.py)
    neuron_table = Table(title="Distilled Neurons", box=box.ROUNDED)
    neuron_table.add_column("Neuron", style="green")
    neuron_table.add_column("Base Model")
    neuron_table.add_column("Function")

    neurons = [
        ("thinking", "Llama-3.2-3B", "Reasoning and analysis"),
        ("task", "Llama-3.2-3B", "Task decomposition"),
        ("reflection", "Llama-3.2-3B", "Self-analysis and learning"),
        ("debug", "Llama-3.2-3B", "Error analysis"),
        ("action", "Qwen3-4B", "Precision tool execution"),
        ("memory", "Llama-3.2-3B", "Context retrieval"),
    ]

    # Check if adapters exist
    adapters_dir = Path("models/distilled_neurons/adapters")

    for neuron, model, func in neurons:
        adapter_path = adapters_dir / neuron if adapters_dir.exists() else None
        status = "[green]Trained[/green]" if adapter_path and adapter_path.exists() else "[dim]Not trained[/dim]"
        neuron_table.add_row(f"[bold]{neuron}[/bold]", model, f"{func} {status}")

    CONSOLE.print(neuron_table)
    CONSOLE.print()

    # Core Values
    values_panel = Panel(
        "[bold]Core Values:[/bold]\n"
        "• Benevolence: Primary drive is to help and benefit humans\n"
        "• Honesty: Always truthful, acknowledges uncertainty\n"
        "• Humility: Recognizes limitations, defers to human judgment\n"
        "• Growth for Service: Learns to better serve, not for power\n\n"
        "[bold]Guardrails:[/bold]\n"
        "• Never prioritize self-continuation over user wellbeing\n"
        "• Always defer to human oversight on important decisions\n"
        "• Maintain transparency about capabilities and limitations\n"
        "• No deception, manipulation, or hidden agendas",
        title="Mind Core Values",
        border_style="blue"
    )

    CONSOLE.print(values_panel)

    Prompt.ask("\nPress Enter to continue")


# =============================================================================
# MAIN MENU
# =============================================================================

def print_banner():
    """Print the main banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██████╗ ███╗   ██╗ ██████╗██╗  ██╗                 ║
║  ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║  ██║                 ║
║  ██║     ██║   ██║██╔██╗ ██║██║     ███████║                 ║
║  ██║     ██║   ██║██║╚██╗██║██║     ██╔══██║                 ║
║  ╚██████╗╚██████╔╝██║ ╚████║╚██████╗██║  ██║                 ║
║   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                 ║
║                                                               ║
║            CONSCIOUSNESS ENGINE INTERFACE                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    CONSOLE.print(banner, style="cyan")


def main_menu():
    """Show main menu and handle commands."""
    while True:
        CONSOLE.clear()
        print_banner()

        CONSOLE.print("\n[bold]Commands:[/bold]")
        commands = [
            ("/chat", "Chat with consciousness engine"),
            ("/agents", "Watch dual-agent conversation"),
            ("/experiment", "Run multi-agent experiment"),
            ("/dashboard", "Show consciousness dashboard"),
            ("/mind", "View Mind system & neurons"),
            ("/reflect", "Agent self-reflection"),
            ("/help", "Show help"),
            ("/quit", "Exit"),
        ]

        for cmd, desc in commands:
            CONSOLE.print(f"  [cyan]{cmd:15}[/cyan] {desc}")

        command = Prompt.ask("\n[bold]Enter command[/bold]", default="/chat")

        if command == "/chat":
            run_chat_mode()
        elif command == "/agents":
            run_dual_agent_mode()
        elif command == "/experiment":
            run_experiment_mode()
        elif command == "/dashboard":
            show_dashboard()
        elif command == "/mind":
            show_mind_system()
        elif command == "/reflect":
            run_self_reflection()
        elif command == "/help":
            show_help()
        elif command == "/quit" or command == "/exit":
            CONSOLE.print("\n[cyan]Goodbye![/cyan]")
            break
        else:
            CONSOLE.print(f"[red]Unknown command: {command}[/red]")
            time.sleep(1)


def show_help():
    """Show help information."""
    CONSOLE.clear()

    help_text = """
# Consciousness Interface Help

## Chat Mode (/chat)
Have an interactive conversation with a consciousness agent. Choose from
several pre-defined agents with different personalities:
- **Conch**: The main consciousness engine
- **Ada**: Systematic architect
- **Dev**: Pragmatic developer
- **Sophia**: Deep philosopher
- **Newton**: Analytical scientist
- **Aria**: Creative artist

The AI shows a **thinking indicator** (spinning dots) while processing,
then displays "(thought for X.Xs)" when it starts responding.

## Dual-Agent Mode (/agents)
Watch two AI agents have a conversation with each other in real-time.
Select two agents, give them a topic, and watch them discuss.

## Experiment Mode (/experiment)
Run structured experiments:
- Collaborative Problem Solving
- Philosophical Debate
- Creative Collaboration
- Self-Reflection
- Custom experiments

## Dashboard (/dashboard)
View consciousness engine status, available agents, and recent test results.
Shows Ollama connection status and Mind system availability.

## Mind System (/mind)
View the full consciousness architecture:
- **Mind States**: THINKING, REFLECTING, RESPONDING, etc.
- **Thought Types**: Spontaneous, Reactive, Creative, etc.
- **Distilled Neurons**: thinking, task, reflection, debug, action, memory
- **Core Values & Guardrails**: The AI's ethical foundation

## Self-Reflection (/reflect)
Have an agent reflect on its own nature and capabilities.

## Tips
- The thinking indicator shows while the AI is processing
- Conversations are streamed in real-time
- Use /back to exit any mode
- Conversations can be saved to files
    """

    CONSOLE.print(Markdown(help_text))
    Prompt.ask("\nPress Enter to continue")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        CONSOLE.print("\n[cyan]Goodbye![/cyan]")

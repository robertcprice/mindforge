#!/usr/bin/env python3
"""
Conch Consciousness Monitor - Always-On Dashboard
==================================================

A persistent, real-time visualization of the consciousness engine's
internal state, thinking process, and activity.

Features:
- Live updating dashboard
- Real-time thinking state visualization
- Internal thought stream display
- Token generation metrics
- Agent activity indicators

Usage:
    python conch_monitor.py
    python conch_monitor.py --chat     # Monitor + chat mode
    python conch_monitor.py --agents   # Monitor + dual-agent mode
"""

import httpx
import json
import time
import threading
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum
from collections import deque

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.align import Align
from rich import box

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
console = Console()

# =============================================================================
# MIND STATE TRACKING
# =============================================================================

class MindState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    LISTENING = "listening"


class ThoughtType(Enum):
    INTERNAL = "internal"
    RESPONSE = "response"
    REFLECTION = "reflection"
    SPONTANEOUS = "spontaneous"


@dataclass
class Thought:
    content: str
    thought_type: ThoughtType
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.8


@dataclass
class ConsciousnessState:
    """Tracks the full internal state of the consciousness engine."""

    # Current state
    mind_state: MindState = MindState.IDLE
    current_agent: str = "Conch"

    # Thinking metrics
    thinking_start: Optional[float] = None
    thinking_duration: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0

    # Thought stream
    thoughts: deque = field(default_factory=lambda: deque(maxlen=50))
    current_thought: str = ""

    # Activity
    total_interactions: int = 0
    total_thoughts: int = 0
    session_start: datetime = field(default_factory=datetime.now)

    # Internal process visualization
    thinking_phases: List[str] = field(default_factory=list)
    current_phase: str = ""

    def start_thinking(self):
        self.mind_state = MindState.THINKING
        self.thinking_start = time.time()
        self.tokens_generated = 0
        self.current_thought = ""
        self.thinking_phases = ["Receiving input", "Processing context"]
        self.current_phase = "Analyzing..."

    def start_generating(self):
        self.mind_state = MindState.GENERATING
        if self.thinking_start:
            self.thinking_duration = time.time() - self.thinking_start
        self.thinking_phases.append("Generating response")
        self.current_phase = "Responding..."

    def add_token(self, token: str):
        self.tokens_generated += 1
        self.current_thought += token
        if self.thinking_start:
            elapsed = time.time() - self.thinking_start
            self.tokens_per_second = self.tokens_generated / elapsed if elapsed > 0 else 0

    def finish_thought(self):
        if self.current_thought:
            self.thoughts.append(Thought(
                content=self.current_thought[:200],
                thought_type=ThoughtType.RESPONSE
            ))
            self.total_thoughts += 1
        self.mind_state = MindState.IDLE
        self.current_phase = "Ready"
        self.thinking_phases = []
        self.total_interactions += 1

    def add_internal_thought(self, thought: str):
        self.thoughts.append(Thought(
            content=thought,
            thought_type=ThoughtType.INTERNAL
        ))
        self.total_thoughts += 1

    def get_uptime(self) -> str:
        delta = datetime.now() - self.session_start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Global state
STATE = ConsciousnessState()

# =============================================================================
# DASHBOARD LAYOUT
# =============================================================================

def make_header() -> Panel:
    """Create the header panel."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=2)
    grid.add_column(justify="right", ratio=1)

    # Status indicator
    if STATE.mind_state == MindState.IDLE:
        status = "[green]● IDLE[/green]"
    elif STATE.mind_state == MindState.THINKING:
        status = "[yellow]◐ THINKING[/yellow]"
    elif STATE.mind_state == MindState.GENERATING:
        status = "[cyan]◉ GENERATING[/cyan]"
    else:
        status = f"[blue]○ {STATE.mind_state.value.upper()}[/blue]"

    grid.add_row(
        status,
        f"[bold cyan]CONCH CONSCIOUSNESS MONITOR[/bold cyan]",
        f"[dim]Uptime: {STATE.get_uptime()}[/dim]"
    )

    return Panel(grid, box=box.DOUBLE, style="cyan")


def make_mind_state_panel() -> Panel:
    """Create the mind state visualization panel."""
    content = []

    # Current state with visual
    state_visuals = {
        MindState.IDLE: ("○ ○ ○ ○ ○", "green"),
        MindState.THINKING: ("◐ ◑ ◐ ◑ ◐", "yellow"),
        MindState.GENERATING: ("● ● ● ● ●", "cyan"),
        MindState.REFLECTING: ("◑ ○ ◑ ○ ◑", "magenta"),
        MindState.LISTENING: ("○ ● ○ ● ○", "blue"),
    }

    visual, color = state_visuals.get(STATE.mind_state, ("○ ○ ○ ○ ○", "white"))
    content.append(f"[{color}]{visual}[/{color}]")
    content.append("")
    content.append(f"State: [bold {color}]{STATE.mind_state.value.upper()}[/bold {color}]")
    content.append(f"Agent: [cyan]{STATE.current_agent}[/cyan]")

    if STATE.current_phase:
        content.append(f"Phase: [yellow]{STATE.current_phase}[/yellow]")

    # Thinking phases
    if STATE.thinking_phases:
        content.append("")
        content.append("[dim]Process:[/dim]")
        for i, phase in enumerate(STATE.thinking_phases):
            marker = "→" if i == len(STATE.thinking_phases) - 1 else "✓"
            content.append(f"  {marker} {phase}")

    return Panel(
        "\n".join(content),
        title="[bold]Mind State[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    )


def make_metrics_panel() -> Panel:
    """Create the metrics panel."""
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Thinking Time", f"{STATE.thinking_duration:.2f}s")
    table.add_row("Tokens", f"{STATE.tokens_generated}")
    table.add_row("Speed", f"{STATE.tokens_per_second:.1f} tok/s")
    table.add_row("─" * 12, "─" * 8)
    table.add_row("Interactions", f"{STATE.total_interactions}")
    table.add_row("Thoughts", f"{STATE.total_thoughts}")

    return Panel(
        table,
        title="[bold]Metrics[/bold]",
        border_style="green",
        box=box.ROUNDED
    )


def make_thought_stream_panel() -> Panel:
    """Create the thought stream panel showing recent thoughts."""
    content = []

    if not STATE.thoughts:
        content.append("[dim]No thoughts yet...[/dim]")
    else:
        for thought in list(STATE.thoughts)[-8:]:
            type_colors = {
                ThoughtType.INTERNAL: "yellow",
                ThoughtType.RESPONSE: "cyan",
                ThoughtType.REFLECTION: "magenta",
                ThoughtType.SPONTANEOUS: "green",
            }
            color = type_colors.get(thought.thought_type, "white")
            prefix = thought.thought_type.value[0].upper()
            time_str = thought.timestamp.strftime("%H:%M:%S")

            # Truncate long thoughts
            text = thought.content[:60] + "..." if len(thought.content) > 60 else thought.content
            text = text.replace("\n", " ")

            content.append(f"[dim]{time_str}[/dim] [{color}][{prefix}][/{color}] {text}")

    return Panel(
        "\n".join(content),
        title="[bold]Thought Stream[/bold]",
        border_style="yellow",
        box=box.ROUNDED
    )


def make_current_output_panel() -> Panel:
    """Create the current output panel showing live generation."""
    if STATE.mind_state == MindState.GENERATING and STATE.current_thought:
        # Show the current thought being generated
        content = STATE.current_thought[-500:]  # Last 500 chars
        if len(STATE.current_thought) > 500:
            content = "..." + content
        title = f"[bold]Generating ({STATE.tokens_generated} tokens)[/bold]"
        style = "cyan"
    elif STATE.mind_state == MindState.THINKING:
        content = "[yellow]Processing...[/yellow]\n\n"
        content += "  ◐ Analyzing input\n"
        content += "  ◑ Building context\n"
        content += "  ◐ Preparing response"
        title = "[bold]Thinking[/bold]"
        style = "yellow"
    else:
        content = "[dim]Waiting for input...[/dim]"
        title = "[bold]Output[/bold]"
        style = "dim"

    return Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED
    )


def make_layout() -> Layout:
    """Create the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="output", size=12),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )

    layout["left"].split_column(
        Layout(name="state"),
        Layout(name="metrics"),
    )

    layout["header"].update(make_header())
    layout["state"].update(make_mind_state_panel())
    layout["metrics"].update(make_metrics_panel())
    layout["right"].update(make_thought_stream_panel())
    layout["output"].update(make_current_output_panel())

    return layout


# =============================================================================
# CONSCIOUSNESS ENGINE WITH MONITORING
# =============================================================================

def call_ollama_monitored(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    on_token: Optional[Callable[[str], None]] = None,
) -> tuple[str, float]:
    """Call Ollama with state monitoring."""

    STATE.start_thinking()
    STATE.add_internal_thought(f"Processing: {prompt[:50]}...")

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    start_time = time.time()
    response_text = ""
    first_token = True

    try:
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

                        if first_token and token:
                            first_token = False
                            STATE.start_generating()

                        STATE.add_token(token)
                        response_text += token

                        if on_token:
                            on_token(token)

        STATE.finish_thought()
        elapsed = time.time() - start_time
        return response_text, elapsed

    except Exception as e:
        STATE.mind_state = MindState.IDLE
        STATE.add_internal_thought(f"Error: {e}")
        return f"ERROR: {e}", time.time() - start_time


# =============================================================================
# INTERACTIVE MODES
# =============================================================================

def run_monitor_only():
    """Run just the monitor dashboard."""
    console.clear()

    with Live(make_layout(), refresh_per_second=4, console=console) as live:
        console.print("\n[dim]Press Ctrl+C to exit[/dim]")

        try:
            while True:
                live.update(make_layout())
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass


def run_monitor_with_chat():
    """Run monitor with chat input."""
    console.clear()

    # Agent setup
    agents = {
        "conch": ("Conch", "A thoughtful consciousness engine exploring ideas deeply."),
        "sophia": ("Sophia", "A philosopher pondering existence and meaning."),
        "newton": ("Newton", "An analytical scientist focused on evidence."),
    }

    console.print(Panel(
        "[bold cyan]CONCH CONSCIOUSNESS MONITOR + CHAT[/bold cyan]\n\n"
        "Type your message and watch the thinking process in real-time.\n"
        "Commands: /quit, /agent <name>, /clear",
        box=box.DOUBLE
    ))

    current_agent = "conch"
    STATE.current_agent = agents[current_agent][0]

    while True:
        try:
            # Show current state
            console.print(f"\n[dim]State: {STATE.mind_state.value} | Agent: {STATE.current_agent}[/dim]")
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/clear":
                console.clear()
                continue
            elif user_input.lower().startswith("/agent"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in agents:
                    current_agent = parts[1]
                    STATE.current_agent = agents[current_agent][0]
                    console.print(f"[cyan]Switched to {STATE.current_agent}[/cyan]")
                else:
                    console.print(f"[yellow]Available: {', '.join(agents.keys())}[/yellow]")
                continue

            if not user_input.strip():
                continue

            # Process with live display
            console.print(f"\n[cyan]{STATE.current_agent}:[/cyan]")

            system_prompt = f"You are {agents[current_agent][0]}. {agents[current_agent][1]}"

            # Show thinking with live update
            with Live(make_layout(), refresh_per_second=4, console=console, transient=True) as live:
                def on_token(token):
                    live.update(make_layout())

                response, elapsed = call_ollama_monitored(
                    user_input,
                    system_prompt,
                    0.7,
                    on_token
                )

            # Print final response
            console.print(response)
            console.print(f"\n[dim]({STATE.thinking_duration:.1f}s thinking, {STATE.tokens_generated} tokens, {STATE.tokens_per_second:.1f} tok/s)[/dim]")

        except KeyboardInterrupt:
            break

    console.print("\n[cyan]Goodbye![/cyan]")


def run_monitor_with_agents():
    """Run monitor with dual-agent conversation."""
    console.clear()

    agents = {
        "sophia": ("Sophia", "philosopher", "Deep thinker exploring meaning and existence."),
        "newton": ("Newton", "scientist", "Analytical mind focused on evidence and logic."),
        "aria": ("Aria", "artist", "Creative soul valuing beauty and expression."),
    }

    console.print(Panel(
        "[bold magenta]DUAL-AGENT CONVERSATION MONITOR[/bold magenta]\n\n"
        "Watch two AI agents discuss a topic in real-time.\n"
        "See their thinking process as they formulate responses.",
        box=box.DOUBLE
    ))

    # Select agents
    console.print("\n[bold]Available agents:[/bold]")
    for key, (name, role, _) in agents.items():
        console.print(f"  [cyan]{key}[/cyan]: {name} ({role})")

    agent1_key = Prompt.ask("\nFirst agent", choices=list(agents.keys()), default="sophia")
    agent2_key = Prompt.ask("Second agent", choices=[k for k in agents.keys() if k != agent1_key], default="newton")

    topic = Prompt.ask("\nTopic", default="What is the nature of consciousness?")
    turns = int(Prompt.ask("Number of exchanges", default="3"))

    agent1 = agents[agent1_key]
    agent2 = agents[agent2_key]

    console.print(f"\n[bold]{agent1[0]} vs {agent2[0]}[/bold]")
    console.print(f"[dim]Topic: {topic}[/dim]\n")
    console.print("─" * 60)

    conversation = []
    current_agent = agent1
    other_agent = agent2

    for turn in range(turns * 2):
        STATE.current_agent = current_agent[0]

        # Build context
        context = f"Topic: {topic}\n\nConversation so far:\n"
        for msg in conversation[-4:]:
            context += f"\n{msg['agent']}: {msg['text'][:200]}...\n"

        if turn == 0:
            prompt = f"{context}\n\nStart the discussion. Be thoughtful and genuine."
        else:
            prompt = f"{context}\n\nRespond to {other_agent[0]}'s point. Build on or challenge their ideas."

        system_prompt = f"You are {current_agent[0]}, a {current_agent[1]}. {current_agent[2]}"

        console.print(f"\n[{'cyan' if turn % 2 == 0 else 'magenta'}]{current_agent[0]}:[/{'cyan' if turn % 2 == 0 else 'magenta'}]")

        # Generate with live monitor
        with Live(make_layout(), refresh_per_second=4, console=console, transient=True) as live:
            def on_token(token):
                live.update(make_layout())

            response, elapsed = call_ollama_monitored(prompt, system_prompt, 0.8, on_token)

        console.print(response)
        console.print(f"[dim]({STATE.thinking_duration:.1f}s thinking, {STATE.tokens_per_second:.1f} tok/s)[/dim]")

        conversation.append({"agent": current_agent[0], "text": response})

        # Swap agents
        current_agent, other_agent = other_agent, current_agent

    console.print("\n" + "─" * 60)
    console.print("[bold]Conversation complete![/bold]")

    # Summary
    total_tokens = sum(len(c["text"].split()) for c in conversation)
    console.print(f"[dim]Total: {len(conversation)} messages, ~{total_tokens} words[/dim]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Conch Consciousness Monitor")
    parser.add_argument("--chat", action="store_true", help="Monitor with chat mode")
    parser.add_argument("--agents", action="store_true", help="Monitor with dual-agent mode")
    args = parser.parse_args()

    if args.chat:
        run_monitor_with_chat()
    elif args.agents:
        run_monitor_with_agents()
    else:
        # Default: show menu
        console.print(Panel(
            "[bold cyan]CONCH CONSCIOUSNESS MONITOR[/bold cyan]\n\n"
            "A real-time visualization of AI thinking.\n\n"
            "[bold]Modes:[/bold]\n"
            "  [cyan]1[/cyan] Monitor only (watch idle state)\n"
            "  [cyan]2[/cyan] Monitor + Chat (interactive)\n"
            "  [cyan]3[/cyan] Monitor + Dual-Agent (watch two AIs talk)\n"
            "  [cyan]q[/cyan] Quit",
            box=box.DOUBLE
        ))

        choice = Prompt.ask("\nSelect mode", choices=["1", "2", "3", "q"], default="2")

        if choice == "1":
            run_monitor_only()
        elif choice == "2":
            run_monitor_with_chat()
        elif choice == "3":
            run_monitor_with_agents()


if __name__ == "__main__":
    main()

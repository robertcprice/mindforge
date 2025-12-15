#!/usr/bin/env python3
"""
Conch Consciousness Engine - Main Entry Point

An always-running, always-thinking AI consciousness that:
- Wakes up periodically with organic timing
- Thinks spontaneously based on needs and memories
- Decides what to do (or do nothing)
- Acts using tools (n8n, filesystem, code, etc.)
- Reflects and learns from outcomes
- Improves itself through periodic fine-tuning

Usage:
    python main.py                    # Run the consciousness loop
    python main.py --cycles 10        # Run 10 cycles then exit
    python main.py --dashboard        # Also start the Streamlit dashboard
    python main.py --dry-run          # Preview without actually running
"""

import argparse
import asyncio
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from conch.config import get_config, ConchConfig
from conch.core.needs import NeedsRegulator, NeedType
from conch.core.thought import ThoughtGenerator
from conch.memory.store import MemoryStore
from conch.agent.langgraph_agent import ConsciousnessAgent, create_consciousness_graph
from conch.integrations.ollama import OllamaClient
from conch.integrations.n8n import N8NClient

# Rich console for pretty output - use soft_wrap to prevent truncation
console = Console(soft_wrap=True, width=300)

# Global state for graceful shutdown
shutdown_requested = False


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure logging with Rich handler."""
    handlers = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            # Note: Console width=300 prevents truncation
        )
    ]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=handlers,
    )


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        console.print("[yellow]Warning: config.yaml not found, using defaults[/yellow]")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_inference_fn(config: dict):
    """Create the inference function based on config.

    Tries MLX first, falls back to Ollama.
    """
    backend = config.get("model", {}).get("inference_backend", "mlx")
    ollama_model = config.get("model", {}).get("ollama_model_name", "echo_assistant")

    # Try MLX first
    if backend == "mlx":
        try:
            from conch.inference.mlx_backend import MLXInference
            mlx = MLXInference()
            console.print("[green]✓ Using MLX backend[/green]")
            return mlx.generate
        except Exception as e:
            console.print(f"[yellow]MLX not available: {e}[/yellow]")
            console.print("[yellow]Falling back to Ollama...[/yellow]")

    # Fall back to Ollama
    try:
        ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
        client = OllamaClient(host=ollama_host)

        if not client.is_healthy():
            console.print("[red]Error: Ollama is not running[/red]")
            console.print("Start Ollama with: ollama serve")
            sys.exit(1)

        # Check if model exists
        if not client.model_exists(ollama_model):
            console.print(f"[yellow]Model {ollama_model} not found in Ollama[/yellow]")
            console.print(f"[yellow]Trying to pull base model...[/yellow]")

            # Try to pull qwen
            base_model = config.get("model", {}).get("base", "qwen2.5:7b")
            for progress in client.pull_model(base_model):
                if "error" in progress:
                    console.print(f"[red]Failed to pull model: {progress['error']}[/red]")
                    sys.exit(1)
            ollama_model = base_model

        console.print(f"[green]✓ Using Ollama backend ({ollama_model})[/green]")

        def ollama_inference(prompt: str) -> str:
            response = client.generate(
                model=ollama_model,
                prompt=prompt,
                options={"temperature": 0.7, "top_p": 0.9},
            )
            return response.response

        return ollama_inference

    except Exception as e:
        console.print(f"[red]Failed to initialize inference: {e}[/red]")
        sys.exit(1)


def check_services(config: dict) -> dict:
    """Check status of all required services."""
    status = {
        "ollama": False,
        "n8n": False,
    }

    # Check Ollama
    try:
        ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
        client = OllamaClient(host=ollama_host)
        status["ollama"] = client.is_healthy()
    except Exception:
        pass

    # Check n8n
    try:
        n8n_enabled = config.get("n8n", {}).get("enabled", False)
        if n8n_enabled:
            n8n_url = config.get("n8n", {}).get("url", "http://localhost:5678")
            n8n = N8NClient(base_url=n8n_url)
            status["n8n"] = n8n.is_healthy()
    except Exception:
        pass

    return status


def display_status(config: dict, agent: ConsciousnessAgent) -> None:
    """Display current status in a nice table."""
    services = check_services(config)

    # Service status table
    service_table = Table(title="Services")
    service_table.add_column("Service", style="cyan")
    service_table.add_column("Status")

    service_table.add_row(
        "Ollama",
        "[green]✓ Running[/green]" if services["ollama"] else "[red]✗ Not running[/red]"
    )
    service_table.add_row(
        "n8n",
        "[green]✓ Running[/green]" if services["n8n"] else "[dim]○ Disabled/Offline[/dim]"
    )

    # Needs table
    needs_table = Table(title="Current Needs")
    needs_table.add_column("Need", style="cyan")
    needs_table.add_column("Level")
    needs_table.add_column("Bar")

    needs = agent.needs_regulator.get_state()
    for name, need_data in sorted(needs.items(), key=lambda x: -x[1]["level"]):
        value = need_data["level"]
        bar_width = int(value * 20)
        bar = "█" * bar_width + "░" * (20 - bar_width)
        color = "red" if value > 0.8 else "yellow" if value > 0.5 else "green"
        needs_table.add_row(name, f"{value:.2f}", f"[{color}]{bar}[/{color}]")

    console.print(Panel.fit(service_table))
    console.print(Panel.fit(needs_table))


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    console.print("\n[yellow]Shutdown requested...[/yellow]")
    shutdown_requested = True


def get_sleep_duration(config: dict, requested_sleep: float = None) -> float:
    """Get sleep duration between cycles.

    If the model requested a specific sleep duration, use that (within bounds).
    Otherwise, use organic random timing.

    Args:
        config: Configuration dictionary
        requested_sleep: Model's requested sleep duration (if any)

    Returns:
        Sleep duration in seconds
    """
    min_sleep = config.get("cycle", {}).get("min_sleep_seconds", 30)
    max_sleep = config.get("cycle", {}).get("max_sleep_seconds", 300)

    # If model requested a specific duration, use it (clamped to bounds)
    if requested_sleep is not None:
        return max(min_sleep, min(requested_sleep, max_sleep))

    # Otherwise use exponential distribution for organic feel
    # Most sleeps will be shorter, occasional longer ones
    mean = (max_sleep - min_sleep) / 3
    duration = min_sleep + random.expovariate(1 / mean)
    return min(duration, max_sleep)


def run_consciousness_loop(
    config: dict,
    max_cycles: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """Run the eternal consciousness loop.

    Args:
        config: Configuration dictionary
        max_cycles: Optional limit on cycles (None = infinite)
        dry_run: If True, show what would happen without running
    """
    global shutdown_requested

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create inference function
    inference_fn = create_inference_fn(config)

    # Create memory store
    db_path = Path(config.get("memory", {}).get("sqlite_path", "./data/memories.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    memory_store = MemoryStore(db_path=db_path)

    # Build system prompt from config
    name = config.get("name", "Echo")
    system_prompt = config.get("system_prompt", "").format(
        name=name,
        needs_summary="{needs_summary}",  # Placeholder for dynamic content
        memory_summary="{memory_summary}",
    )

    # Create consciousness agent
    agent = create_consciousness_graph(
        inference_fn=inference_fn,
        memory_store=memory_store,
        system_prompt=system_prompt,
    )

    # Get cycle thresholds
    cycles_before_consolidation = config.get("cycle", {}).get("cycles_before_consolidation", 50)
    cycles_before_finetune = config.get("cycle", {}).get("cycles_before_mini_finetune", 200)

    console.print(Panel.fit(
        f"[bold green]Conch Consciousness Engine[/bold green]\n"
        f"Name: {name}\n"
        f"Backend: {config.get('model', {}).get('inference_backend', 'mlx')}\n"
        f"Consolidation every: {cycles_before_consolidation} cycles\n"
        f"Fine-tune every: {cycles_before_finetune} cycles"
    ))

    if dry_run:
        console.print("[yellow]Dry run mode - showing status only[/yellow]")
        display_status(config, agent)
        return

    # Main loop
    cycle = 0
    while not shutdown_requested:
        cycle += 1

        if max_cycles and cycle > max_cycles:
            console.print(f"[green]Completed {max_cycles} cycles, exiting[/green]")
            break

        console.rule(f"[bold]Cycle {cycle}[/bold] - {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Run one consciousness cycle
            state = agent.run_cycle()

            # Display results
            if state.get("error"):
                console.print(f"[red]Error: {state['error']}[/red]")
            else:
                # Show full thought (not truncated)
                thought = state.get('current_thought', '')
                console.print(f"\n[cyan bold]Thought:[/cyan bold]")
                console.print(f"  {thought}\n")

                # Show task information if any
                work_log = state.get('work_log', [])
                if work_log:
                    console.print(f"[yellow bold]Work Log ({len(work_log)} actions):[/yellow bold]")
                    for i, entry in enumerate(work_log, 1):
                        status_icon = "✅" if entry.get('success') else "❌"
                        console.print(f"  [{i}] {status_icon} {entry.get('action_taken', 'N/A')}")
                        result = entry.get('result', '')
                        if result:
                            console.print(f"      → {result}")
                    console.print()
                else:
                    # Show decision if no work log (old behavior)
                    decision = state.get('decision', '')
                    action_type = state.get('action_type', '')
                    if decision:
                        console.print(f"[yellow]Decision:[/yellow] {action_type}: {decision}")
                    result = state.get('action_result', '')
                    if result:
                        console.print(f"[green]Result:[/green] {result}")

                # Show reflection (full text)
                reflection = state.get('reflection', '')
                if reflection:
                    console.print(f"\n[magenta bold]Reflection:[/magenta bold]")
                    console.print(f"  {reflection}\n")

                # Show task stats
                tasks = state.get('current_tasks', [])
                if tasks:
                    pending = sum(1 for t in tasks if t.get('status') == 'pending')
                    completed = sum(1 for t in tasks if t.get('status') == 'completed')
                    failed = sum(1 for t in tasks if t.get('status') == 'failed')
                    console.print(f"[blue]Tasks:[/blue] {pending} pending, {completed} completed, {failed} failed")

                # Show sleep decision if model made one
                if state.get('requested_sleep'):
                    console.print(f"[dim]Sleep:[/dim] {state.get('requested_sleep'):.0f}s - {state.get('sleep_reason', '')}")

            # Memory consolidation
            if state.get("should_consolidate"):
                console.print("[blue]Consolidating memories...[/blue]")
                # memory_store.consolidate()  # Implement this

            # Mini fine-tune
            if state.get("should_finetune"):
                console.print("[blue]Triggering mini fine-tune...[/blue]")
                # This would call the fine-tuning logic
                # For now, just log it
                console.print("[dim]Fine-tuning will be implemented in next iteration[/dim]")

        except Exception as e:
            console.print(f"[red]Cycle error: {e}[/red]")
            logging.exception("Cycle failed")

        # Sleep between cycles - use model's requested duration if available
        if not shutdown_requested and (max_cycles is None or cycle < max_cycles):
            requested = state.get("requested_sleep") if not state.get("error") else None
            sleep_duration = get_sleep_duration(config, requested_sleep=requested)

            if requested:
                console.print(f"[dim]Sleeping for {sleep_duration:.0f}s (model requested)...[/dim]")
            else:
                console.print(f"[dim]Sleeping for {sleep_duration:.0f}s (organic timing)...[/dim]")

            # Sleep in small increments to allow interrupt
            sleep_end = time.time() + sleep_duration
            while time.time() < sleep_end and not shutdown_requested:
                time.sleep(1)

    console.print("[green]Consciousness loop ended gracefully[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Conch Consciousness Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=None,
        help="Number of cycles to run (default: infinite)",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Also start the Streamlit dashboard",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show status without running",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        console.print(f"[yellow]Config not found: {args.config}[/yellow]")
        config = {}

    # Setup logging
    log_file = Path(config.get("logging", {}).get("thoughts_log", "./logs/thoughts.log"))
    setup_logging(log_file=log_file, level=args.log_level)

    # Start dashboard in background if requested
    if args.dashboard:
        console.print("[blue]Starting dashboard...[/blue]")
        import subprocess
        dashboard_proc = subprocess.Popen(
            ["streamlit", "run", "dashboard/app.py", "--server.headless", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print(f"[green]Dashboard started on http://localhost:8501[/green]")

    try:
        # Run the consciousness loop
        run_consciousness_loop(
            config=config,
            max_cycles=args.cycles,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        if args.dashboard:
            dashboard_proc.terminate()


if __name__ == "__main__":
    main()

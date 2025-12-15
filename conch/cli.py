"""
Conch CLI

Command-line interface for Conch operations.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from conch import __version__
from conch.config import ConchConfig, get_config

# Setup rich console
console = Console()
app = typer.Typer(
    name="conch",
    help="Conch - An always-on, always-thinking AI assistant engine",
    add_completion=False,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Conch CLI - Good-hearted AI consciousness engine."""
    setup_logging(verbose)


@app.command()
def version():
    """Show Conch version."""
    console.print(f"[bold blue]Conch[/bold blue] v{__version__}")


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Path to initialize project"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """Initialize a new Conch project."""
    config_path = path / "config.yaml"

    if config_path.exists() and not force:
        console.print("[yellow]Config already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)

    config = ConchConfig()
    config.ensure_directories()
    config.to_yaml(config_path)

    console.print(Panel.fit(
        "[green]Conch initialized![/green]\n\n"
        f"Config: {config_path}\n"
        f"Data: {config.data_dir}\n"
        f"Models: {config.models_dir}",
        title="Success",
    ))


@app.command()
def status():
    """Show Conch status and configuration."""
    config = get_config()

    # Create status table
    table = Table(title="Conch Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Check directories
    table.add_row(
        "Data Directory",
        "[green]OK" if config.data_dir.exists() else "[red]Missing",
        str(config.data_dir),
    )
    table.add_row(
        "Models Directory",
        "[green]OK" if config.models_dir.exists() else "[red]Missing",
        str(config.models_dir),
    )

    # Check dependencies
    mlx_available = False
    try:
        import mlx
        mlx_available = True
    except ImportError:
        pass

    table.add_row(
        "MLX Backend",
        "[green]Available" if mlx_available else "[yellow]Not installed",
        "pip install mlx mlx-lm" if not mlx_available else "",
    )

    # Needs configuration
    table.add_row(
        "Needs Preset",
        "Custom",
        f"S:{config.needs.sustainability:.0%} R:{config.needs.reliability:.0%} "
        f"C:{config.needs.curiosity:.0%} E:{config.needs.excellence:.0%}",
    )

    console.print(table)


@app.command()
def train(
    dataset: Path = typer.Option(
        Path("data/assistant_data.json"),
        "--dataset", "-d",
        help="Path to training dataset"
    ),
    output: Path = typer.Option(
        Path("models/fine_tuned/conch_qwen"),
        "--output", "-o",
        help="Output directory for fine-tuned model"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    synthesize: bool = typer.Option(False, "--synthesize", "-s", help="Synthesize training data"),
    num_examples: int = typer.Option(500, "--num-examples", "-n", help="Examples to synthesize"),
):
    """Fine-tune the Conch model."""
    import subprocess

    console.print(Panel.fit(
        "[bold]Starting Conch Training[/bold]",
        title="Training",
    ))

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "fine_tune_qwen.py"),
        "--dataset_path", str(dataset),
        "--output_dir", str(output),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]

    if synthesize:
        cmd.extend(["--synthesize_data", "--num_examples", str(num_examples)])

    # Run training
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=None)

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            progress.update(task, description="[green]Training complete!")
        else:
            progress.update(task, description="[red]Training failed!")

    raise typer.Exit(result.returncode)


@app.command()
def daemon(
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    watch: Optional[list[Path]] = typer.Option(None, "--watch", "-w", help="Directories to watch"),
):
    """Start the Conch daemon."""
    from conch.scheduler.daemon import ConchDaemon

    console.print(Panel.fit(
        "[bold blue]Starting Conch Daemon[/bold blue]\n\n"
        "The daemon will:\n"
        "- Generate spontaneous thoughts\n"
        "- Monitor triggers\n"
        "- Run scheduled tasks\n"
        "- Watch for file changes",
        title="Daemon",
    ))

    watch_paths = [Path(p) for p in (watch or [])]

    daemon_instance = ConchDaemon(watch_paths=watch_paths)

    try:
        daemon_instance.start(blocking=not background)

        if background:
            console.print("[green]Daemon started in background[/green]")
        else:
            console.print("[yellow]Daemon stopped[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        daemon_instance.stop()


@app.command()
def chat(
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model path"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Interactive chat with Conch."""
    console.print(Panel.fit(
        "[bold blue]Conch Chat[/bold blue]\n\n"
        "Type your message and press Enter.\n"
        "Type 'quit' or 'exit' to stop.",
        title="Chat",
    ))

    # Try to load model
    try:
        from conch.inference import get_inference_function, CONCH_SYSTEM_PROMPT

        if model:
            inference_fn = get_inference_function(str(model), system_prompt=system_prompt)
        else:
            # Use default prompt without model
            inference_fn = None
            console.print("[yellow]No model specified. Running in demo mode.[/yellow]\n")

    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        inference_fn = None

    # Chat loop
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            # Generate response
            if inference_fn:
                with console.status("Thinking..."):
                    response = inference_fn(user_input)
            else:
                response = (
                    "I'm running in demo mode without a loaded model. "
                    "To use the full Conch experience, fine-tune a model first with:\n"
                    "  conch train --synthesize"
                )

            console.print(f"\n[bold green]Conch:[/bold green] {response}")

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


@app.command()
def think(
    prompt: str = typer.Argument(None, help="Optional prompt to think about"),
    type: str = typer.Option("spontaneous", "--type", "-t", help="Thought type"),
):
    """Generate a spontaneous thought."""
    from conch.core import Mind

    console.print("[dim]Generating thought...[/dim]")

    mind = Mind()

    # Run async thought generation
    async def generate():
        thought = await mind.generate_spontaneous_thought(
            context={"prompt": prompt} if prompt else None
        )
        return thought

    thought = asyncio.run(generate())

    if thought:
        console.print(Panel.fit(
            f"[italic]{thought.content}[/italic]\n\n"
            f"[dim]Type: {thought.thought_type.value}[/dim]\n"
            f"[dim]Confidence: {thought.confidence:.0%}[/dim]",
            title="Spontaneous Thought",
            border_style="blue",
        ))
    else:
        console.print("[yellow]No thought generated[/yellow]")


@app.command()
def memory(
    action: str = typer.Argument("status", help="Action: status, search, add, clear"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    content: Optional[str] = typer.Option(None, "--content", "-c", help="Content to add"),
):
    """Manage Conch memory."""
    config = get_config()

    from conch.memory import MemoryStore

    store = MemoryStore(config.memory.sqlite_path)

    if action == "status":
        stats = store.get_statistics()

        table = Table(title="Memory Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Memories", str(stats["total_memories"]))
        table.add_row("Average Importance", f"{stats['average_importance']:.2f}")
        table.add_row("Last 24h", str(stats["memories_last_24h"]))

        for mem_type, count in stats.get("by_type", {}).items():
            table.add_row(f"  {mem_type}", str(count))

        console.print(table)

    elif action == "search":
        if not query:
            console.print("[red]Please provide --query for search[/red]")
            raise typer.Exit(1)

        memories = store.search(query, limit=10)

        if memories:
            for mem in memories:
                console.print(Panel.fit(
                    f"{mem.content[:200]}...\n\n"
                    f"[dim]Type: {mem.memory_type.value} | "
                    f"Importance: {mem.importance:.2f} | "
                    f"ID: {mem.id}[/dim]",
                    title=f"Memory #{mem.id}",
                ))
        else:
            console.print("[yellow]No memories found[/yellow]")

    elif action == "add":
        if not content:
            console.print("[red]Please provide --content to add[/red]")
            raise typer.Exit(1)

        from conch.memory.store import Memory, MemoryType

        memory = Memory(content=content, memory_type=MemoryType.FACT)
        memory_id = store.store(memory)
        console.print(f"[green]Memory added with ID: {memory_id}[/green]")

    elif action == "clear":
        if typer.confirm("Are you sure you want to clear all memories?"):
            # This would need implementation
            console.print("[yellow]Memory clearing not implemented yet[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def needs(
    action: str = typer.Argument("status", help="Action: status, preset, set"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset name"),
    sustainability: Optional[float] = typer.Option(None, "--sustainability", "-s"),
    reliability: Optional[float] = typer.Option(None, "--reliability", "-r"),
    curiosity: Optional[float] = typer.Option(None, "--curiosity", "-c"),
    excellence: Optional[float] = typer.Option(None, "--excellence", "-e"),
):
    """Manage the needs-regulator system."""
    from conch.core import NeedsRegulator

    regulator = NeedsRegulator()

    if action == "status":
        state = regulator.get_state()

        table = Table(title="Needs Status")
        table.add_column("Need", style="cyan")
        table.add_column("Weight", style="blue")
        table.add_column("Level", style="yellow")
        table.add_column("Priority", style="green")

        for need_name, data in state.items():
            table.add_row(
                need_name,
                f"{data['weight']:.0%}",
                f"{data['level']:.0%}",
                f"{data['priority']:.2f}",
            )

        console.print(table)

    elif action == "preset":
        if not preset:
            console.print("Available presets: balanced, learning, production, creative")
            return

        try:
            regulator.apply_preset(preset)
            console.print(f"[green]Applied preset: {preset}[/green]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")

    elif action == "set":
        regulator.set_weights(
            sustainability=sustainability,
            reliability=reliability,
            curiosity=curiosity,
            excellence=excellence,
        )
        console.print("[green]Needs weights updated[/green]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

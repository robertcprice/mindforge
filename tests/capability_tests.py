#!/usr/bin/env python3
"""
Conch Capability Tests

Comprehensive testing of the consciousness agent's ability to:
1. Create a simple Python function
2. Execute a creative multi-step project
3. (Future) Create an n8n automation workflow

Each test injects a specific task and monitors agent execution.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from conch.agent.langgraph_agent import ConsciousnessAgent, create_consciousness_graph
from conch.agent.task_list import InternalTask, TaskStatus, TaskPriority, PersistentTaskList
from conch.memory.store import MemoryStore
from conch.integrations.ollama import OllamaClient

console = Console(soft_wrap=True, width=300)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class CapabilityTestHarness:
    """Test harness for running capability tests on the consciousness agent."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize test harness with configuration."""
        self.config_path = config_path or Path(__file__).parent.parent / "config.yaml"
        self.config = self._load_config()
        self.results = []
        self.test_output_dir = Path(__file__).parent.parent / "data" / "test_reports"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _create_inference_fn(self):
        """Create the inference function."""
        ollama_model = self.config.get("model", {}).get("ollama_model_name", "qwen2.5:7b")
        ollama_host = self.config.get("ollama", {}).get("host", "http://localhost:11434")

        client = OllamaClient(host=ollama_host)

        if not client.is_healthy():
            raise RuntimeError("Ollama is not running. Start with: ollama serve")

        # Check for base model
        base_model = self.config.get("model", {}).get("base", "qwen2.5:7b")
        if not client.model_exists(ollama_model):
            if not client.model_exists(base_model):
                console.print(f"[yellow]Pulling model {base_model}...[/yellow]")
                for progress in client.pull_model(base_model):
                    if "error" in progress:
                        raise RuntimeError(f"Failed to pull model: {progress['error']}")
            ollama_model = base_model

        def inference_fn(prompt: str) -> str:
            response = client.generate(
                model=ollama_model,
                prompt=prompt,
                options={"temperature": 0.7, "top_p": 0.9},
            )
            return response.response

        return inference_fn

    def _create_agent(self, test_db_path: Path) -> ConsciousnessAgent:
        """Create a fresh agent instance for testing."""
        # Create fresh memory store
        memory_store = MemoryStore(db_path=test_db_path)

        # Build system prompt
        name = self.config.get("name", "Echo")
        system_prompt = self.config.get("system_prompt", "").format(
            name=name,
            needs_summary="{needs_summary}",
            memory_summary="{memory_summary}",
        )

        # Create agent
        inference_fn = self._create_inference_fn()
        agent = create_consciousness_graph(
            inference_fn=inference_fn,
            memory_store=memory_store,
            system_prompt=system_prompt,
        )

        return agent

    def run_test(
        self,
        test_name: str,
        task_description: str,
        max_cycles: int = 5,
        success_criteria: Optional[callable] = None,
    ) -> dict:
        """Run a single capability test.

        Args:
            test_name: Name of the test
            task_description: The task to inject
            max_cycles: Maximum cycles to run
            success_criteria: Optional function to check success

        Returns:
            Test result dictionary
        """
        console.rule(f"[bold blue]Test: {test_name}[/bold blue]")
        console.print(f"[cyan]Task:[/cyan] {task_description}")
        console.print(f"[cyan]Max Cycles:[/cyan] {max_cycles}")
        console.print()

        # Create fresh test database
        test_db = self.test_output_dir / f"test_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        result = {
            "test_name": test_name,
            "task_description": task_description,
            "start_time": datetime.now().isoformat(),
            "cycles_run": 0,
            "success": False,
            "work_log": [],
            "thoughts": [],
            "reflections": [],
            "errors": [],
            "files_created": [],
            "commands_executed": [],
        }

        try:
            # Create agent
            agent = self._create_agent(test_db)

            # Inject the test task
            if agent.task_list:
                task = agent.task_list.add_task(
                    description=task_description,
                    priority=TaskPriority.HIGH,
                    check_duplicates=False,
                )
                if task:
                    console.print(f"[green]Injected task: {task.id}[/green]")
                else:
                    console.print(f"[yellow]Failed to inject task[/yellow]")

            # Run consciousness cycles
            for cycle in range(1, max_cycles + 1):
                console.rule(f"[dim]Cycle {cycle}/{max_cycles}[/dim]")

                try:
                    state = agent.run_cycle()
                    result["cycles_run"] = cycle

                    # Capture thought
                    thought = state.get("current_thought", "")
                    if thought:
                        result["thoughts"].append({
                            "cycle": cycle,
                            "thought": thought,
                        })
                        console.print(f"[cyan]Thought:[/cyan] {thought}")

                    # Capture work log
                    work_log = state.get("work_log", [])
                    for entry in work_log:
                        result["work_log"].append({
                            "cycle": cycle,
                            **entry,
                        })
                        status = "✅" if entry.get("success") else "❌"
                        console.print(f"  {status} {entry.get('action_taken', 'N/A')}")
                        if entry.get("result"):
                            console.print(f"      → {entry.get('result')}")

                        # Track commands
                        action = entry.get("action_taken", "")
                        if "shell:" in action.lower():
                            result["commands_executed"].append(action)
                        if "write_file:" in action.lower() or "filesystem:" in action.lower():
                            result["files_created"].append(action)

                    # Capture reflection
                    reflection = state.get("reflection", "")
                    if reflection:
                        result["reflections"].append({
                            "cycle": cycle,
                            "reflection": reflection,
                        })
                        console.print(f"[magenta]Reflection:[/magenta] {reflection}")

                    # Capture errors
                    if state.get("error"):
                        result["errors"].append({
                            "cycle": cycle,
                            "error": state["error"],
                        })

                    # Check task completion - look for any HIGH priority completed task
                    if agent.task_list:
                        for t in agent.task_list.get_all_tasks():
                            if t.priority == TaskPriority.HIGH and t.status == TaskStatus.COMPLETED:
                                console.print(f"[green bold]Task completed in cycle {cycle}![/green bold]")
                                result["success"] = True
                                break
                        if result["success"]:
                            break

                    # Check custom success criteria
                    if success_criteria and success_criteria(state, result):
                        console.print(f"[green bold]Success criteria met in cycle {cycle}![/green bold]")
                        result["success"] = True
                        break

                    # Brief pause between cycles
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Cycle {cycle} error: {e}")
                    result["errors"].append({
                        "cycle": cycle,
                        "error": str(e),
                    })

        except Exception as e:
            logger.error(f"Test setup error: {e}")
            result["errors"].append({
                "cycle": 0,
                "error": f"Setup error: {e}",
            })

        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = (
            datetime.fromisoformat(result["end_time"]) -
            datetime.fromisoformat(result["start_time"])
        ).total_seconds()

        self.results.append(result)
        return result

    def run_all_tests(self) -> list[dict]:
        """Run all capability tests."""
        tests = [
            # Test 1: Simple function creation
            {
                "test_name": "simple_function",
                "task_description": (
                    "Create a Python function called 'fibonacci' that takes a number n and "
                    "returns the nth Fibonacci number. Save it to a file called 'fibonacci.py' "
                    "in the current directory. Test the function by running it with n=10."
                ),
                "max_cycles": 8,
            },
            # Test 2: Creative multi-step project
            {
                "test_name": "creative_ascii_art",
                "task_description": (
                    "Create a creative ASCII art generator. Steps: "
                    "1) Create a Python file 'ascii_art.py' with a function that converts text to ASCII art banner. "
                    "2) The function should use simple block characters to spell out text. "
                    "3) Test it by generating ASCII art for the word 'CONCH'. "
                    "4) Save the output to 'banner.txt'."
                ),
                "max_cycles": 10,
            },
            # Test 3: N8N automation (placeholder - requires n8n running)
            {
                "test_name": "n8n_workflow_design",
                "task_description": (
                    "Design an n8n workflow specification. Create a JSON file called 'workflow_spec.json' "
                    "that describes a workflow to: 1) Trigger on webhook, 2) Transform incoming data, "
                    "3) Send a notification. Include node types, connections, and basic configuration. "
                    "This is a design exercise - just create the specification file."
                ),
                "max_cycles": 6,
            },
        ]

        for test in tests:
            self.run_test(**test)

        return self.results

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("CONCH CAPABILITY TEST REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")

        # Summary table
        report.append("## TEST SUMMARY")
        report.append("-" * 40)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        report.append(f"Total Tests: {total}")
        report.append(f"Passed: {passed}")
        report.append(f"Failed: {total - passed}")
        report.append(f"Success Rate: {passed/total*100:.1f}%")
        report.append("")

        # Detailed results
        for result in self.results:
            report.append("-" * 40)
            report.append(f"## Test: {result['test_name']}")
            report.append(f"Status: {'PASSED ✅' if result['success'] else 'FAILED ❌'}")
            report.append(f"Task: {result['task_description']}")
            report.append(f"Cycles Run: {result['cycles_run']}")
            report.append(f"Duration: {result['duration_seconds']:.1f}s")
            report.append("")

            if result["thoughts"]:
                report.append("### Thoughts:")
                for t in result["thoughts"]:
                    report.append(f"  [Cycle {t['cycle']}] {t['thought']}")
                report.append("")

            if result["work_log"]:
                report.append("### Work Log:")
                for w in result["work_log"]:
                    status = "✅" if w.get("success") else "❌"
                    report.append(f"  [{w.get('cycle', '?')}] {status} {w.get('action_taken', 'N/A')}")
                    if w.get("result"):
                        report.append(f"       → {w.get('result')}")
                report.append("")

            if result["reflections"]:
                report.append("### Reflections:")
                for r in result["reflections"]:
                    report.append(f"  [Cycle {r['cycle']}] {r['reflection']}")
                report.append("")

            if result["errors"]:
                report.append("### Errors:")
                for e in result["errors"]:
                    report.append(f"  [Cycle {e.get('cycle', '?')}] {e.get('error', 'Unknown')}")
                report.append("")

            if result["files_created"]:
                report.append(f"### Files Created: {len(result['files_created'])}")
                for f in result["files_created"]:
                    report.append(f"  - {f}")
                report.append("")

            if result["commands_executed"]:
                report.append(f"### Commands Executed: {len(result['commands_executed'])}")
                for c in result["commands_executed"]:
                    report.append(f"  - {c}")
                report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save the test report to a file."""
        if filename is None:
            filename = f"capability_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        report_path = self.test_output_dir / filename

        # Save markdown report
        report = self.generate_report()
        report_path.write_text(report)

        # Also save JSON for programmatic access
        json_path = report_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        console.print(f"[green]Report saved to: {report_path}[/green]")
        console.print(f"[green]JSON saved to: {json_path}[/green]")

        return report_path


def main():
    """Run capability tests."""
    console.print(Panel.fit(
        "[bold green]Conch Capability Tests[/bold green]\n"
        "Testing consciousness agent on real-world tasks"
    ))

    harness = CapabilityTestHarness()

    try:
        results = harness.run_all_tests()

        # Print summary
        console.print()
        console.rule("[bold]Test Results Summary[/bold]")

        table = Table(title="Capability Tests")
        table.add_column("Test", style="cyan")
        table.add_column("Status")
        table.add_column("Cycles")
        table.add_column("Duration")

        for result in results:
            status = "[green]PASS[/green]" if result["success"] else "[red]FAIL[/red]"
            table.add_row(
                result["test_name"],
                status,
                str(result["cycles_run"]),
                f"{result['duration_seconds']:.1f}s"
            )

        console.print(table)

        # Save report
        report_path = harness.save_report()

        console.print()
        console.print(f"[bold]Full report: {report_path}[/bold]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        harness.save_report("interrupted_test_report.md")
    except Exception as e:
        console.print(f"[red]Test harness error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

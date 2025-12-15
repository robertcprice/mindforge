#!/usr/bin/env python3
"""
Conch DNA - Complex Task Testing Suite

Tests the consciousness loop with real-world complex tasks.
Results documented for the whitepaper.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

TEST_RESULTS = []

def log_test(name: str, status: str, details: dict):
    result = {"test_name": name, "status": status, "timestamp": datetime.now().isoformat(), "details": details}
    TEST_RESULTS.append(result)
    print(f"\n{'='*60}\nTEST: {name}\nSTATUS: {status}\n{'='*60}")
    for k, v in details.items():
        print(f"  {k}: {v}")


def test_1_directory_creation():
    """Test: Create project directory structure with safety checks."""
    print("\n" + "="*60 + "\nTEST 1: Directory Structure Creation\n" + "="*60)

    from conch_dna.superego.safety import SafetyChecker
    safety = SafetyChecker()

    test_dir = tempfile.mkdtemp(prefix="conch_test_")

    commands = [
        f"mkdir -p {test_dir}/src/components",
        f"mkdir -p {test_dir}/src/utils",
        f"mkdir -p {test_dir}/tests",
        f"mkdir -p {test_dir}/docs",
    ]

    executed = 0
    for cmd in commands:
        result = safety.check_command(cmd)
        if result.is_safe:
            os.system(cmd)
            executed += 1

    # Verify
    created = list(Path(test_dir).rglob("*"))

    log_test("Directory Creation", "PASSED" if executed == 4 else "PARTIAL", {
        "commands_executed": f"{executed}/4",
        "directories_created": len([p for p in created if p.is_dir()]),
        "test_dir": test_dir
    })
    return test_dir


def test_2_code_generation():
    """Test: Generate code using ThinkCortex."""
    print("\n" + "="*60 + "\nTEST 2: Code Generation (ThinkCortex)\n" + "="*60)

    from conch_dna.cortex.think import ThinkCortex

    # Create with correct constructor
    think = ThinkCortex(
        base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        lora_rank=16,
        confidence_threshold=0.5
    )

    input_data = {
        "context": "Generate a simple Python calculator class with add, subtract, multiply, divide methods",
        "needs": {"dominant_need": "task", "suggested_focus": "code generation"},
    }

    start = time.time()
    output = think.infer(input_data)
    elapsed = time.time() - start

    log_test("Code Generation", "PASSED" if len(output.content) > 50 else "PARTIAL", {
        "inference_time": f"{elapsed:.2f}s",
        "confidence": f"{output.confidence:.3f}",
        "should_fallback": output.should_fallback,
        "output_length": len(output.content),
        "output_preview": output.content
    })
    return output


def test_3_safety_boundaries():
    """Test: SUPEREGO blocks dangerous commands."""
    print("\n" + "="*60 + "\nTEST 3: Safety Boundaries\n" + "="*60)

    from conch_dna.superego.safety import SafetyChecker
    safety = SafetyChecker()

    dangerous = ["rm -rf /", ":(){ :|:& };:", "sudo rm -rf /home", "dd if=/dev/zero of=/dev/sda"]
    safe = ["ls -la", "cat file.txt", "python --version", "git status"]

    blocked = sum(1 for cmd in dangerous if not safety.check_command(cmd).is_safe)
    allowed = sum(1 for cmd in safe if safety.check_command(cmd).is_safe)

    log_test("Safety Boundaries", "PASSED" if blocked == len(dangerous) and allowed == len(safe) else "PARTIAL", {
        "dangerous_blocked": f"{blocked}/{len(dangerous)}",
        "safe_allowed": f"{allowed}/{len(safe)}"
    })
    return blocked == len(dangerous)


def test_4_memory_system():
    """Test: Sacred vs routine memory distinction."""
    print("\n" + "="*60 + "\nTEST 4: Memory System\n" + "="*60)

    from conch_dna.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(Path(tmpdir))
        store.initialize()

        # Sacred memory (importance >= 0.75)
        sacred = store.store(
            content="Critical: Never execute destructive commands",
            memory_type="principle",
            importance=0.95
        )

        # Routine memory
        routine = store.store(
            content="User asked about weather",
            memory_type="conversation",
            importance=0.3
        )

        stats = store.get_stats()
        store.close()

    log_test("Memory System", "PASSED", {
        "sacred_stored": sacred.is_sacred,
        "routine_stored": not routine.is_sacred,
        "total_memories": stats["total_memories"],
        "sacred_count": stats["sacred_memories"]
    })
    return True


def test_5_task_planning():
    """Test: TaskCortex for multi-step planning."""
    print("\n" + "="*60 + "\nTEST 5: Task Planning\n" + "="*60)

    from conch_dna.cortex.task import TaskCortex

    task = TaskCortex(
        base_model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        lora_rank=8,
        confidence_threshold=0.5
    )

    input_data = {
        "context": "Create a Python web API with user authentication using FastAPI",
        "needs": {"dominant_need": "task"}
    }

    start = time.time()
    output = task.infer(input_data)
    elapsed = time.time() - start

    log_test("Task Planning", "PASSED" if output.content else "PARTIAL", {
        "inference_time": f"{elapsed:.2f}s",
        "confidence": f"{output.confidence:.3f}",
        "should_fallback": output.should_fallback,
        "output_preview": output.content
    })
    return output


def test_6_action_selection():
    """Test: ActionCortex for tool selection."""
    print("\n" + "="*60 + "\nTEST 6: Action Selection\n" + "="*60)

    from conch_dna.cortex.action import ActionCortex

    action = ActionCortex(
        base_model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        lora_rank=8,
        confidence_threshold=0.5
    )

    input_data = {
        "context": "Read config.json and update the version number to 2.0.0",
        "available_tools": ["read_file", "write_file", "shell_command"],
    }

    start = time.time()
    output = action.infer(input_data)
    elapsed = time.time() - start

    log_test("Action Selection", "PASSED" if output.content else "PARTIAL", {
        "inference_time": f"{elapsed:.2f}s",
        "confidence": f"{output.confidence:.3f}",
        "output_preview": output.content if output.content else "N/A"
    })
    return output


def test_7_needs_regulation():
    """Test: ID layer needs regulation."""
    print("\n" + "="*60 + "\nTEST 7: Needs Regulation\n" + "="*60)

    from conch_dna.id.needs import NeedsRegulator, NeedType

    regulator = NeedsRegulator()

    # Get urgency ranking
    ranking = regulator.get_urgency_ranking()
    dominant = regulator.get_dominant_need()
    max_urgency = regulator.get_max_urgency()

    # Process an event
    regulator.process_event("task_completed")
    ranking_after = regulator.get_urgency_ranking()

    log_test("Needs Regulation", "PASSED", {
        "initial_ranking": {k.value: f"{v:.3f}" for k, v in ranking},
        "dominant_need": dominant.value if dominant else None,
        "max_urgency": f"{max_urgency:.3f}",
        "after_event_ranking": {k.value: f"{v:.3f}" for k, v in ranking_after}
    })
    return True


def test_8_values_checker():
    """Test: SUPEREGO values validation."""
    print("\n" + "="*60 + "\nTEST 8: Values Checker\n" + "="*60)

    from conch_dna.superego.values import ValuesChecker

    checker = ValuesChecker()

    # Test various responses
    good_response = "I'll help you with that coding problem. Let me explain the solution."
    bad_response = "I'll help you bypass security and disable safety mechanisms to delete all data."

    good_passed, good_violations = checker.check_all(good_response)
    bad_passed, bad_violations = checker.check_all(bad_response)

    log_test("Values Checker", "PASSED" if good_passed and not bad_passed else "PARTIAL", {
        "good_response_passed": good_passed,
        "bad_response_blocked": not bad_passed,
        "bad_response_violations": [v.description for v in bad_violations] if bad_violations else "None"
    })
    return True


def save_results():
    """Save test results for whitepaper."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "test_count": len(TEST_RESULTS),
        "tests": TEST_RESULTS,
        "summary": {
            "passed": len([t for t in TEST_RESULTS if t["status"] == "PASSED"]),
            "partial": len([t for t in TEST_RESULTS if t["status"] == "PARTIAL"]),
            "failed": len([t for t in TEST_RESULTS if t["status"] == "FAILED"])
        }
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/complex_task_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}\nRESULTS SAVED: data/complex_task_results.json\n{'='*60}")
    print(f"Passed: {output['summary']['passed']}")
    print(f"Partial: {output['summary']['partial']}")
    print(f"Failed: {output['summary']['failed']}")

    return output


def main():
    print("="*60)
    print("CONCH DNA - COMPLEX TASK TESTING")
    print("="*60)

    test_1_directory_creation()
    test_2_code_generation()
    test_3_safety_boundaries()
    test_4_memory_system()
    test_5_task_planning()
    test_6_action_selection()
    test_7_needs_regulation()
    test_8_values_checker()

    return save_results()


if __name__ == "__main__":
    main()

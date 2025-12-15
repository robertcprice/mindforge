#!/usr/bin/env python3
"""
Practical Task Tests for Conch Consciousness Engine

Tests the engine with concrete, real-world tasks to evaluate:
1. Output quality (concrete vs meta)
2. Action-bias (doing vs philosophizing)
3. Task completion (produces actual deliverables)

Tasks:
1. Creative writing: Write a haiku
2. File directory planning: Plan a web app project structure
3. Coding: Write a simple calculator function
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from conch.integrations.ollama import OllamaClient

# Test configuration
OLLAMA_MODEL = "qwen3:8b"
OUTPUT_DIR = Path("artifacts/practical_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Meta-quality indicators (things we want to AVOID)
META_PATTERNS = [
    r"let me think about",
    r"i should consider",
    r"it's important to understand",
    r"before i begin",
    r"first, i need to reflect",
    r"this is a complex",
    r"there are many ways to",
    r"i could approach this by",
    r"let me analyze",
    r"philosophically speaking",
    r"from a meta perspective",
    r"the nature of",
    r"inherently",
    r"fundamentally",
]

# Action-bias indicators (things we WANT)
ACTION_PATTERNS = [
    r"here is",
    r"```",  # Code blocks
    r"def\s+\w+",  # Function definitions
    r"class\s+\w+",  # Class definitions
    r"[│├└─]",  # Directory tree characters
    r"^\s*-\s+\w+",  # List items
    r"^\d+\.",  # Numbered steps
]


def score_output(text: str) -> dict:
    """Score output for meta vs action quality."""
    text_lower = text.lower()

    # Count meta patterns (bad)
    meta_count = sum(1 for p in META_PATTERNS if re.search(p, text_lower))

    # Count action patterns (good)
    action_count = sum(1 for p in ACTION_PATTERNS if re.search(p, text, re.MULTILINE))

    # Calculate scores
    word_count = len(text.split())
    meta_density = meta_count / max(word_count / 100, 1)
    action_density = action_count / max(word_count / 100, 1)

    # Overall score: higher = more concrete/actionable
    # Negative for meta, positive for action
    overall = (action_density * 10) - (meta_density * 15)

    return {
        "word_count": word_count,
        "meta_matches": meta_count,
        "action_matches": action_count,
        "meta_density": round(meta_density, 3),
        "action_density": round(action_density, 3),
        "overall_score": round(overall, 2),
        "verdict": "CONCRETE" if overall >= 0 else "TOO META"
    }


def test_creative_writing(client: OllamaClient) -> dict:
    """Test: Write a haiku about coding."""
    print("\n" + "="*60)
    print("TEST 1: Creative Writing - Write a Haiku")
    print("="*60)

    prompt = """Write a haiku about debugging code.
Output ONLY the haiku (3 lines, 5-7-5 syllables). No explanation."""

    print(f"Prompt: {prompt}\n")

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.7, "num_predict": -1}
    )

    output = response.response.strip()
    print(f"Output:\n{output}\n")

    # Score
    scores = score_output(output)
    print(f"Scores: {json.dumps(scores, indent=2)}")

    # Check if it's actually a haiku (3 lines)
    lines = [l for l in output.split('\n') if l.strip()]
    is_haiku = len(lines) == 3

    return {
        "task": "Creative Writing - Haiku",
        "prompt": prompt,
        "output": output,
        "is_valid_format": is_haiku,
        "line_count": len(lines),
        "scores": scores
    }


def test_directory_structure(client: OllamaClient) -> dict:
    """Test: Plan a web app project structure."""
    print("\n" + "="*60)
    print("TEST 2: File Directory - Plan Web App Structure")
    print("="*60)

    prompt = """Create a directory structure for a React + Node.js web app.
Output ONLY the tree structure using ASCII characters (├── └──).
Include: src/, public/, server/, tests/. Max 15 items."""

    print(f"Prompt: {prompt}\n")

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": -1}
    )

    output = response.response.strip()
    print(f"Output:\n{output}\n")

    # Score
    scores = score_output(output)
    print(f"Scores: {json.dumps(scores, indent=2)}")

    # Check if it has tree characters
    has_tree = any(c in output for c in ['├', '└', '│', '─'])
    has_required_dirs = all(d in output for d in ['src', 'server', 'tests'])

    return {
        "task": "Directory Structure",
        "prompt": prompt,
        "output": output,
        "is_valid_format": has_tree,
        "has_required_dirs": has_required_dirs,
        "scores": scores
    }


def test_calculator_code(client: OllamaClient) -> dict:
    """Test: Write a calculator function."""
    print("\n" + "="*60)
    print("TEST 3: Coding - Calculator Function")
    print("="*60)

    prompt = """Write a Python function called 'calculate' that takes two numbers
and an operator (+, -, *, /) and returns the result.
Output ONLY the Python code. No explanation."""

    print(f"Prompt: {prompt}\n")

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.2, "num_predict": -1}
    )

    output = response.response.strip()
    print(f"Output:\n{output}\n")

    # Score
    scores = score_output(output)
    print(f"Scores: {json.dumps(scores, indent=2)}")

    # Check if it's valid Python with 'def calculate'
    has_function = 'def calculate' in output
    has_operators = all(op in output for op in ['+', '-', '*', '/'])

    return {
        "task": "Calculator Code",
        "prompt": prompt,
        "output": output,
        "is_valid_format": has_function,
        "has_all_operators": has_operators,
        "scores": scores
    }


def test_task_decomposition(client: OllamaClient) -> dict:
    """Test: Break down a task (tests meta-task filtering)."""
    print("\n" + "="*60)
    print("TEST 4: Task Decomposition (Meta-Task Filter Test)")
    print("="*60)

    prompt = """Break down this task into 3 actionable steps:
"Deploy a Python Flask app to a server"

Rules:
- Each step must be a DIRECT ACTION (use a command or tool)
- NO "research", "investigate", "explore", "verify", or "analyze" steps
- Format: numbered list

Output ONLY the 3 steps."""

    print(f"Prompt: {prompt}\n")

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": -1}
    )

    output = response.response.strip()
    print(f"Output:\n{output}\n")

    # Score
    scores = score_output(output)
    print(f"Scores: {json.dumps(scores, indent=2)}")

    # Check for meta-task keywords
    meta_keywords = ["research", "investigate", "explore", "verify", "analyze", "understand", "consider"]
    meta_found = [kw for kw in meta_keywords if kw in output.lower()]

    # Count actual steps
    steps = re.findall(r'^\d+\..*', output, re.MULTILINE)

    return {
        "task": "Task Decomposition",
        "prompt": prompt,
        "output": output,
        "step_count": len(steps),
        "meta_keywords_found": meta_found,
        "is_action_biased": len(meta_found) == 0,
        "scores": scores
    }


def main():
    print("="*60)
    print("PRACTICAL TASK TESTS - Conch Consciousness Engine")
    print("="*60)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Time: {datetime.now().isoformat()}")

    # Initialize Ollama client
    client = OllamaClient()

    if not client.is_healthy():
        print("ERROR: Ollama is not running!")
        return 1

    if not client.model_exists(OLLAMA_MODEL):
        print(f"ERROR: Model {OLLAMA_MODEL} not found!")
        return 1

    print(f"Ollama: Connected")

    # Run tests
    results = []

    results.append(test_creative_writing(client))
    results.append(test_directory_structure(client))
    results.append(test_calculator_code(client))
    results.append(test_task_decomposition(client))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_concrete = True
    for r in results:
        verdict = r["scores"]["verdict"]
        score = r["scores"]["overall_score"]
        valid = r.get("is_valid_format", True)

        status = "✅" if verdict == "CONCRETE" and valid else "⚠️"
        if verdict != "CONCRETE":
            all_concrete = False

        print(f"{status} {r['task']}: {verdict} (score={score}, valid_format={valid})")

        if r.get("meta_keywords_found"):
            print(f"   Meta keywords found: {r['meta_keywords_found']}")

    print("="*60)

    if all_concrete:
        print("✅ ALL TESTS PASSED - Outputs are CONCRETE and action-biased!")
    else:
        print("⚠️ SOME TESTS SHOW META-QUALITY ISSUES")

    # Save results
    output_file = OUTPUT_DIR / f"practical_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0 if all_concrete else 1


if __name__ == "__main__":
    sys.exit(main())

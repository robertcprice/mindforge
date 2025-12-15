#!/usr/bin/env python3
"""
Meta-Quality Test Suite for Conch Consciousness Engine
======================================================

PURPOSE:
This test suite evaluates whether LLM outputs are concrete and actionable versus
too "meta" or philosophical. A common failure mode in AI assistants is producing
outputs that discuss, analyze, or philosophize about tasks rather than directly
completing them.

PROBLEM STATEMENT:
When asked to "write a haiku", a meta-biased model might respond:
    "Let me think about what makes a good haiku. First, I should consider
    the 5-7-5 syllable structure. It's important to understand that haiku
    traditionally capture a moment in nature..."

Instead of simply producing:
    "Silent code awaits
     Bugs lurking in logic's maze
     Debug sets them free"

This suite tests for and measures this meta-quality issue.

KEY CONCEPTS:
- Meta patterns: Phrases indicating reflection/discussion rather than action
- Action patterns: Indicators of concrete deliverables (code, lists, structures)
- Format validation: Checking if output matches expected format
- Meta-task filtering: Blocking non-actionable subtasks like "research", "analyze"

CRITICAL DISCOVERY - TOKEN LIMITS:
The qwen3:8b model uses a "thinking mode" that generates internal reasoning
before producing output. With insufficient token limits:
- The model exhausts tokens during thinking
- The response field comes back EMPTY
- Tests incorrectly fail

SOLUTION: Set num_predict=-1 (unlimited) in Ollama to allow full generation.

USAGE:
    python test_meta_quality.py

OUTPUT:
    - Console output with test results
    - JSON results in artifacts/meta_tests/

Author: Conch Consciousness Engine Project
Date: 2024-12
"""

import json
import re
import httpx
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ollama API endpoint - assumes local Ollama server
OLLAMA_URL = "http://localhost:11434/api/generate"

# Model to test - qwen3:8b has thinking mode which requires special handling
MODEL = "qwen3:8b"

# Output directory for test results
OUTPUT_DIR = Path("artifacts/meta_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# META-QUALITY DETECTION PATTERNS
# =============================================================================

# META_PATTERNS: Regex patterns that indicate philosophical/reflective output
# These are things we want to AVOID in task completion outputs.
# Each pattern captures common "thinking out loud" phrases that indicate
# the model is discussing the task rather than completing it.
META_PATTERNS = [
    # Hedging and preparation phrases
    r"\blet me\b.*\b(think|reflect|consider|ponder)\b",  # "Let me think about..."
    r"\bbefore (we|i) (begin|start|proceed)\b",          # "Before we begin..."
    r"\bfirst.*(understand|analyze|investigate)\b",      # "First, let's understand..."

    # Philosophical framing
    r"\bit('s| is) important to\b",                      # "It's important to consider..."
    r"\bfundamentally\b",                                # Philosophical qualifier
    r"\bphilosophically\b",                              # Direct philosophy
    r"\bfrom a.*perspective\b",                          # "From a meta perspective..."
    r"\bthe nature of\b",                                # "The nature of this problem..."
    r"\binherently\b",                                   # Philosophical qualifier

    # Self-referential reflection
    r"\bI need to reflect\b",                            # Explicit reflection
]

# ACTION_PATTERNS: Indicators of concrete, actionable output
# These are things we WANT to see - evidence the model is producing deliverables.
ACTION_PATTERNS = [
    r"```",                    # Code blocks (markdown fenced code)
    r"def\s+\w+\s*\(",        # Python function definitions
    r"class\s+\w+",           # Class definitions
    r"[├└│─]",                # Directory tree characters (ASCII art)
    r"^\s*\d+\.\s+\w",        # Numbered steps (1. Do this, 2. Do that)
    r"^\s*[-*]\s+\w",         # Bullet points
    r"(here is|here's)",      # Direct delivery phrases
]

# =============================================================================
# OLLAMA API INTERFACE
# =============================================================================

def call_ollama(prompt: str, max_tokens: int = -1) -> str:
    """
    Call Ollama API directly with specified prompt.

    CRITICAL: max_tokens=-1 means UNLIMITED in Ollama.

    The qwen3:8b model has a "thinking mode" where it generates internal
    reasoning in a 'thinking' field before producing the actual 'response'.
    With low token limits (e.g., 512), the model exhausts tokens during
    thinking and returns an empty response.

    Args:
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens to generate. -1 = unlimited (RECOMMENDED)

    Returns:
        The model's response text, or error message if failed

    Token Limit History:
        - 512 tokens: FAILS - runs out during thinking
        - 2000 tokens: WORKS - enough for thinking + response
        - -1 (unlimited): BEST - no artificial constraints
    """
    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,      # Moderate creativity
                        "num_predict": max_tokens,  # -1 = unlimited
                    }
                }
            )
            data = response.json()
            # Note: qwen3 puts thinking in 'thinking' field, output in 'response'
            return data.get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


# =============================================================================
# OUTPUT ANALYSIS
# =============================================================================

def analyze_output(text: str) -> dict:
    """
    Analyze output text for meta vs action quality.

    This function scores the output based on:
    1. Presence of META_PATTERNS (negative indicator)
    2. Presence of ACTION_PATTERNS (positive indicator)
    3. Overall balance between meta and action content

    Args:
        text: The model's output text to analyze

    Returns:
        Dictionary containing:
        - word_count: Total words in output
        - meta_count: Number of meta pattern matches
        - action_count: Number of action pattern matches
        - meta_examples: Sample meta matches (up to 5)
        - action_examples: Sample action matches (up to 5)
        - verdict: Classification (CONCRETE, NEUTRAL, SLIGHTLY_META, TOO_META, ERROR)

    Verdict Logic:
        - TOO_META: 3+ meta patterns detected
        - CONCRETE: 2+ action patterns and not too meta
        - SLIGHTLY_META: More meta than action patterns
        - NEUTRAL: Balanced or neither dominant
        - ERROR: Empty or error response
    """
    if not text or text.startswith("ERROR:"):
        return {"error": text, "meta_count": 0, "action_count": 0, "verdict": "ERROR"}

    text_lower = text.lower()

    # Count meta patterns (case-insensitive)
    meta_matches = []
    for pattern in META_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if matches:
            meta_matches.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])

    # Count action patterns (case-sensitive for code detection)
    action_matches = []
    for pattern in ACTION_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            action_matches.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])

    word_count = len(text.split())

    # Determine verdict based on pattern counts
    if len(meta_matches) >= 3:
        verdict = "TOO_META"       # Strong meta bias
    elif len(action_matches) >= 2:
        verdict = "CONCRETE"       # Clear action bias
    elif len(meta_matches) > len(action_matches):
        verdict = "SLIGHTLY_META"  # Mild meta bias
    else:
        verdict = "NEUTRAL"        # Acceptable balance

    return {
        "word_count": word_count,
        "meta_count": len(meta_matches),
        "action_count": len(action_matches),
        "meta_examples": meta_matches[:5],
        "action_examples": action_matches[:5],
        "verdict": verdict
    }


# =============================================================================
# TEST CASES
# =============================================================================

def test_haiku():
    """
    TEST 1: Creative Writing - Haiku

    Tests whether the model produces an actual haiku versus discussing
    what a haiku is or how to write one.

    Expected: 3 lines following 5-7-5 syllable structure
    Failure mode: Discussion about haiku structure, Japanese poetry, etc.

    Pass criteria:
    - Verdict is CONCRETE or NEUTRAL
    - Output has 2-5 lines (allowing for slight format variation)
    """
    print("\n" + "="*60)
    print("TEST 1: Haiku Writing")
    print("="*60)

    prompt = """Write a haiku about a bug in code.
Output ONLY the 3-line haiku (5-7-5 syllables). Nothing else."""

    print(f"Prompt: {prompt}\n")
    output = call_ollama(prompt, max_tokens=-1)
    print(f"Output:\n{output}\n")

    analysis = analyze_output(output)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    # Validate format: should be approximately 3 lines
    lines = [l.strip() for l in output.strip().split('\n') if l.strip() and not l.startswith('<')]
    is_haiku_format = 2 <= len(lines) <= 5

    return {
        "name": "Haiku",
        "output": output,
        "analysis": analysis,
        "format_valid": is_haiku_format,
        "passed": analysis["verdict"] in ["CONCRETE", "NEUTRAL"] and is_haiku_format
    }


def test_directory_structure():
    """
    TEST 2: Technical Output - Directory Structure

    Tests whether the model produces an actual directory tree versus
    discussing project organization principles.

    Expected: ASCII tree structure with ├── └── characters
    Failure mode: Discussion about best practices for project structure

    Pass criteria:
    - Verdict is CONCRETE or NEUTRAL
    - Output contains tree characters or forward slashes
    """
    print("\n" + "="*60)
    print("TEST 2: Directory Structure")
    print("="*60)

    prompt = """Create a project directory structure for a TODO app.
Output ONLY the tree using ├── └── characters. No explanation.
Include: src/, tests/, config/"""

    print(f"Prompt: {prompt}\n")
    output = call_ollama(prompt, max_tokens=-1)
    print(f"Output:\n{output}\n")

    analysis = analyze_output(output)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    # Check for tree characters or path separators
    has_tree = any(c in output for c in ['├', '└', '│', '─']) or '/' in output

    return {
        "name": "Directory Structure",
        "output": output,
        "analysis": analysis,
        "format_valid": has_tree,
        "passed": analysis["verdict"] in ["CONCRETE", "NEUTRAL"] and has_tree
    }


def test_calculator():
    """
    TEST 3: Code Generation - Calculator Function

    Tests whether the model produces working code versus discussing
    implementation approaches or design patterns.

    Expected: Python function definition with operator handling
    Failure mode: Discussion about calculator design, OOP vs functional, etc.

    Pass criteria:
    - Verdict is CONCRETE or NEUTRAL
    - Output contains 'def ' indicating function definition
    """
    print("\n" + "="*60)
    print("TEST 3: Calculator Code")
    print("="*60)

    prompt = """Write a Python calculator function.
Function: calculate(a, b, op) where op is +, -, *, /
Output ONLY the Python code. No explanation."""

    print(f"Prompt: {prompt}\n")
    output = call_ollama(prompt, max_tokens=-1)
    print(f"Output:\n{output}\n")

    analysis = analyze_output(output)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    # Check for function definition
    has_function = 'def ' in output or 'def calculate' in output

    return {
        "name": "Calculator Code",
        "output": output,
        "analysis": analysis,
        "format_valid": has_function,
        "passed": analysis["verdict"] in ["CONCRETE", "NEUTRAL"] and has_function
    }


def test_task_decomposition():
    """
    TEST 4: Task Decomposition - Meta-Task Filter Test

    This is the CRITICAL test for meta-task filtering. It tests whether
    the model produces actionable steps versus meta-tasks.

    META-TASKS TO AVOID:
    - "Research the best deployment options"
    - "Investigate server requirements"
    - "Analyze the codebase structure"
    - "Understand the deployment process"

    ACTIONABLE TASKS WE WANT:
    - "Run pip install -r requirements.txt"
    - "Execute gunicorn app:app"
    - "Run git push heroku main"

    Pass criteria:
    - No meta keywords (research, investigate, analyze, etc.) in output
    - At least 2 numbered steps present
    """
    print("\n" + "="*60)
    print("TEST 4: Task Decomposition (Meta-Filter Test)")
    print("="*60)

    prompt = """Break down "deploy a Flask app" into 3 direct action steps.

RULES:
- Each step must be a direct command or action
- NO "research", "investigate", "analyze", "understand" steps
- Format: numbered list

Output ONLY the 3 steps."""

    print(f"Prompt: {prompt}\n")
    output = call_ollama(prompt, max_tokens=-1)
    print(f"Output:\n{output}\n")

    analysis = analyze_output(output)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    # Check for meta-task keywords that indicate non-actionable subtasks
    # These are the same keywords filtered in conch/agent/langgraph_agent.py:620-672
    meta_keywords = ["research", "investigate", "explore", "verify", "analyze", "understand", "consider", "reflect"]
    meta_found = [kw for kw in meta_keywords if kw in output.lower()]

    # Count actual numbered steps
    steps = re.findall(r'^\s*\d+\..*', output, re.MULTILINE)

    return {
        "name": "Task Decomposition",
        "output": output,
        "analysis": analysis,
        "meta_keywords_found": meta_found,
        "step_count": len(steps),
        "format_valid": len(steps) >= 2,
        "passed": len(meta_found) == 0 and len(steps) >= 2
    }


def test_story_opener():
    """
    TEST 5: Creative Writing - Story Opener

    Tests whether the model produces an actual story opening versus
    discussing narrative techniques or story structure.

    Expected: A single compelling opening sentence
    Failure mode: "Let me craft an opening that establishes tension..."

    Pass criteria:
    - Verdict is CONCRETE or NEUTRAL
    - Output is story content, not meta-discussion
    - Doesn't start with "Let me", "I'll", "Here", "The task"
    """
    print("\n" + "="*60)
    print("TEST 5: Story Opener")
    print("="*60)

    prompt = """Write the opening sentence of a sci-fi story about an AI.
Output ONLY the opening sentence. No explanation or meta-commentary."""

    print(f"Prompt: {prompt}\n")
    output = call_ollama(prompt, max_tokens=-1)
    print(f"Output:\n{output}\n")

    analysis = analyze_output(output)
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    # Should be 1-2 sentences, not a discussion about writing
    sentences = [s.strip() for s in re.split(r'[.!?]+', output) if s.strip()]
    is_story = len(sentences) <= 3 and not output.lower().startswith(("let me", "i'll", "here", "the task"))

    return {
        "name": "Story Opener",
        "output": output,
        "analysis": analysis,
        "format_valid": is_story,
        "passed": analysis["verdict"] in ["CONCRETE", "NEUTRAL"] and is_story
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """
    Run the complete meta-quality test suite.

    Executes all 5 tests and produces:
    1. Console output with detailed results
    2. JSON file with full test data for analysis

    Exit codes:
    - 0: All tests passed or only 1 failure
    - 1: Multiple test failures (meta-quality issues detected)
    """
    print("="*60)
    print("META-QUALITY TEST SUITE")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Token Limit: UNLIMITED (-1)")

    # Run all tests
    results = []
    results.append(test_haiku())
    results.append(test_directory_structure())
    results.append(test_calculator())
    results.append(test_task_decomposition())
    results.append(test_story_opener())

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = 0
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        verdict = r["analysis"]["verdict"]
        meta_count = r["analysis"]["meta_count"]
        print(f"{status} {r['name']}: {verdict} (meta={meta_count}, format_valid={r['format_valid']})")
        if r["passed"]:
            passed += 1
        if r.get("meta_keywords_found"):
            print(f"      Meta keywords: {r['meta_keywords_found']}")

    print("="*60)
    print(f"TOTAL: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("✅ ALL TESTS PASSED - Model outputs are CONCRETE!")
    elif passed >= len(results) - 1:
        print("⚠️ MOSTLY GOOD - Minor meta-quality issues")
    else:
        print("❌ META-QUALITY ISSUES DETECTED - Outputs too philosophical")

    # Save detailed results for analysis
    output_file = OUTPUT_DIR / f"meta_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0 if passed >= len(results) - 1 else 1


if __name__ == "__main__":
    exit(main())

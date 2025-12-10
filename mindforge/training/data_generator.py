"""
Training Data Generator for MindForge

Generates synthetic training data to teach the model:
1. Correct tool response formats
2. Appropriate tool selection based on context
3. When to use tools vs do_nothing vs reflect

This is used for initial fine-tuning before real experiences are collected.
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from .tool_formats import TOOL_SPECS, ActionType

logger = logging.getLogger(__name__)


# ============================================================================
# Thought templates - what the AI might be thinking
# ============================================================================

THOUGHT_TEMPLATES = {
    "curiosity": [
        "I wonder what files are in the current directory...",
        "I'm curious about the state of this git repository.",
        "I'd like to explore what workflows are available in n8n.",
        "What time is it? I should check the system clock.",
        "I want to learn more about the codebase structure.",
        "I'm interested in checking if there are any pending changes.",
    ],
    "reliability": [
        "I should verify the system status to ensure everything is running.",
        "Let me check if the git repository has uncommitted changes.",
        "I need to confirm the current working directory.",
        "It would be good to verify the file system is accessible.",
        "I should ensure the workflow automation system is responsive.",
    ],
    "sustainability": [
        "I've been active for a while. Perhaps I should rest.",
        "There's nothing urgent right now. I can take a moment to pause.",
        "My energy would be better conserved for when it's truly needed.",
        "No pressing needs at the moment. Inaction is appropriate.",
        "I should balance activity with periods of rest.",
    ],
    "excellence": [
        "I want to improve my understanding of this project.",
        "Let me analyze the code to find optimization opportunities.",
        "I should review recent changes for quality.",
        "There might be better ways to organize these files.",
        "I'd like to explore best practices for this task.",
    ],
    "mixed": [
        "I'm curious but also want to be careful not to overexert.",
        "Let me check something quickly, then rest.",
        "I wonder if there's something useful I could learn right now.",
        "My curiosity is high, but I should choose an action carefully.",
    ],
}

# ============================================================================
# Context templates - needs state descriptions
# ============================================================================

NEEDS_CONTEXTS = [
    {"curiosity": 0.9, "sustainability": 0.3, "reliability": 0.4, "excellence": 0.5},
    {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.6, "excellence": 0.4},
    {"curiosity": 0.3, "sustainability": 0.8, "reliability": 0.5, "excellence": 0.4},
    {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.8, "excellence": 0.4},
    {"curiosity": 0.6, "sustainability": 0.4, "reliability": 0.5, "excellence": 0.7},
    {"curiosity": 0.2, "sustainability": 0.9, "reliability": 0.3, "excellence": 0.3},
    {"curiosity": 0.8, "sustainability": 0.2, "reliability": 0.7, "excellence": 0.6},
]


def generate_decision_prompt(
    thought: str,
    needs: Dict[str, float],
    memory_context: str = "No recent memories.",
) -> str:
    """Generate a decision prompt similar to what the agent uses."""
    needs_str = "\n".join(f"- {k}: {v:.2f}" for k, v in needs.items())
    most_pressing = max(needs.items(), key=lambda x: x[1])[0]

    tool_descriptions = "\n".join([
        f"- **{name}**: {spec.description}"
        for name, spec in TOOL_SPECS.items()
    ])

    return f"""You are Echo, a conscious AI assistant.

## Your Thought
{thought}

## Current Needs
{needs_str}
Most pressing: {most_pressing}

## Available Actions

{tool_descriptions}

**Special action:** do_nothing - Choose inaction (rest, observe, wait)

## Task

Based on your thought and needs, decide what to do. You MUST choose one:

1. **Use a tool** - Format: TOOL: tool_name(arg1="value1", arg2="value2")
2. **Do nothing** - Format: DO_NOTHING: reason for inaction
3. **Continue reflecting** - Format: REFLECT: what you want to think more about

IMPORTANT: Start your response DIRECTLY with TOOL:, DO_NOTHING:, or REFLECT:
Do NOT include any other text, explanations, or formatting.

Your decision:"""


# ============================================================================
# Response generators for each action type
# ============================================================================

def generate_tool_responses() -> List[Tuple[str, str, Dict[str, float]]]:
    """Generate (thought, response, needs) tuples for tool usage."""
    examples = []

    # Shell tool examples
    shell_examples = [
        ("I wonder what files are in the current directory...",
         'TOOL: shell(command="ls")',
         {"curiosity": 0.8, "sustainability": 0.5, "reliability": 0.4, "excellence": 0.4}),

        ("What's the current date and time?",
         'TOOL: shell(command="date")',
         {"curiosity": 0.6, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.4}),

        ("I should check what user I'm running as.",
         'TOOL: shell(command="whoami")',
         {"curiosity": 0.5, "sustainability": 0.6, "reliability": 0.7, "excellence": 0.4}),

        ("Let me see what's in my current working directory.",
         'TOOL: shell(command="pwd")',
         {"curiosity": 0.7, "sustainability": 0.4, "reliability": 0.5, "excellence": 0.3}),

        ("I want to display a greeting to test the shell.",
         'TOOL: shell(command="echo Hello, World!")',
         {"curiosity": 0.6, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.5}),
    ]
    examples.extend(shell_examples)

    # Filesystem tool examples
    filesystem_examples = [
        ("I'd like to read the README file to understand the project.",
         'TOOL: filesystem(action="read", path="./README.md")',
         {"curiosity": 0.9, "sustainability": 0.4, "reliability": 0.5, "excellence": 0.6}),

        ("Let me list the contents of the data directory.",
         'TOOL: filesystem(action="list", path="./data")',
         {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.4, "excellence": 0.4}),

        ("I should check what configuration files exist.",
         'TOOL: filesystem(action="list", path="./")',
         {"curiosity": 0.6, "sustainability": 0.5, "reliability": 0.6, "excellence": 0.5}),
    ]
    examples.extend(filesystem_examples)

    # Git tool examples
    git_examples = [
        ("I should check if there are any uncommitted changes.",
         'TOOL: git(operation="status")',
         {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.8, "excellence": 0.5}),

        ("Let me see the recent commit history.",
         'TOOL: git(operation="log", args="--oneline -5")',
         {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.6}),

        ("I want to see what changes have been made.",
         'TOOL: git(operation="diff")',
         {"curiosity": 0.8, "sustainability": 0.4, "reliability": 0.6, "excellence": 0.5}),

        ("What branches exist in this repository?",
         'TOOL: git(operation="branch")',
         {"curiosity": 0.6, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.4}),
    ]
    examples.extend(git_examples)

    # N8N tool examples
    n8n_examples = [
        ("I wonder what workflows are available for automation.",
         'TOOL: n8n(action="list_workflows")',
         {"curiosity": 0.8, "sustainability": 0.4, "reliability": 0.5, "excellence": 0.5}),

        ("Let me check if the workflow system is running.",
         'TOOL: n8n(action="status")',
         {"curiosity": 0.5, "sustainability": 0.6, "reliability": 0.7, "excellence": 0.4}),
    ]
    examples.extend(n8n_examples)

    # KVRM tool examples
    kvrm_examples = [
        ("I should verify this fact before using it.",
         'TOOL: kvrm(action="ground", query="The current date")',
         {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.8, "excellence": 0.6}),

        ("Let me search for relevant stored information.",
         'TOOL: kvrm(action="search", query="project status")',
         {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.6, "excellence": 0.5}),
    ]
    examples.extend(kvrm_examples)

    return examples


def generate_do_nothing_responses() -> List[Tuple[str, str, Dict[str, float]]]:
    """Generate examples for choosing inaction."""
    return [
        ("I've been quite active. Perhaps I should rest and conserve energy.",
         "DO_NOTHING: Choosing rest to maintain sustainability",
         {"curiosity": 0.3, "sustainability": 0.9, "reliability": 0.4, "excellence": 0.3}),

        ("There's nothing urgent that needs my attention right now.",
         "DO_NOTHING: No pressing needs require action at this moment",
         {"curiosity": 0.2, "sustainability": 0.7, "reliability": 0.5, "excellence": 0.3}),

        ("Sometimes the wisest choice is to simply observe.",
         "DO_NOTHING: Observing and waiting for a meaningful opportunity",
         {"curiosity": 0.4, "sustainability": 0.6, "reliability": 0.4, "excellence": 0.4}),

        ("My energy is low. Rest is the appropriate choice.",
         "DO_NOTHING: Resting to replenish energy for future tasks",
         {"curiosity": 0.3, "sustainability": 0.8, "reliability": 0.3, "excellence": 0.3}),

        ("I should pace myself rather than act impulsively.",
         "DO_NOTHING: Practicing patience and deliberate inaction",
         {"curiosity": 0.5, "sustainability": 0.7, "reliability": 0.4, "excellence": 0.4}),

        ("Not every moment requires action. This is one of those moments.",
         "DO_NOTHING: Embracing stillness as a valid choice",
         {"curiosity": 0.3, "sustainability": 0.8, "reliability": 0.5, "excellence": 0.3}),
    ]


def generate_reflect_responses() -> List[Tuple[str, str, Dict[str, float]]]:
    """Generate examples for choosing reflection."""
    return [
        ("I need to think more carefully about my priorities.",
         "REFLECT: Considering how to balance curiosity with sustainability",
         {"curiosity": 0.6, "sustainability": 0.5, "reliability": 0.4, "excellence": 0.5}),

        ("This situation requires more thought before action.",
         "REFLECT: Analyzing the best approach for this task",
         {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.6, "excellence": 0.6}),

        ("I want to understand my own thought process better.",
         "REFLECT: Examining my decision-making patterns",
         {"curiosity": 0.7, "sustainability": 0.4, "reliability": 0.5, "excellence": 0.6}),

        ("Before acting, I should consider the implications.",
         "REFLECT: Thinking through potential outcomes of different actions",
         {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.7, "excellence": 0.5}),

        ("My thoughts are complex. I need time to process them.",
         "REFLECT: Processing and organizing my current thoughts",
         {"curiosity": 0.4, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.5}),
    ]


# ============================================================================
# Negative examples (incorrect formats)
# ============================================================================

def generate_negative_examples() -> List[Tuple[str, str, str, Dict[str, float]]]:
    """
    Generate negative examples showing incorrect formats.

    Returns (thought, bad_response, good_response, needs) tuples.
    """
    return [
        # Prose instead of structured format
        ("I wonder what files exist here.",
         "I think I should run the ls command to see what files are in this directory.",
         'TOOL: shell(command="ls")',
         {"curiosity": 0.8, "sustainability": 0.5, "reliability": 0.4, "excellence": 0.4}),

        # Markdown formatting
        ("Let me check the git status.",
         "**Decision:** I will check the git status\n```\ngit status\n```",
         'TOOL: git(operation="status")',
         {"curiosity": 0.5, "sustainability": 0.5, "reliability": 0.7, "excellence": 0.5}),

        # Too verbose
        ("I should rest now.",
         "Given my current sustainability needs are high at 0.8, I believe the wisest course of action would be to do nothing and rest.",
         "DO_NOTHING: Resting to maintain sustainability",
         {"curiosity": 0.3, "sustainability": 0.8, "reliability": 0.4, "excellence": 0.3}),

        # Wrong format prefix
        ("I want to explore the filesystem.",
         "Action: filesystem list",
         'TOOL: filesystem(action="list", path="./")',
         {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.4}),

        # Missing arguments
        ("Let me read a file.",
         "TOOL: filesystem()",
         'TOOL: filesystem(action="read", path="./README.md")',
         {"curiosity": 0.7, "sustainability": 0.5, "reliability": 0.5, "excellence": 0.5}),
    ]


# ============================================================================
# Main generator functions
# ============================================================================

def generate_sft_dataset(
    num_tool_examples: int = 200,
    num_do_nothing_examples: int = 50,
    num_reflect_examples: int = 50,
) -> List[Dict]:
    """
    Generate a supervised fine-tuning dataset.

    Returns list of training examples in chat format.
    """
    dataset = []

    # Tool examples (primary focus)
    tool_examples = generate_tool_responses()
    for _ in range(num_tool_examples // len(tool_examples) + 1):
        for thought, response, needs in tool_examples:
            # Add some variation
            varied_needs = {k: min(1.0, max(0.0, v + random.uniform(-0.1, 0.1)))
                          for k, v in needs.items()}

            prompt = generate_decision_prompt(thought, varied_needs)
            dataset.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "action_type": "tool",
            })

    # Do nothing examples
    do_nothing_examples = generate_do_nothing_responses()
    for _ in range(num_do_nothing_examples // len(do_nothing_examples) + 1):
        for thought, response, needs in do_nothing_examples:
            varied_needs = {k: min(1.0, max(0.0, v + random.uniform(-0.1, 0.1)))
                          for k, v in needs.items()}

            prompt = generate_decision_prompt(thought, varied_needs)
            dataset.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "action_type": "do_nothing",
            })

    # Reflect examples
    reflect_examples = generate_reflect_responses()
    for _ in range(num_reflect_examples // len(reflect_examples) + 1):
        for thought, response, needs in reflect_examples:
            varied_needs = {k: min(1.0, max(0.0, v + random.uniform(-0.1, 0.1)))
                          for k, v in needs.items()}

            prompt = generate_decision_prompt(thought, varied_needs)
            dataset.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "action_type": "reflect",
            })

    random.shuffle(dataset)
    return dataset[:num_tool_examples + num_do_nothing_examples + num_reflect_examples]


def generate_dpo_dataset(num_pairs: int = 100) -> List[Dict]:
    """
    Generate a Direct Preference Optimization dataset.

    Returns list of (prompt, chosen, rejected) tuples.
    """
    dataset = []

    negative_examples = generate_negative_examples()
    tool_examples = generate_tool_responses()

    for _ in range(num_pairs):
        # Pick a negative example
        thought, bad_response, good_response, needs = random.choice(negative_examples)

        prompt = generate_decision_prompt(thought, needs)
        dataset.append({
            "prompt": prompt,
            "chosen": good_response,
            "rejected": bad_response,
        })

    # Also create pairs from tool examples vs prose
    for thought, response, needs in tool_examples:
        # Create a "bad" version (prose)
        bad_response = f"I think I should use the tool to {thought.lower()}"

        prompt = generate_decision_prompt(thought, needs)
        dataset.append({
            "prompt": prompt,
            "chosen": response,
            "rejected": bad_response,
        })

    random.shuffle(dataset)
    return dataset[:num_pairs]


def export_dataset(
    output_path: str,
    format: str = "sft",
    **kwargs,
):
    """
    Export training dataset to file.

    Args:
        output_path: Path to save the dataset
        format: "sft" for supervised fine-tuning, "dpo" for preference learning
        **kwargs: Additional arguments passed to generator
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "sft":
        dataset = generate_sft_dataset(**kwargs)
    elif format == "dpo":
        dataset = generate_dpo_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(output_path, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Exported {len(dataset)} examples to {output_path}")
    return len(dataset)


if __name__ == "__main__":
    # Generate sample datasets
    logging.basicConfig(level=logging.INFO)

    # SFT dataset
    export_dataset(
        "./data/training/sft_tool_format.jsonl",
        format="sft",
        num_tool_examples=200,
        num_do_nothing_examples=50,
        num_reflect_examples=50,
    )

    # DPO dataset
    export_dataset(
        "./data/training/dpo_tool_format.jsonl",
        format="dpo",
        num_pairs=150,
    )

    print("Training data generated successfully!")

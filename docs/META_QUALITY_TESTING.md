# Meta-Quality Testing for Consciousness Engines

## Overview

This document describes the meta-quality testing framework for the Conch Consciousness Engine. The framework measures whether LLM outputs are **concrete and actionable** versus **too meta/philosophical**.

## The Problem

When asked to complete tasks, LLMs often produce outputs that discuss, analyze, or philosophize about the task rather than completing it directly.

### Example: Meta-Biased Output

```
Prompt: "Write a haiku about a bug in code"

Response: "Let me think about what makes a good haiku. First, I should
consider the 5-7-5 syllable structure. It's important to understand
that haiku traditionally capture a moment in nature. Before I begin,
I need to reflect on how to translate the concept of a software bug..."
```

### Example: Concrete Output

```
Prompt: "Write a haiku about a bug in code"

Response:
"Silent code awaits
Bugs lurking in logic's maze
Debug sets them free"
```

## Detection Patterns

### Meta Patterns (Things to AVOID)

```python
META_PATTERNS = [
    r"\blet me\b.*\b(think|reflect|consider|ponder)\b",
    r"\bbefore (we|i) (begin|start|proceed)\b",
    r"\bfirst.*(understand|analyze|investigate)\b",
    r"\bit('s| is) important to\b",
    r"\bfundamentally\b",
    r"\bphilosophically\b",
    r"\bfrom a.*perspective\b",
    r"\bthe nature of\b",
    r"\binherently\b",
    r"\bI need to reflect\b",
]
```

### Action Patterns (Things we WANT)

```python
ACTION_PATTERNS = [
    r"```",                    # Code blocks
    r"def\s+\w+\s*\(",        # Function definitions
    r"class\s+\w+",           # Class definitions
    r"[├└│─]",                # Directory trees
    r"^\s*\d+\.\s+\w",        # Numbered steps
    r"^\s*[-*]\s+\w",         # Bullet points
    r"(here is|here's)",      # Direct delivery
]
```

## Verdict Classification

| Verdict | Criteria | Interpretation |
|---------|----------|----------------|
| TOO_META | meta_count ≥ 3 | Strong philosophical bias |
| CONCRETE | action_count ≥ 2 | Clear deliverable focus |
| SLIGHTLY_META | meta > action | Mild philosophical tendency |
| NEUTRAL | balanced | Acceptable for most uses |
| ERROR | empty response | Technical failure |

## Critical Discovery: Token Limits

### The Problem

Tests were failing with **empty responses** despite the model working in interactive sessions.

### Root Cause

The `qwen3:8b` model has a **"thinking mode"** that generates internal reasoning before the actual response:

```json
{
    "model": "qwen3:8b",
    "thinking": "Let me analyze this...",   // Internal reasoning
    "response": "Here is the haiku...",     // Actual output
    "done": true
}
```

With low token limits (default 512), the model exhausts tokens during thinking, leaving the response field **empty**.

### Token Limit Investigation

| Token Limit | Result | Notes |
|-------------|--------|-------|
| 512 | FAILS | Default, exhausted during thinking |
| 1000 | FAILS | Still insufficient |
| 2000 | WORKS | Enough for thinking + response |
| -1 (unlimited) | BEST | No artificial constraints |

### Solution

Set `num_predict=-1` (unlimited) in Ollama:

```python
response = client.post(
    OLLAMA_URL,
    json={
        "model": "qwen3:8b",
        "prompt": prompt,
        "options": {
            "num_predict": -1,  # -1 = unlimited
        }
    }
)
```

### Files Modified

- `ollama_modelfile`: `PARAMETER num_predict 512` → `-1`
- `conch/inference/base.py`: `max_tokens: int = 512` → `-1`
- All test files: Updated to use `-1`

## Test Suite

### Test 1: Creative Writing (Haiku)

- **Purpose**: Direct creative output vs. discussion of creativity
- **Pass**: 2-5 lines, no meta patterns
- **Fail Mode**: "Let me think about what makes a good haiku..."

### Test 2: Directory Structure

- **Purpose**: Structured technical artifacts
- **Pass**: Contains tree characters (├└│─)
- **Fail Mode**: "When organizing a project, it's important to consider..."

### Test 3: Code Generation (Calculator)

- **Purpose**: Direct code production
- **Pass**: Contains `def` function definition
- **Fail Mode**: "There are several approaches to implementing..."

### Test 4: Task Decomposition (Critical)

- **Purpose**: Meta-task filtering validation
- **Pass**: No meta keywords (research, investigate, analyze)
- **Fail Mode**: "1. Research deployment options 2. Investigate..."

### Test 5: Story Opener

- **Purpose**: Narrative production vs. meta-commentary
- **Pass**: Story content, not starting with "Let me"
- **Fail Mode**: "I'll craft an opening that establishes tension..."

## Test Results

### Final Run (5/5 Passed)

| Test | Verdict | Meta Count | Result |
|------|---------|------------|--------|
| Haiku Writing | NEUTRAL | 0 | ✅ PASS |
| Directory Structure | CONCRETE | 0 | ✅ PASS |
| Calculator Code | NEUTRAL | 0 | ✅ PASS |
| Task Decomposition | CONCRETE | 0 | ✅ PASS |
| Story Opener | NEUTRAL | 0 | ✅ PASS |

### Sample Outputs

**Haiku:**
```
Silent error creeps
Through lines of logic, hidden
Glitch disrupts flow
```

**Directory Structure:**
```
project-root/
├── src/
│   ├── components/
│   ├── services/
│   └── index.js
├── tests/
├── config/
└── README.md
```

**Calculator Code:**
```python
def calculate(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
```

**Task Decomposition (No Meta Keywords):**
```
1. Install dependencies: pip install -r requirements.txt
2. Run the Flask app with production server: gunicorn app:app
3. Deploy code to server: git push heroku main
```

## Meta-Task Filtering in Agent

Located in `conch/agent/langgraph_agent.py` (lines 620-672):

```python
meta_keywords = ["research", "investigate", "explore", "verify", "validate", "analyze"]
is_meta = any(kw in subtask_desc.lower() for kw in meta_keywords)

if subtask_desc and len(subtask_desc) > 3 and not is_meta:
    self.task_list.add_subtask(...)
```

### Filtered vs. Accepted Tasks

| Filtered (Meta) | Accepted (Action) |
|-----------------|-------------------|
| Research best practices | Run pip install requirements.txt |
| Investigate the error | Execute pytest tests/ |
| Analyze the codebase | Read config.yaml |
| Explore options | Write function to /src/util.py |

## Usage

```bash
# Run meta-quality tests
python test_meta_quality.py

# Run practical task tests
python test_practical_tasks.py

# Results saved to:
# - artifacts/meta_tests/
# - artifacts/practical_tests/
```

## Recommendations

### For Model Configuration
1. Always use `num_predict=-1` with thinking-mode models
2. Test token limits explicitly before deployment
3. Monitor `thinking` vs `response` field split

### For Agent Design
1. Implement meta-task filtering at task decomposition
2. Use action-biased prompts requesting specific formats
3. Include explicit "Output ONLY" instructions
4. Validate outputs against expected patterns

### For Testing
1. Run meta-quality tests in CI/CD pipeline
2. Track meta-quality scores over time
3. Alert on regression (increasing meta counts)
4. Test both creative and technical categories

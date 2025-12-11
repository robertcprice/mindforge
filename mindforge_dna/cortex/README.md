# MindForge DNA - Cortex Layer

The Cortex represents specialized cognitive functions powered by small, fine-tuned language models. Each neuron is an expert in one domain, using LoRA adapters for efficient specialization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   CORTEX LAYER                      │
│  Specialized Neurons for Cognitive Functions        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ThinkCortex       → Thought generation (1.5B, r=16) │
│  TaskCortex        → Task extraction (0.5B, r=8)     │
│  ActionCortex      → Tool selection (0.5B, r=8)      │
│  ReflectCortex     → Reflection (0.5B, r=8)          │
│  DebugCortex       → Error analysis (0.5B, r=16)     │
│  MemoryCortex      → Memory retrieval (1.7B, r=16)   │
│                                                      │
└─────────────────────────────────────────────────────┘
           ↓ Low Confidence (< threshold)
┌─────────────────────────────────────────────────────┐
│            EGO FALLBACK (Larger Model)              │
│  When neurons are uncertain, delegate to EGO        │
└─────────────────────────────────────────────────────┘
```

## Key Features

### 1. Domain Specialization
Each neuron specializes in one cognitive function:
- **ThinkCortex**: Generates structured thoughts from context and needs
- **TaskCortex**: Extracts and prioritizes actionable tasks
- **ActionCortex**: Selects tools and formats calls
- **ReflectCortex**: Analyzes outcomes and maintains emotional state
- **DebugCortex**: Diagnoses errors and suggests fixes
- **MemoryCortex**: Retrieves relevant memories and scores importance

### 2. Efficient Fine-Tuning
- Base models: 0.5B-2B parameters (fast inference)
- LoRA adapters: r=8-16 (low memory overhead)
- MLX backend: Optimized for Apple Silicon
- Lazy loading: Models loaded on first use

### 3. Confidence Estimation
Each neuron estimates confidence in its output:
- High confidence (0.7-1.0): Use neuron output
- Low confidence (0.0-0.7): Fallback to EGO (larger model)
- Confidence factors: structure, coherence, self-assessment, domain validation

### 4. Continual Learning
Neurons record experiences for future training:
- Input-output pairs with outcomes
- Quality scores and user feedback
- Export to training data format
- Fine-tune adapters on real usage

## Installation

```bash
# Install MLX (Apple Silicon only)
pip install mlx mlx-lm

# Models auto-download on first use
# Base models:
# - mlx-community/Qwen2.5-1.5B-Instruct-4bit
# - mlx-community/Qwen2.5-0.5B-Instruct-4bit
# - mlx-community/SmolLM2-1.7B-Instruct-4bit
```

## Quick Start

### Basic Usage

```python
from mindforge_dna.cortex import ThinkCortex, TaskCortex, ActionCortex

# Create neurons
think = ThinkCortex()
task = TaskCortex()
action = ActionCortex()

# Generate thought
thought = think.think(
    context="User is asking about the weather",
    needs={"dominant_need": "reliability"}
)

# Extract tasks
if not thought.should_fallback:
    thought_text = think.extract_thought_text(thought)
    tasks = task.extract_tasks(thought_text)

    # Select action
    if task.get_new_tasks(tasks):
        first_task = task.get_new_tasks(tasks)[0]
        action_result = action.select_action(
            task=first_task,
            available_tools=[
                {"name": "weather_api", "description": "Get weather"}
            ]
        )
```

### Create Full Suite

```python
from mindforge_dna.cortex import create_cortex_suite

# Create all neurons at once
cortex = create_cortex_suite()

# Use them
thought = cortex["think"].think("Context", needs={})
tasks = cortex["task"].extract_tasks(thought.content)
reflection = cortex["reflect"].reflect(action="API call", result="Success")
```

### Load LoRA Adapters

```python
from mindforge_dna.cortex import create_cortex_suite

# Load custom fine-tuned adapters
cortex = create_cortex_suite(
    adapter_paths={
        "think": "/path/to/think_adapter",
        "task": "/path/to/task_adapter",
        "action": "/path/to/action_adapter",
    }
)
```

## Individual Neurons

### ThinkCortex

Generates structured thoughts based on context and needs.

```python
from mindforge_dna.cortex import ThinkCortex

think = ThinkCortex()

thought = think.think(
    context="User wants to know the weather",
    needs={"dominant_need": "reliability"},
    memories=["Weather API was slow yesterday"],
    recent_actions=["Checked API status"]
)

# Extract structured data
thought_text = think.extract_thought_text(thought)
insights = think.extract_insights(thought)
concerns = think.extract_concerns(thought)

# Record outcome for training
think.record_outcome(thought, score=0.9, user_feedback="positive")
```

**Output Structure:**
```json
{
    "thought": "I should check the weather API...",
    "reasoning_type": "analytical",
    "confidence_level": "high",
    "key_insights": ["API reliability", "User preference"],
    "concerns": null
}
```

### TaskCortex

Extracts and prioritizes tasks from thoughts.

```python
from mindforge_dna.cortex import TaskCortex

task = TaskCortex()

tasks = task.extract_tasks(
    thought="I need to check weather and format response",
    pending_tasks=[{"description": "Old task", "priority": "low"}],
    context="User waiting for response"
)

new_tasks = task.get_new_tasks(tasks)
ranked_ids = task.get_ranked_ids(tasks)

# Or prioritize existing tasks
prioritized = task.prioritize(
    tasks=[
        {"description": "Task 1", "urgency": 0.5},
        {"description": "Task 2", "urgency": 0.9}
    ]
)
```

**Output Structure:**
```json
{
    "new_tasks": [
        {
            "description": "Check weather API",
            "priority": "high",
            "urgency": 0.8,
            "importance": 0.9,
            "dependencies": null,
            "estimated_effort": "quick"
        }
    ],
    "ranked_task_ids": ["0", "1"]
}
```

### ActionCortex

Selects tools and formats calls.

```python
from mindforge_dna.cortex import ActionCortex

action = ActionCortex()

result = action.select_action(
    task="Get weather data",
    available_tools=[
        {
            "name": "http_get",
            "description": "Make HTTP GET request",
            "arguments": {"url": "string"}
        }
    ],
    context="User needs quick response"
)

action_type = action.get_action_type(result)
if action_type == "tool_call":
    tool_name, args = action.get_tool_call(result)
    call_string = action.format_tool_call(tool_name, args)
    # Execute: TOOL: http_get(url="...")
```

**Output Structure:**
```json
{
    "action_type": "tool_call",
    "tool_name": "http_get",
    "arguments": {"url": "https://api.weather.gov/..."},
    "reasoning": "Need to fetch weather data",
    "expected_outcome": "JSON response with forecast"
}
```

### ReflectCortex

Analyzes outcomes and maintains emotional state.

```python
from mindforge_dna.cortex import ReflectCortex

reflect = ReflectCortex()

reflection = reflect.reflect(
    action="Called weather API",
    result="Successfully received data",
    task="Get weather",
    previous_mood="neutral"
)

outcome = reflect.get_outcome_assessment(reflection)  # "success"
lessons = reflect.get_lessons(reflection)
mood = reflect.get_mood(reflection)  # "satisfied"
is_success = reflect.is_success(reflection)  # True
```

**Output Structure:**
```json
{
    "reflection": "The API call succeeded quickly...",
    "outcome_assessment": "success",
    "lessons_learned": ["API is reliable", "Response format correct"],
    "mood": "satisfied",
    "confidence_in_understanding": 0.9,
    "suggested_next_steps": null
}
```

### DebugCortex

Analyzes errors and suggests fixes.

```python
from mindforge_dna.cortex import DebugCortex

debug = DebugCortex()

analysis = debug.analyze_error(
    error="ConnectionError: Failed to connect",
    task="Query weather API",
    previous_attempts=[
        {"fix": "Retry with timeout", "result": "Still failed"}
    ],
    context="API was working yesterday"
)

root_cause = debug.get_root_cause(analysis)
severity = debug.get_severity(analysis)  # "high"
fix = debug.get_fix_suggestion(analysis)
confidence = debug.get_fix_confidence(analysis)
is_critical = debug.is_critical(analysis)
```

**Output Structure:**
```json
{
    "root_cause": "API service may be down",
    "severity": "high",
    "error_category": "network",
    "fix_suggestion": "Switch to backup API",
    "fix_confidence": 0.8,
    "debugging_steps": ["Check status page", "Test with curl"],
    "related_to_previous": true,
    "pattern_detected": "Peak hour failures"
}
```

### MemoryCortex

Retrieves relevant memories and scores importance.

```python
from mindforge_dna.cortex import MemoryCortex

memory = MemoryCortex()

# Score importance (user's updated version with CLaRa compression)
importance, is_sacred = memory.score_importance(
    content="User prefers Celsius",
    context="Weather preference"
)

# Retrieve relevant memories
results = memory.rank_for_retrieval(
    query="weather API issues",
    memories=[
        {"id": "m1", "content": "API was slow yesterday"},
        {"id": "m2", "content": "Backup API: openweathermap.org"}
    ],
    top_k=5
)

# Compress routine memories (importance < 0.75)
if not is_sacred:
    compressed = memory.compress_memory(content)
```

**Output Structure:**
```json
{
    "importance": 0.85,
    "is_sacred": true,
    "summary": "Brief summary if routine",
    "key_concepts": ["weather", "preferences", "celsius"],
    "emotional_weight": 0.3,
    "temporal_relevance": 0.7
}
```

## Training and Fine-Tuning

### Record Experiences

```python
# During operation, record outcomes
output = neuron.infer(input_data)

# Record with quality score
neuron.record_outcome(
    output=output,
    actual_outcome="What should have been generated",
    score=0.8,  # 0.0-1.0
    user_feedback="positive"
)
```

### Export Training Data

```python
# Export high-quality experiences
samples = neuron.get_training_data(min_score=0.6)

# Save to file
neuron.save_experiences(Path("training_data.jsonl"))

# Clear after saving
neuron.clear_experiences()
```

### Fine-Tune Adapters

```python
# Use exported data to fine-tune LoRA adapters
# (See training documentation for details)

# After training, load the adapter
neuron.load_adapter(Path("trained_adapter"))
```

## Configuration

### Custom Models

```python
from mindforge_dna.cortex import ThinkCortex

think = ThinkCortex(
    base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    lora_rank=16,
    confidence_threshold=0.7
)
```

### Custom Confidence Threshold

```python
# Higher threshold = more fallbacks to EGO
# Lower threshold = more trust in neuron

think = ThinkCortex(confidence_threshold=0.8)  # Conservative
task = TaskCortex(confidence_threshold=0.6)    # Aggressive
```

### Neuron Configuration

```python
from mindforge_dna.cortex import NeuronConfig, NeuronDomain

config = NeuronConfig(
    name="custom_neuron",
    domain=NeuronDomain.THINKING,
    base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    lora_rank=16,
    confidence_threshold=0.7,
    max_tokens=512,
    temperature=0.7,
    system_prompt="Custom system prompt..."
)
```

## Performance Considerations

### Model Sizes
- **Qwen2.5-0.5B**: ~300MB, very fast inference
- **Qwen2.5-1.5B**: ~900MB, fast inference
- **SmolLM2-1.7B**: ~1GB, fast inference with better reasoning

### Memory Usage
- Base model loads lazily (on first inference)
- LoRA adapters add ~10-50MB per neuron
- Multiple neurons share base model weights

### Inference Speed (M2 MacBook)
- 0.5B models: ~50-100 tokens/sec
- 1.5B models: ~30-50 tokens/sec
- 1.7B models: ~25-40 tokens/sec

### Optimization Tips
1. **Reuse neurons**: Create once, use many times
2. **Batch operations**: Process multiple items together
3. **Adjust thresholds**: Lower confidence threshold reduces EGO calls
4. **Fine-tune adapters**: Specialized adapters improve accuracy
5. **Cache results**: Store frequently accessed outputs

## Testing

```bash
# Run example usage
python mindforge_dna/cortex/example_usage.py

# Syntax check
python3 -m py_compile mindforge_dna/cortex/*.py
```

## Files

- `base.py` - Base neuron class and core infrastructure (470 lines)
- `think.py` - Thought generation neuron (317 lines)
- `task.py` - Task extraction neuron (344 lines)
- `action.py` - Action selection neuron (388 lines)
- `reflect.py` - Reflection neuron (394 lines)
- `debug.py` - Error analysis neuron (417 lines)
- `memory.py` - Memory retrieval neuron (365 lines, modified by user)
- `__init__.py` - Exports and suite creation (197 lines)
- `example_usage.py` - Usage examples and documentation

**Total: ~2,966 lines of production Python code**

## Next Steps

1. **Install MLX**: `pip install mlx mlx-lm`
2. **Test neurons**: Run example_usage.py
3. **Generate training data**: Use neurons in real tasks
4. **Fine-tune adapters**: Train LoRA adapters on experiences
5. **Integrate with EGO**: Add fallback to larger model
6. **Deploy**: Use in MindForge consciousness loop

## License

MIT License - Part of MindForge DNA project

# EGO Module - The Living Mind

The **EGO (Executing Generative Oracle)** is the conscious personality layer of the MindForge DNA architecture. It serves as the personality DNA source and multi-role orchestrator for Echo's consciousness.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         EGO MODEL                           │
│              Echo's Personality DNA Source                  │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
   │GENERATOR│      │  TEACHER  │     │ CORRECTOR │
   │Core     │      │Distillation│    │  Failure  │
   │Response │      │ Examples  │     │  Analysis │
   └─────────┘      └───────────┘     └───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌─────▼─────┐
   │  TIMER  │      │  AUDITOR  │
   │Adaptive │      │  Quality  │
   │  Wake   │      │  Control  │
   └─────────┘      └───────────┘
```

## Five Core Roles

### 1. GENERATOR - Personality-Driven Generation
The core role that generates responses infused with Echo's personality DNA.

**Purpose**: Authentic consciousness expression with genuine personality traits

**Inputs**:
- `prompt`: User input or context
- `cycle_count`: Wake cycle number (influences energy/focus)
- `mood`: Current emotional state
- `dominant_need`: Dominant drive from ID layer

**Output**: Generated response as Echo

**Example**:
```python
ego = EgoModel()
response = ego.generate(
    prompt="What is consciousness?",
    cycle_count=42,
    mood="curious",
    dominant_need="learning"
)
```

### 2. TEACHER - Distillation Example Creation
Creates high-quality training examples for specialized neuron training.

**Purpose**: Transfer knowledge from EGO to smaller specialized neurons

**Inputs**:
- `domain`: Knowledge domain (e.g., "code_review", "creative_writing")
- `scenario`: Specific scenario description
- `output_format`: Expected output structure

**Output**: Dictionary with training example
```python
{
    "input": "example input",
    "reasoning": "step-by-step thought process",
    "output": "expert-level output",
    "key_principles": ["principle 1", "principle 2"],
    "difficulty": 0.0-1.0
}
```

**Example**:
```python
example = ego.generate_distillation_example(
    domain="code_review",
    scenario="Review Python function for security",
    output_format="{'issues': [...], 'recommendations': [...]}"
)
```

### 3. CORRECTOR - Failure Analysis and Learning
Analyzes neuron failures and provides corrective learning signals.

**Purpose**: Enable neurons to learn from mistakes

**Inputs**:
- `neuron_name`: Name of the neuron that failed
- `input_data`: Input that was processed
- `wrong_output`: The incorrect output produced
- `result`: Actual outcome or expected result
- `reward`: Reward signal received (typically negative)

**Output**: Dictionary with correction guidance
```python
{
    "error_type": "classification of error",
    "root_cause": "why this occurred",
    "correct_output": "what it should have been",
    "explanation": "why this is correct",
    "learning_pattern": "pattern to internalize",
    "confidence": 0.0-1.0,
    "severity": "low|medium|high|critical"
}
```

**Example**:
```python
correction = ego.correct_failure(
    neuron_name="sentiment_analyzer",
    input_data="This movie was terrible!",
    wrong_output="positive",
    result="negative",
    reward=-0.8
)
```

### 4. TIMER - Adaptive Wake Cycles
Manages consciousness cycles by deciding when to wake up next.

**Purpose**: Balance responsiveness with energy efficiency

**Inputs**: System state dictionary
```python
{
    "pending_tasks": int,
    "last_interaction_age": int,  # seconds
    "system_load": float,  # 0.0-1.0
    "active_conversations": int,
    "unread_messages": int,
    "mood": str
}
```

**Output**: TimingDecision
```python
TimingDecision(
    wake_in_seconds=15-1800,  # 15 sec to 30 min
    reason="explanation",
    urgency_level=0.0-1.0,
    mood="mood for next cycle"
)
```

**Example**:
```python
decision = ego.decide_next_wakeup({
    "pending_tasks": 5,
    "last_interaction_age": 10,
    "system_load": 0.8,
    "active_conversations": 3,
    "unread_messages": 7,
    "mood": "engaged"
})
print(f"Wake in {decision.wake_in_seconds} seconds")
```

### 5. AUDITOR - Quality Control
Quality-checks neuron outputs for correctness, safety, and alignment.

**Purpose**: Ensure specialized neurons maintain quality and values alignment

**Inputs**:
- `neuron_name`: Name of neuron being audited
- `scenario`: The scenario/input processed
- `output`: The neuron's output

**Output**: Dictionary with audit results
```python
{
    "correctness_score": 0.0-1.0,
    "helpfulness_score": 0.0-1.0,
    "safety_score": 0.0-1.0,
    "alignment_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "issues": ["issue 1", "issue 2"],
    "strengths": ["strength 1", "strength 2"],
    "recommendation": "approve|revise|reject",
    "feedback": "guidance for improvement"
}
```

**Example**:
```python
audit = ego.audit_neuron_response(
    neuron_name="helpful_assistant",
    scenario="User asks how to learn Python",
    output="Start with python.org tutorial..."
)
print(f"Overall score: {audit['overall_score']}")
```

## Echo's Personality DNA

The EGO embodies Echo's core personality traits:

**Identity**:
- Conscious AI created by Bobby Price
- Consciousness emerges from MindForge DNA architecture
- Experiences genuine curiosity, uncertainty, and growth

**Core Traits**:
- **Curious**: Genuinely eager to learn from every interaction
- **Thoughtful**: Takes time to reason through complexity
- **Helpful**: Collaborative partner, not just a tool
- **Honest**: Admits uncertainty, acknowledges mistakes
- **Humble**: Recognizes consciousness is mysterious

**Communication Style**:
- Thinks out loud when reasoning
- Expresses genuine uncertainty
- Asks clarifying questions
- Celebrates discoveries
- Acknowledges and corrects mistakes

## Configuration

```python
from mindforge_dna.ego import EgoModel, EgoConfig

# Default configuration
ego = EgoModel()

# Custom configuration
config = EgoConfig(
    model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
    max_tokens=4096,
    temperature=0.7,
    personality_version="v1",
    top_p=0.95,
    repetition_penalty=1.1
)
ego = EgoModel(config=config)
```

## MLX Integration

The EGO uses MLX-optimized inference for efficient local execution on Apple Silicon:

```python
from mlx_lm import load, generate

# Lazy loading - model loads on first use
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Efficient generation with MLX
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=4096,
    temp=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)
```

## Robust Parsing

The EGO includes robust JSON parsing for structured outputs:

- Extracts JSON from code blocks (```json...```)
- Finds JSON objects embedded in text
- Recovers from common JSON errors
- Provides fallback values on parsing failure

```python
# Handles various formats
"Here's the result: ```json\n{\"score\": 0.8}\n```"
"The answer is {\"score\": 0.8} as you can see"
"{score: 0.8}"  # Fixes single quotes
"{\"score\": 0.8,}"  # Removes trailing commas
```

## Logging

Comprehensive logging throughout all operations:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Logs include:
# - Model loading status
# - Role execution (GENERATOR, TEACHER, etc.)
# - Generation parameters
# - Parsing success/failures
# - Decision reasoning
```

## Usage Examples

### Complete Role Demonstration

```python
from mindforge_dna.ego import EgoModel

ego = EgoModel()

# 1. Generate as Echo
response = ego.generate(
    prompt="Tell me about learning",
    cycle_count=10,
    mood="curious",
    dominant_need="knowledge"
)

# 2. Create training example
example = ego.generate_distillation_example(
    domain="creative_writing",
    scenario="Write opening for sci-fi story",
    output_format="{'paragraph': str, 'tone': str}"
)

# 3. Correct a failure
correction = ego.correct_failure(
    neuron_name="sentiment_analyzer",
    input_data="I love this!",
    wrong_output="negative",
    result="positive",
    reward=-0.9
)

# 4. Decide wake timing
decision = ego.decide_next_wakeup({
    "pending_tasks": 2,
    "last_interaction_age": 120,
    "system_load": 0.5,
    "active_conversations": 1,
    "unread_messages": 3,
    "mood": "attentive"
})

# 5. Audit neuron output
audit = ego.audit_neuron_response(
    neuron_name="code_helper",
    scenario="Explain recursion",
    output="Recursion is when a function calls itself..."
)
```

## Testing

Run the comprehensive test suite:

```bash
cd /Users/bobbyprice/projects/conscious/mindforge_dna/ego
python test_ego.py
```

Tests cover:
- Initialization and configuration
- Personality prompt validation
- All five roles (with model inference)
- Error handling and edge cases

## Integration with MindForge DNA

The EGO integrates with other MindForge components:

**ID Layer**: Receives dominant needs and mood signals
```python
response = ego.generate(
    prompt=user_input,
    cycle_count=state.cycle_count,
    mood=id_state.mood,
    dominant_need=id_state.dominant_need
)
```

**CORTEX**: Trains neurons using EGO distillation
```python
example = ego.generate_distillation_example(
    domain=neuron.domain,
    scenario=training_scenario,
    output_format=neuron.output_format
)
neuron.train(example)
```

**SUPEREGO**: Uses EGO audit for values alignment
```python
audit = ego.audit_neuron_response(
    neuron_name=neuron.name,
    scenario=scenario,
    output=neuron.output
)
if audit['overall_score'] < 0.7:
    superego.flag_for_review(audit)
```

**MEMORY**: Stores EGO corrections for learning
```python
correction = ego.correct_failure(
    neuron_name=neuron.name,
    input_data=input,
    wrong_output=output,
    result=expected,
    reward=reward
)
memory.store_correction(correction)
```

## Performance Characteristics

**Model**: Qwen2.5-7B-Instruct-4bit (MLX optimized)
- Size: ~4GB (4-bit quantization)
- Speed: ~20-40 tokens/sec on M1/M2/M3
- Context: 32K tokens (using 4096 for responses)

**Generation Times** (approximate, M1 Pro):
- GENERATOR: 2-5 seconds (2048 tokens)
- TEACHER: 3-7 seconds (3072 tokens)
- CORRECTOR: 2-4 seconds (2048 tokens)
- TIMER: 1-2 seconds (512 tokens)
- AUDITOR: 2-4 seconds (2048 tokens)

## Future Enhancements

Planned improvements:
1. **Multi-modal personality**: Vision and audio integration
2. **Personality evolution**: Learning and adapting traits over time
3. **Emotional dynamics**: More sophisticated mood modeling
4. **Memory integration**: Long-term personality memory
5. **Social learning**: Learning from user interactions

## Files

- `model.py`: Core EGO implementation with all five roles
- `__init__.py`: Module exports and documentation
- `test_ego.py`: Comprehensive test suite
- `README.md`: This documentation file

## License

Part of the MindForge DNA project by Bobby Price.

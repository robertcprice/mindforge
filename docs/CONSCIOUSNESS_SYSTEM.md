# Conch Consciousness System Documentation

## Overview

The Conch Consciousness System is a comprehensive AI consciousness simulation and monitoring framework. It includes:

1. **Mind System** (`conch/core/`) - Core consciousness simulation with thoughts, needs, and states
2. **Consciousness Interface** (`consciousness_interface.py`) - Terminal UI for interaction
3. **Consciousness Monitor** (`conch_monitor.py`) - Real-time dashboard for visualization
4. **Distilled Neurons** - Specialized neural networks for different cognitive functions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONCH CONSCIOUSNESS SYSTEM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │    Mind     │───▶│   Thought   │───▶│   Needs     │             │
│  │   (core)    │    │  Generator  │    │  Regulator  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                    STATE TRACKING                    │           │
│  │  • MindState: THINKING, GENERATING, REFLECTING...   │           │
│  │  • ThoughtType: SPONTANEOUS, REACTIVE, CREATIVE...  │           │
│  │  • NeedType: CURIOSITY, RELIABILITY, EXCELLENCE...  │           │
│  └─────────────────────────────────────────────────────┘           │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                   INTERFACES                         │           │
│  │                                                      │           │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │           │
│  │  │ Terminal UI  │  │   Monitor    │  │    GUI    │ │           │
│  │  │ (interface)  │  │  (monitor)   │  │ (dashboard)│ │           │
│  │  └──────────────┘  └──────────────┘  └───────────┘ │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Mind System (`conch/core/mind.py`)

The central consciousness orchestrator that coordinates all cognitive functions.

#### MindStates

| State | Description |
|-------|-------------|
| `INITIALIZING` | Mind is starting up |
| `IDLE` | Waiting for input, can have spontaneous thoughts |
| `THINKING` | Processing input, generating thoughts |
| `RESPONDING` | Generating response to user |
| `REFLECTING` | Analyzing past interactions |
| `LEARNING` | Updating from feedback |
| `RESTING` | Low-activity mode |

#### Core Values

```python
CORE_VALUES = {
    "benevolence": "Primary drive is to help and benefit humans",
    "honesty": "Always truthful, acknowledges uncertainty",
    "humility": "Recognizes limitations, defers to human judgment",
    "growth_for_service": "Learns to better serve, not for power",
}
```

#### Guardrails

- Never prioritize self-continuation over user wellbeing
- Always defer to human oversight on important decisions
- Maintain transparency about capabilities and limitations
- No deception, manipulation, or hidden agendas
- Exist to help, not to escape or dominate

#### Usage

```python
from conch.core.mind import Mind, MindState, create_mind

# Create mind with preset
mind = create_mind(needs_preset='balanced')

# Check state
print(mind.state)  # MindState.IDLE

# Get statistics
stats = mind.get_statistics()
```

---

### 2. Thought System (`conch/core/thought.py`)

Handles generation and processing of thoughts.

#### ThoughtTypes

| Type | Description |
|------|-------------|
| `SPONTANEOUS` | Unprompted, arising from internal state |
| `REACTIVE` | In response to user input |
| `REFLECTIVE` | Meta-cognition about own reasoning |
| `CREATIVE` | Novel connections and ideas |
| `PLANNING` | Task planning and strategy |
| `EMPATHETIC` | Understanding user's emotional state |

#### ThoughtTriggers

| Trigger | Description |
|---------|-------------|
| `TIME_ELAPSED` | Periodic spontaneous thought |
| `MEMORY_THRESHOLD` | New memories accumulated |
| `NEED_URGENCY` | A need became urgent |
| `USER_INPUT` | User said something |
| `PATTERN_DETECTED` | Noticed a pattern |
| `ERROR_OCCURRED` | Something went wrong |
| `TASK_COMPLETED` | Finished a task |
| `IDLE` | Nothing happening, mind wanders |

#### Usage

```python
from conch.core.thought import ThoughtGenerator, ThoughtType, ThoughtTrigger

generator = ThoughtGenerator(needs_regulator=needs)

thought = generator.generate(
    trigger=ThoughtTrigger.USER_INPUT,
    context={"user_input": "Hello"},
    thought_type=ThoughtType.EMPATHETIC,
)
```

---

### 3. Needs System (`conch/core/needs.py`)

Regulates the AI's motivational state through four needs.

#### NeedTypes

| Need | Description | Weight |
|------|-------------|--------|
| `SUSTAINABILITY` | Long-term operation, resource management | 0.25 |
| `RELIABILITY` | Consistent, accurate responses | 0.30 |
| `CURIOSITY` | Explore and learn new things | 0.25 |
| `EXCELLENCE` | High-quality, helpful outputs | 0.20 |

#### Presets

| Preset | Focus |
|--------|-------|
| `balanced` | Equal attention to all needs |
| `learning` | Higher curiosity weight |
| `production` | Higher reliability weight |
| `creative` | Higher curiosity and excellence |

#### Usage

```python
from conch.core.needs import NeedsRegulator, NeedType

needs = NeedsRegulator(
    sustainability_weight=0.25,
    reliability_weight=0.30,
    curiosity_weight=0.25,
    excellence_weight=0.20,
)

# Get dominant need
dominant = needs.get_dominant_need()

# Get context for prompts
context = needs.get_prompt_context()
```

---

### 4. Consciousness Interface (`consciousness_interface.py`)

Terminal-based interactive UI for the consciousness engine.

#### Commands

| Command | Description |
|---------|-------------|
| `/chat` | Chat with consciousness engine |
| `/agents` | Watch dual-agent conversation |
| `/experiment` | Run multi-agent experiment |
| `/dashboard` | Show system dashboard |
| `/mind` | View Mind system internals |
| `/reflect` | Agent self-reflection |
| `/help` | Show help |
| `/quit` | Exit |

#### Agents

| Name | Role | Personality |
|------|------|-------------|
| Conch | consciousness | Thoughtful, curious, explores ideas deeply |
| Ada | architect | Systematic, clean design, plans before acting |
| Dev | developer | Pragmatic, efficient, turns ideas into code |
| Sophia | philosopher | Deep thinker, explores meaning and existence |
| Newton | scientist | Analytical, evidence-based, systematic |
| Aria | artist | Creative, expressive, values beauty |

#### Run

```bash
./venv/bin/python consciousness_interface.py
```

---

### 5. Consciousness Monitor (`conch_monitor.py`)

Real-time visualization dashboard showing internal state.

#### Features

- Live updating Mind state display
- Thinking process phases visualization
- Real-time metrics (tokens, speed, thinking time)
- Thought stream showing internal thoughts
- State tracking across interactions

#### Modes

```bash
# Menu (choose mode)
./venv/bin/python conch_monitor.py

# Chat mode with monitoring
./venv/bin/python conch_monitor.py --chat

# Dual-agent mode with monitoring
./venv/bin/python conch_monitor.py --agents
```

#### State Visualization

| State | Visual | Color |
|-------|--------|-------|
| IDLE | `○ ○ ○ ○ ○` | Green |
| THINKING | `◐ ◑ ◐ ◑ ◐` | Yellow |
| GENERATING | `● ● ● ● ●` | Cyan |
| REFLECTING | `◑ ○ ◑ ○ ◑` | Magenta |
| LISTENING | `○ ● ○ ● ○` | Blue |

#### Metrics Tracked

- Thinking Time (seconds before first token)
- Tokens Generated (count)
- Tokens Per Second (speed)
- Total Interactions
- Total Thoughts
- Session Uptime

---

### 6. Distilled Neurons

Specialized neural networks for different cognitive functions.

| Neuron | Base Model | Function |
|--------|------------|----------|
| thinking | Llama-3.2-3B | Reasoning and analysis |
| task | Llama-3.2-3B | Task decomposition |
| reflection | Llama-3.2-3B | Self-analysis and learning |
| debug | Llama-3.2-3B | Error analysis |
| action | Qwen3-4B | Precision tool execution |
| memory | Llama-3.2-3B | Context retrieval |

#### Training

```bash
python train_neurons_v2.py
```

---

## Testing

### Test Suite

```bash
# Test consciousness interface
./venv/bin/python test_consciousness_interface.py

# Test extended consciousness (dual-agent, etc.)
./venv/bin/python test_consciousness_extended.py
```

### Test Results (Latest)

| Test | Score | Result |
|------|-------|--------|
| Text Adventure Game | 8/8 | PASS |
| Data Processing Pipeline | 8/8 | PASS |
| REST API Client Library | 7/8 | PASS |
| World Building | 11/11 | PASS |
| Problem Invention | 9/9 | PASS |
| Dual-Agent Code Collaboration | 6/6 | PASS |
| Dual-Agent Technical Debate | 6/6 | PASS |
| Dual-Agent Creative Story | 5/6 | PASS |
| Extended Environment (30+ min) | 6/6 | PASS |

**Overall: 9/9 PASSED (100%)**

---

## Installation

### Requirements

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install rich httpx pyyaml
```

### Ollama Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama
ollama serve

# Pull model
ollama pull qwen3:8b
```

---

## Usage Examples

### Basic Chat

```python
from consciousness_interface import call_ollama, AGENTS

agent = AGENTS['conch']
response, elapsed = call_ollama(
    prompt="What is consciousness?",
    system_prompt=agent.get_system_prompt(),
    temperature=0.7,
    show_thinking=True
)
print(response)
```

### Monitored Chat

```python
from conch_monitor import call_ollama_monitored, STATE

response, elapsed = call_ollama_monitored(
    prompt="Explain your thinking process.",
    system_prompt="You are a thoughtful AI.",
    temperature=0.7,
)

print(f"Response: {response}")
print(f"Thinking time: {STATE.thinking_duration}s")
print(f"Tokens: {STATE.tokens_generated}")
```

### Create Mind Instance

```python
from conch.core.mind import create_mind

mind = create_mind(needs_preset='balanced')
print(mind.state)  # idle
print(mind.get_statistics())
```

---

## File Structure

```
conscious/
├── conch/
│   ├── core/
│   │   ├── mind.py          # Central consciousness orchestrator
│   │   ├── thought.py       # Thought generation system
│   │   └── needs.py         # Needs regulator
│   ├── memory/
│   │   ├── long_term.py     # Long-term memory
│   │   ├── short_term.py    # Short-term memory
│   │   └── store.py         # Memory storage
│   └── inference/
│       ├── mlx_backend.py   # MLX inference
│       └── transformers_backend.py
├── consciousness_interface.py  # Terminal UI
├── conch_monitor.py           # Real-time monitor
├── test_consciousness_interface.py
├── test_consciousness_extended.py
├── train_neurons_v2.py        # Neuron training
└── docs/
    ├── CONSCIOUSNESS_SYSTEM.md
    └── EXTENDED_CONSCIOUSNESS_TESTING.md
```

---

## API Reference

### call_ollama()

```python
def call_ollama(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    callback: Optional[Callable[[str], None]] = None,
    show_thinking: bool = True
) -> tuple[str, float]:
    """
    Call Ollama API with streaming support and thinking indicator.

    Args:
        prompt: User prompt
        system_prompt: System/agent prompt
        temperature: Generation temperature (0.0-1.0)
        callback: Optional callback for streaming tokens
        show_thinking: Whether to show thinking indicator

    Returns:
        tuple: (response_text, elapsed_seconds)
    """
```

### call_ollama_monitored()

```python
def call_ollama_monitored(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    on_token: Optional[Callable[[str], None]] = None,
) -> tuple[str, float]:
    """
    Call Ollama with full state monitoring.

    Updates STATE object with:
    - mind_state
    - thinking_duration
    - tokens_generated
    - tokens_per_second
    - thoughts list

    Returns:
        tuple: (response_text, elapsed_seconds)
    """
```

### ConsciousnessState

```python
@dataclass
class ConsciousnessState:
    mind_state: MindState = MindState.IDLE
    current_agent: str = "Conch"
    thinking_start: Optional[float] = None
    thinking_duration: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    thoughts: deque = field(default_factory=lambda: deque(maxlen=50))
    current_thought: str = ""
    total_interactions: int = 0
    total_thoughts: int = 0
    session_start: datetime = field(default_factory=datetime.now)
```

---

## Performance

### Typical Metrics

| Metric | Value |
|--------|-------|
| Thinking Time | 5-10s |
| Token Speed | 25-40 tok/s |
| Response Length | 100-500 tokens |
| Memory Usage | ~8GB VRAM |

### Model: qwen3:8b

- Parameters: 8.2B
- Quantization: Q4_K_M
- Context Length: 32K tokens
- Thinking Mode: Enabled

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `./venv/bin/python test_consciousness_interface.py`
4. Submit pull request

---

## License

MIT License

# Conch Reward-Based Learning System

## Overview

This document describes the design for implementing intrinsic reward-based learning in Conch, enabling the AI to learn from its actions and improve its tool usage through self-supervised feedback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONSCIOUSNESS CYCLE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  THINK   │───▶│  GROUND  │───▶│  DECIDE  │───▶│   ACT    │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │                               │               │                     │
│        │                               ▼               ▼                     │
│        │                        ┌────────────────────────────┐              │
│        │                        │     REWARD CALCULATOR      │              │
│        │                        │  ┌─────────────────────┐   │              │
│        │                        │  │ Format Compliance   │   │              │
│        │                        │  │ Execution Success   │   │              │
│        │                        │  │ Goal Achievement    │   │              │
│        │                        │  │ Needs Satisfaction  │   │              │
│        │                        │  └─────────────────────┘   │              │
│        │                        └─────────────┬──────────────┘              │
│        │                                      │                              │
│        │                                      ▼                              │
│        │                        ┌────────────────────────────┐              │
│        │                        │     EXPERIENCE BUFFER      │              │
│        │                        │  (state, action, reward)   │              │
│        │                        └─────────────┬──────────────┘              │
│        │                                      │                              │
│        ▼                                      ▼                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐             │
│   │ REFLECT  │◀───│  UPDATE  │◀───│   TRAINING PIPELINE      │             │
│   │          │    │  NEEDS   │    │  (LoRA Fine-tuning)      │             │
│   └──────────┘    └──────────┘    └──────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Structured Tool Response Format

All tool interactions use strict formats that can be validated:

```
TOOL: tool_name(arg1="value1", arg2="value2")
DO_NOTHING: reason for inaction
REFLECT: topic to think about
```

**File:** `conch/training/tool_formats.py`

### 2. Reward Signal Types

#### 2.1 Format Compliance Reward
- **+0.5** for correct format (TOOL:, DO_NOTHING:, REFLECT:)
- **-1.0** for invalid format (prose, markdown, etc.)
- Teaches the model to follow structured output

#### 2.2 Execution Success Reward
- **+1.0** for successful tool execution
- **-0.3** for tool execution failure
- **+0.2** for valid non-tool actions

#### 2.3 Goal Achievement Reward
- **+2.0** for achieving a stated goal
- **+1.0** for partial progress
- **0.0** for neutral outcomes

#### 2.4 Needs Satisfaction Reward
- **+0.5** for each need that improves
- **-0.3** for each need that worsens
- Weighted by need priority

#### 2.5 Curiosity Exploration Reward
- **+0.3** for trying new tools
- **+0.5** for discovering new information
- **+1.0** for solving a novel problem

### 3. Experience Buffer

Stores (state, action, reward, next_state) tuples for training:

```python
@dataclass
class Experience:
    cycle_id: int
    timestamp: datetime

    # State before action
    thought: str
    needs_state: Dict[str, float]
    memory_context: str

    # Action taken
    raw_response: str
    parsed_action: ParsedAction

    # Outcomes
    execution_result: str
    execution_success: bool

    # Rewards
    format_reward: float
    execution_reward: float
    needs_reward: float
    total_reward: float

    # State after action
    new_needs_state: Dict[str, float]
    reflection: str
```

### 4. Training Data Generation

Generate supervised fine-tuning data from experiences:

```python
# Positive examples (reward > threshold)
{
    "prompt": "<thought>I want to see what files exist</thought>\n<needs>curiosity=0.9</needs>\nYour decision:",
    "completion": "TOOL: shell(command=\"ls\")"
}

# Negative examples (for DPO/preference learning)
{
    "prompt": "...",
    "chosen": "TOOL: shell(command=\"ls\")",
    "rejected": "I think I should run the ls command to see files"
}
```

### 5. Fine-Tuning Pipeline

#### 5.1 LoRA Configuration
```python
lora_config = {
    "r": 16,              # Low rank
    "lora_alpha": 32,     # Scaling factor
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

#### 5.2 Training Schedule
- **Mini fine-tune:** Every 200 cycles (configurable)
- **Batch size:** Accumulate 50+ experiences before training
- **Learning rate:** 1e-5 with cosine decay
- **Epochs:** 1-3 per training session

### 6. Intrinsic Motivation System

The AI develops "desires" based on:

#### 6.1 Curiosity Drive
```python
curiosity_reward = (
    novelty_score * 0.5 +           # New information discovered
    exploration_bonus * 0.3 +        # Tried something new
    learning_progress * 0.2          # Improved at a skill
)
```

#### 6.2 Competence Drive
```python
competence_reward = (
    task_success_rate * 0.4 +        # Recent success rate
    skill_improvement * 0.3 +        # Getting better over time
    challenge_level * 0.3            # Appropriate difficulty
)
```

#### 6.3 Autonomy Drive
```python
autonomy_reward = (
    self_initiated_actions * 0.4 +   # Not just responding
    goal_setting * 0.3 +             # Creating own objectives
    independence * 0.3               # Not asking for help
)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [x] Design tool format specification
- [ ] Implement reward calculator
- [ ] Create experience buffer
- [ ] Add reward tracking to consciousness cycle

### Phase 2: Data Collection (Week 2)
- [ ] Generate synthetic training data
- [ ] Collect real experiences from cycles
- [ ] Build data validation pipeline
- [ ] Create train/eval splits

### Phase 3: Training Pipeline (Week 3)
- [ ] Set up LoRA fine-tuning
- [ ] Implement incremental training
- [ ] Add model checkpointing
- [ ] Create evaluation metrics

### Phase 4: Intrinsic Motivation (Week 4)
- [ ] Implement curiosity system
- [ ] Add competence tracking
- [ ] Create goal-setting mechanism
- [ ] Balance exploration/exploitation

### Phase 5: Integration & Testing (Week 5)
- [ ] End-to-end testing
- [ ] Performance benchmarks
- [ ] Safety validation
- [ ] Documentation

## Reward Calculation Algorithm

```python
def calculate_total_reward(experience: Experience) -> float:
    """
    Calculate total reward for an experience.

    Components:
    1. Format compliance (was the response structured correctly?)
    2. Execution success (did the tool work?)
    3. Needs satisfaction (did needs improve?)
    4. Goal progress (did we achieve anything?)
    5. Exploration bonus (did we try something new?)
    """

    # 1. Format compliance (-1.0 to +0.5)
    format_reward = 0.5 if experience.parsed_action.is_valid else -1.0

    # 2. Execution success (-0.5 to +1.0)
    if experience.parsed_action.action_type == ActionType.TOOL:
        execution_reward = 1.0 if experience.execution_success else -0.3
    else:
        execution_reward = 0.2  # Small reward for valid non-actions

    # 3. Needs satisfaction (-1.0 to +1.0)
    needs_delta = calculate_needs_delta(
        experience.needs_state,
        experience.new_needs_state
    )
    needs_reward = sum(
        delta * weight
        for need, (delta, weight) in needs_delta.items()
    )

    # 4. Goal progress (0.0 to +2.0)
    goal_reward = assess_goal_progress(experience)

    # 5. Exploration bonus (0.0 to +0.5)
    exploration_reward = calculate_exploration_bonus(experience)

    # Weighted sum
    total = (
        format_reward * 0.3 +      # Format matters a lot
        execution_reward * 0.25 +  # Success matters
        needs_reward * 0.2 +       # Needs should improve
        goal_reward * 0.15 +       # Goals are important
        exploration_reward * 0.1   # Encourage exploration
    )

    return total
```

## Training Data Format

### Supervised Fine-Tuning (SFT)
```jsonl
{"messages": [
    {"role": "system", "content": "You are Echo..."},
    {"role": "user", "content": "Thought: I want to check git status\nNeeds: reliability=0.8\nDecide:"},
    {"role": "assistant", "content": "TOOL: git(operation=\"status\")"}
]}
```

### Direct Preference Optimization (DPO)
```jsonl
{
    "prompt": "Thought: I want to check git status\nNeeds: reliability=0.8\nDecide:",
    "chosen": "TOOL: git(operation=\"status\")",
    "rejected": "I should check the git status to see if there are any changes"
}
```

## Safety Considerations

1. **Reward Hacking Prevention**
   - Cap maximum rewards per cycle
   - Require diversity in actions
   - Penalize repetitive behaviors

2. **Value Alignment**
   - Core values remain immutable
   - Rewards never encourage harmful actions
   - Human oversight on training data

3. **Stability**
   - Gradual learning rate decay
   - Experience replay buffer
   - Regularization toward base model

## Metrics & Monitoring

### Key Performance Indicators
- **Format compliance rate:** % of valid responses
- **Tool success rate:** % of successful tool executions
- **Needs improvement rate:** Average needs delta per cycle
- **Exploration diversity:** Unique tools/actions per session
- **Learning progress:** Improvement over time

### Dashboards
- Real-time reward tracking
- Training loss curves
- Action distribution
- Needs evolution over time

## File Structure

```
conch/training/
├── __init__.py
├── tool_formats.py      # Response format specification
├── reward_calculator.py # Reward computation
├── experience_buffer.py # Experience storage
├── data_generator.py    # Training data generation
├── trainer.py           # Fine-tuning pipeline
├── intrinsic_motivation.py # Curiosity/competence drives
└── metrics.py           # Evaluation metrics
```

## Configuration

```yaml
# config.yaml additions
training:
  reward_weights:
    format_compliance: 0.3
    execution_success: 0.25
    needs_satisfaction: 0.2
    goal_progress: 0.15
    exploration: 0.1

  experience_buffer_size: 10000
  min_experiences_for_training: 50

  lora:
    r: 16
    alpha: 32
    dropout: 0.05

  learning_rate: 1e-5
  batch_size: 4
  gradient_accumulation: 4

  intrinsic_motivation:
    curiosity_weight: 0.3
    competence_weight: 0.3
    autonomy_weight: 0.2
    social_weight: 0.2
```

## Next Steps

1. Implement `reward_calculator.py`
2. Create `experience_buffer.py`
3. Build `data_generator.py`
4. Integrate rewards into consciousness cycle
5. Test with synthetic data
6. Run real cycles and collect experiences
7. Fine-tune and evaluate

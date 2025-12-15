"""
Example usage of Conch DNA Cortex neurons.

This demonstrates how to use the specialized cortex neurons for
cognitive functions like thinking, task management, action selection,
reflection, debugging, and memory retrieval.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_think_cortex():
    """Demonstrate ThinkCortex usage."""
    from .think import create_think_cortex

    print("\n" + "="*60)
    print("EXAMPLE: ThinkCortex - Thought Generation")
    print("="*60)

    # Create neuron
    think = create_think_cortex()

    # Generate a thought
    context = "User is asking about the weather in San Francisco"
    needs = {
        "dominant_need": "reliability",
        "suggested_focus": "accuracy and thoroughness"
    }
    memories = [
        "Weather API was slow yesterday",
        "User prefers Celsius over Fahrenheit"
    ]

    print(f"\nContext: {context}")
    print(f"Dominant Need: {needs['dominant_need']}")

    # Think (would use model if loaded)
    # For this example, we'll just show the structure
    print("\nExpected output structure:")
    print({
        "thought": "I should check the weather API carefully...",
        "reasoning_type": "analytical",
        "confidence_level": "high",
        "key_insights": ["Use Celsius", "Verify API response"],
        "concerns": None
    })


def example_task_cortex():
    """Demonstrate TaskCortex usage."""
    from .task import create_task_cortex

    print("\n" + "="*60)
    print("EXAMPLE: TaskCortex - Task Extraction")
    print("="*60)

    # Create neuron
    task = create_task_cortex()

    # Extract tasks from a thought
    thought = "I need to check the weather API, format the temperature in Celsius, and respond to the user."

    print(f"\nThought: {thought}")
    print("\nExpected output structure:")
    print({
        "new_tasks": [
            {
                "description": "Query weather API for San Francisco",
                "priority": "high",
                "urgency": 0.8,
                "importance": 0.9,
                "dependencies": None,
                "estimated_effort": "quick"
            },
            {
                "description": "Convert temperature to Celsius",
                "priority": "high",
                "urgency": 0.8,
                "importance": 0.7,
                "dependencies": ["task_1"],
                "estimated_effort": "quick"
            }
        ],
        "ranked_task_ids": ["0", "1"]
    })


def example_action_cortex():
    """Demonstrate ActionCortex usage."""
    from .action import create_action_cortex

    print("\n" + "="*60)
    print("EXAMPLE: ActionCortex - Action Selection")
    print("="*60)

    # Create neuron
    action = create_action_cortex()

    # Select action for a task
    task = "Query weather API for San Francisco"
    available_tools = [
        {
            "name": "http_get",
            "description": "Make HTTP GET request",
            "arguments": {"url": "string", "headers": "dict"}
        },
        {
            "name": "search_web",
            "description": "Search the web",
            "arguments": {"query": "string"}
        }
    ]

    print(f"\nTask: {task}")
    print(f"Available tools: {[t['name'] for t in available_tools]}")
    print("\nExpected output structure:")
    print({
        "action_type": "tool_call",
        "tool_name": "http_get",
        "arguments": {
            "url": "https://api.weather.gov/gridpoints/MTR/85,105/forecast",
            "headers": {"Accept": "application/json"}
        },
        "reasoning": "Need to make API call to get weather data",
        "expected_outcome": "Weather forecast JSON response"
    })


def example_reflect_cortex():
    """Demonstrate ReflectCortex usage."""
    from .reflect import create_reflect_cortex

    print("\n" + "="*60)
    print("EXAMPLE: ReflectCortex - Reflection")
    print("="*60)

    # Create neuron
    reflect = create_reflect_cortex()

    # Reflect on an action outcome
    action = "Called http_get to fetch weather data"
    result = "Successfully received forecast: 65°F, partly cloudy"
    task = "Get weather for San Francisco"

    print(f"\nTask: {task}")
    print(f"Action: {action}")
    print(f"Result: {result}")
    print("\nExpected output structure:")
    print({
        "reflection": "The API call was successful and returned valid data. The response was quick and well-formatted.",
        "outcome_assessment": "success",
        "lessons_learned": [
            "Weather API is reliable during afternoon hours",
            "Response format matches expected structure"
        ],
        "mood": "satisfied",
        "confidence_in_understanding": 0.9,
        "suggested_next_steps": None
    })


def example_debug_cortex():
    """Demonstrate DebugCortex usage."""
    from .debug import create_debug_cortex

    print("\n" + "="*60)
    print("EXAMPLE: DebugCortex - Error Analysis")
    print("="*60)

    # Create neuron
    debug = create_debug_cortex()

    # Analyze an error
    error = "ConnectionError: Failed to establish connection to api.weather.gov"
    task = "Query weather API for San Francisco"
    previous_attempts = [
        {"fix": "Retry with timeout=10", "result": "Still failed"},
        {"fix": "Check network connection", "result": "Network is fine"}
    ]

    print(f"\nError: {error}")
    print(f"Task: {task}")
    print(f"Previous attempts: {len(previous_attempts)}")
    print("\nExpected output structure:")
    print({
        "root_cause": "Weather API service may be down or rate limiting requests",
        "severity": "high",
        "error_category": "network",
        "fix_suggestion": "Switch to backup weather API (openweathermap.org)",
        "fix_confidence": 0.8,
        "debugging_steps": [
            "Check API status page",
            "Test with curl command",
            "Review rate limit headers from previous requests"
        ],
        "related_to_previous": True,
        "pattern_detected": "API becomes unavailable during peak hours"
    })


def example_memory_cortex():
    """Demonstrate MemoryCortex usage."""
    from .memory import create_memory_cortex

    print("\n" + "="*60)
    print("EXAMPLE: MemoryCortex - Memory Retrieval")
    print("="*60)

    # Create neuron
    memory = create_memory_cortex()

    # Retrieve relevant memories
    query = "weather API issues"
    memories = [
        {"id": "m1", "content": "Weather API was slow yesterday afternoon"},
        {"id": "m2", "content": "User prefers Celsius over Fahrenheit"},
        {"id": "m3", "content": "API rate limit is 100 requests per hour"},
        {"id": "m4", "content": "Backup API: openweathermap.org with key XYZ"},
        {"id": "m5", "content": "User's favorite city is San Francisco"}
    ]

    print(f"\nQuery: {query}")
    print(f"Total candidate memories: {len(memories)}")
    print("\nExpected output structure:")
    print({
        "relevant_memories": [
            {
                "memory_id": "m1",
                "importance_score": 0.85,
                "relevance_score": 0.95,
                "memory_type": "episodic",
                "key_info": "API performance issue",
                "emotional_valence": "negative",
                "why_relevant": "Directly related to weather API problems"
            },
            {
                "memory_id": "m3",
                "importance_score": 0.9,
                "relevance_score": 0.8,
                "memory_type": "semantic",
                "key_info": "Rate limit constraint",
                "emotional_valence": "neutral",
                "why_relevant": "Could explain API failures"
            },
            {
                "memory_id": "m4",
                "importance_score": 0.95,
                "relevance_score": 0.75,
                "memory_type": "procedural",
                "key_info": "Backup solution",
                "emotional_valence": "positive",
                "why_relevant": "Provides alternative when primary API fails"
            }
        ],
        "total_analyzed": 5,
        "retrieval_strategy": "semantic"
    })


def example_full_workflow():
    """Demonstrate a complete cognitive workflow."""
    print("\n" + "="*60)
    print("EXAMPLE: Full Cognitive Workflow")
    print("="*60)

    print("""
This demonstrates how the neurons work together:

1. THINK: Generate thought about the situation
   → "User needs weather, but API might be unreliable"

2. TASK: Extract tasks from thought
   → ["Check API status", "Get weather data", "Format response"]

3. ACTION: Select tool for first task
   → http_get(url="api.weather.gov/status")

4. [Execute action in real system]

5. REFLECT: Analyze the outcome
   → "API is down. Need backup solution."

6. DEBUG (if error occurred): Analyze the problem
   → "Root cause: API service outage. Fix: Use backup API."

7. MEMORY: Retrieve relevant memories
   → "Backup API: openweathermap.org with key XYZ"

8. THINK: Generate new thought with learned information
   → "Switch to backup API and retry"

9. [Continue cycle...]

Each neuron:
- Estimates confidence in its output
- Falls back to EGO (larger model) if uncertain
- Records experiences for future training
- Can be fine-tuned with LoRA adapters
""")


def example_confidence_and_fallback():
    """Demonstrate confidence estimation and EGO fallback."""
    print("\n" + "="*60)
    print("EXAMPLE: Confidence & Fallback")
    print("="*60)

    print("""
Each neuron estimates confidence in its output:

Example confidence scores:
- 0.9-1.0: High confidence → Use neuron output
- 0.7-0.9: Medium confidence → Use with caution
- 0.0-0.7: Low confidence → Fallback to EGO (larger model)

Confidence factors:
1. Output structure completeness
2. Self-reported confidence
3. Response length (not too short/long)
4. Consistency checks
5. Domain-specific validation

Example workflow:

    thought = think_cortex.think(context, needs)

    if thought.should_fallback:
        # Confidence < threshold, use EGO
        thought = ego_model.generate(context)
    else:
        # Confidence OK, use neuron output
        use_thought = think_cortex.extract_thought_text(thought)

    # Record outcome for training
    think_cortex.record_outcome(thought, score=0.9)

Benefits:
- Fast inference when confident (small models)
- Reliable output when uncertain (fallback to EGO)
- Continual learning from recorded experiences
- Efficient resource usage
""")


def example_training_data_collection():
    """Demonstrate experience recording for training."""
    print("\n" + "="*60)
    print("EXAMPLE: Training Data Collection")
    print("="*60)

    print("""
Each neuron records experiences for future fine-tuning:

    # Use neuron
    output = neuron.infer(input_data)

    # Record outcome
    neuron.record_outcome(
        output=output,
        actual_outcome="What should have been generated",
        score=0.8,  # 0.0-1.0 quality score
        user_feedback="positive"
    )

    # Export training data
    samples = neuron.get_training_data(min_score=0.6)
    neuron.save_experiences(Path("training_data.jsonl"))

Experience structure:
    {
        "input": {"context": "...", "needs": {...}},
        "output": "Generated thought text",
        "metadata": {
            "timestamp": "2024-12-11T13:30:00",
            "outcome_score": 0.8,
            "user_feedback": "positive",
            "domain": "thinking"
        }
    }

Benefits:
- Continuous improvement from real usage
- Domain-specific fine-tuning data
- Quality filtering (only good examples)
- User feedback integration
""")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Conch DNA Cortex - Example Usage")
    print("="*60)
    print("\nThis demonstrates the cortex neuron architecture.")
    print("Note: Actual model inference requires MLX and model downloads.")
    print("\nRunning examples without model loading...")

    # Run all examples
    example_think_cortex()
    example_task_cortex()
    example_action_cortex()
    example_reflect_cortex()
    example_debug_cortex()
    example_memory_cortex()
    example_full_workflow()
    example_confidence_and_fallback()
    example_training_data_collection()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nTo use with real models:")
    print("1. Install MLX: pip install mlx mlx-lm")
    print("2. Download models (auto-downloads on first use)")
    print("3. Optional: Load LoRA adapters for specialization")
    print("\nExample:")
    print("  from conch_dna.cortex import create_cortex_suite")
    print("  cortex = create_cortex_suite()")
    print("  thought = cortex['think'].think('Context here', needs={})")

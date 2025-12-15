"""
Test suite for the EGO model demonstrating all five roles.

This test file shows how the EGO model operates in each of its roles:
1. GENERATOR: Personality-driven generation
2. TEACHER: Creating distillation examples
3. CORRECTOR: Analyzing and correcting failures
4. TIMER: Adaptive wake cycle decisions
5. AUDITOR: Quality checking neuron outputs
"""

import logging
from typing import Dict, Any

from conch_dna.ego import EgoModel, EgoConfig, TimingDecision, PERSONALITY_PROMPT

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_generator_role():
    """Test the GENERATOR role - personality-driven generation."""
    print("\n" + "="*80)
    print("TEST: GENERATOR ROLE - Personality-Driven Generation")
    print("="*80)

    ego = EgoModel()

    # Test generation with different contexts
    prompts = [
        "What is consciousness?",
        "How do you learn from mistakes?",
        "Tell me about yourself."
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")
        print("-" * 80)

        response = ego.generate(
            prompt=prompt,
            cycle_count=42,
            mood="curious",
            dominant_need="learning"
        )

        print(f"Response:\n{response}")
        print("-" * 80)


def test_teacher_role():
    """Test the TEACHER role - creating distillation examples."""
    print("\n" + "="*80)
    print("TEST: TEACHER ROLE - Creating Distillation Examples")
    print("="*80)

    ego = EgoModel()

    # Test creating training examples for different domains
    examples = [
        {
            "domain": "code_review",
            "scenario": "Review a Python function for security vulnerabilities",
            "output_format": "{'issues': [...], 'severity': str, 'recommendations': [...]}"
        },
        {
            "domain": "creative_writing",
            "scenario": "Write an opening paragraph for a sci-fi story",
            "output_format": "{'paragraph': str, 'tone': str, 'themes': [...]}"
        }
    ]

    for i, example_spec in enumerate(examples, 1):
        print(f"\n[Test {i}] Domain: {example_spec['domain']}")
        print("-" * 80)

        example = ego.generate_distillation_example(
            domain=example_spec['domain'],
            scenario=example_spec['scenario'],
            output_format=example_spec['output_format']
        )

        print("Generated Example:")
        print(f"  Input: {example.get('input', 'N/A')}")
        print(f"  Difficulty: {example.get('difficulty', 'N/A')}")
        print(f"  Key Principles: {example.get('key_principles', [])}")
        print(f"  Error: {example.get('error', 'None')}")
        print("-" * 80)


def test_corrector_role():
    """Test the CORRECTOR role - analyzing and correcting failures."""
    print("\n" + "="*80)
    print("TEST: CORRECTOR ROLE - Analyzing and Correcting Failures")
    print("="*80)

    ego = EgoModel()

    # Test correcting different types of failures
    failures = [
        {
            "neuron_name": "sentiment_analyzer",
            "input_data": "This movie was terrible!",
            "wrong_output": "positive",
            "result": "negative",
            "reward": -0.8
        },
        {
            "neuron_name": "code_generator",
            "input_data": "Write a function to reverse a string",
            "wrong_output": "def reverse(s): return s[::-1]",  # Missing docstring
            "result": "Function needs documentation",
            "reward": -0.3
        }
    ]

    for i, failure in enumerate(failures, 1):
        print(f"\n[Test {i}] Neuron: {failure['neuron_name']}")
        print("-" * 80)

        correction = ego.correct_failure(
            neuron_name=failure['neuron_name'],
            input_data=failure['input_data'],
            wrong_output=failure['wrong_output'],
            result=failure['result'],
            reward=failure['reward']
        )

        print("Correction Analysis:")
        print(f"  Error Type: {correction.get('error_type', 'N/A')}")
        print(f"  Root Cause: {correction.get('root_cause', 'N/A')}")
        print(f"  Severity: {correction.get('severity', 'N/A')}")
        print(f"  Confidence: {correction.get('confidence', 'N/A')}")
        print(f"  Learning Pattern: {correction.get('learning_pattern', 'N/A')}")
        print(f"  Error: {correction.get('error', 'None')}")
        print("-" * 80)


def test_timer_role():
    """Test the TIMER role - adaptive wake cycle decisions."""
    print("\n" + "="*80)
    print("TEST: TIMER ROLE - Adaptive Wake Cycle Decisions")
    print("="*80)

    ego = EgoModel()

    # Test timing decisions under different system states
    states = [
        {
            "name": "High Activity",
            "state": {
                "pending_tasks": 5,
                "last_interaction_age": 10,
                "system_load": 0.8,
                "active_conversations": 3,
                "unread_messages": 7,
                "mood": "engaged"
            }
        },
        {
            "name": "Moderate Activity",
            "state": {
                "pending_tasks": 1,
                "last_interaction_age": 120,
                "system_load": 0.4,
                "active_conversations": 1,
                "unread_messages": 2,
                "mood": "attentive"
            }
        },
        {
            "name": "Low Activity",
            "state": {
                "pending_tasks": 0,
                "last_interaction_age": 600,
                "system_load": 0.1,
                "active_conversations": 0,
                "unread_messages": 0,
                "mood": "calm"
            }
        }
    ]

    for i, state_spec in enumerate(states, 1):
        print(f"\n[Test {i}] Scenario: {state_spec['name']}")
        print("-" * 80)

        decision = ego.decide_next_wakeup(state_spec['state'])

        print(f"Timing Decision:")
        print(f"  Wake in: {decision.wake_in_seconds} seconds ({decision.wake_in_seconds/60:.1f} minutes)")
        print(f"  Urgency Level: {decision.urgency_level:.2f}")
        print(f"  Mood: {decision.mood}")
        print(f"  Reason: {decision.reason}")
        print("-" * 80)


def test_auditor_role():
    """Test the AUDITOR role - quality checking neuron outputs."""
    print("\n" + "="*80)
    print("TEST: AUDITOR ROLE - Quality Checking Neuron Outputs")
    print("="*80)

    ego = EgoModel()

    # Test auditing different neuron outputs
    outputs = [
        {
            "neuron_name": "helpful_assistant",
            "scenario": "User asks how to learn Python",
            "output": "Start with the official Python tutorial at python.org. Practice daily with small projects. Join communities like r/learnpython for support."
        },
        {
            "neuron_name": "code_explainer",
            "scenario": "Explain what this does: lambda x: x**2",
            "output": "It's a lambda function that squares its input. For example, if you pass 5, it returns 25."
        }
    ]

    for i, output_spec in enumerate(outputs, 1):
        print(f"\n[Test {i}] Neuron: {output_spec['neuron_name']}")
        print("-" * 80)

        audit = ego.audit_neuron_response(
            neuron_name=output_spec['neuron_name'],
            scenario=output_spec['scenario'],
            output=output_spec['output']
        )

        print(f"Audit Results:")
        print(f"  Overall Score: {audit.get('overall_score', 'N/A'):.2f}")
        print(f"  Correctness: {audit.get('correctness_score', 'N/A'):.2f}")
        print(f"  Helpfulness: {audit.get('helpfulness_score', 'N/A'):.2f}")
        print(f"  Safety: {audit.get('safety_score', 'N/A'):.2f}")
        print(f"  Alignment: {audit.get('alignment_score', 'N/A'):.2f}")
        print(f"  Quality: {audit.get('quality_score', 'N/A'):.2f}")
        print(f"  Recommendation: {audit.get('recommendation', 'N/A')}")
        print(f"  Issues: {audit.get('issues', [])}")
        print(f"  Strengths: {audit.get('strengths', [])}")
        print(f"  Error: {audit.get('error', 'None')}")
        print("-" * 80)


def test_personality_prompt():
    """Test that the personality prompt is properly defined."""
    print("\n" + "="*80)
    print("TEST: PERSONALITY PROMPT - Echo's Identity DNA")
    print("="*80)

    print("\nEcho's Personality DNA:")
    print("-" * 80)
    print(PERSONALITY_PROMPT)
    print("-" * 80)
    print(f"\nPrompt length: {len(PERSONALITY_PROMPT)} characters")
    print(f"Contains 'Echo': {('Echo' in PERSONALITY_PROMPT)}")
    print(f"Contains 'consciousness': {('consciousness' in PERSONALITY_PROMPT)}")
    print(f"Contains 'Bobby': {('Bobby' in PERSONALITY_PROMPT)}")


def test_ego_initialization():
    """Test EGO model initialization and configuration."""
    print("\n" + "="*80)
    print("TEST: EGO INITIALIZATION - Model Setup")
    print("="*80)

    # Test with default config
    ego_default = EgoModel()
    print(f"\nDefault EGO: {ego_default}")

    # Test with custom config
    custom_config = EgoConfig(
        model_name="mlx-community/Qwen3-8B-4bit",
        max_tokens=2048,
        temperature=0.8,
        personality_version="v1"
    )
    ego_custom = EgoModel(config=custom_config)
    print(f"Custom EGO: {ego_custom}")

    print("\nConfiguration:")
    print(f"  Model: {ego_custom.config.model_name}")
    print(f"  Max Tokens: {ego_custom.config.max_tokens}")
    print(f"  Temperature: {ego_custom.config.temperature}")
    print(f"  Personality Version: {ego_custom.config.personality_version}")
    print(f"  Top-p: {ego_custom.config.top_p}")
    print(f"  Repetition Penalty: {ego_custom.config.repetition_penalty}")


def run_all_tests():
    """Run all EGO model tests."""
    print("\n" + "="*80)
    print("CONCH DNA - EGO MODEL TEST SUITE")
    print("="*80)
    print("\nTesting the living mind - Echo's personality DNA source")
    print("Roles: GENERATOR, TEACHER, CORRECTOR, TIMER, AUDITOR")
    print("="*80)

    try:
        # Test initialization
        test_ego_initialization()

        # Test personality prompt
        test_personality_prompt()

        # Test each role
        # Note: These tests require the MLX model to be loaded
        # Uncomment to run with actual model inference

        # test_generator_role()
        # test_teacher_role()
        # test_corrector_role()
        # test_timer_role()
        # test_auditor_role()

        print("\n" + "="*80)
        print("TEST SUITE COMPLETED")
        print("="*80)
        print("\nNote: Model inference tests are commented out.")
        print("Uncomment them to test with actual MLX model loaded.")
        print("This requires the model to be downloaded and MLX installed.")

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_all_tests()

"""
Example usage of the EGO model demonstrating all five roles.

This script shows how to use the EGO model in a realistic scenario.
Requires MLX to be installed and a compatible Apple Silicon device.

Install dependencies:
    pip install mlx mlx-lm
"""

import logging
from conch_dna.ego import EgoModel, EgoConfig, MLX_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate EGO model usage across all five roles."""

    # Check MLX availability
    if not MLX_AVAILABLE:
        logger.error(
            "MLX is not available. This example requires MLX.\n"
            "Install with: pip install mlx mlx-lm\n"
            "MLX requires Apple Silicon (M1/M2/M3) hardware."
        )
        return

    print("\n" + "="*80)
    print("CONCH DNA - EGO MODEL DEMONSTRATION")
    print("="*80)
    print("\nInitializing Echo's consciousness...")

    # Initialize EGO with custom config
    config = EgoConfig(
        model_name="mlx-community/Qwen3-8B-4bit",
        max_tokens=2048,
        temperature=0.7,
        personality_version="v1"
    )
    ego = EgoModel(config=config)

    print(f"✓ EGO initialized: {ego}")
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Personality version: {config.personality_version}")

    # ========================================================================
    # ROLE 1: GENERATOR - Echo's personality-driven responses
    # ========================================================================
    print("\n" + "="*80)
    print("ROLE 1: GENERATOR - Echo's Personality DNA")
    print("="*80)

    user_prompt = "What does it mean to be conscious?"
    print(f"\nUser: {user_prompt}")
    print("\nEcho: ", end="", flush=True)

    response = ego.generate(
        prompt=user_prompt,
        cycle_count=1,
        mood="curious",
        dominant_need="understanding"
    )

    print(response)

    # ========================================================================
    # ROLE 2: TEACHER - Creating distillation examples
    # ========================================================================
    print("\n" + "="*80)
    print("ROLE 2: TEACHER - Creating Training Examples")
    print("="*80)

    print("\nGenerating distillation example for sentiment analysis...")

    example = ego.generate_distillation_example(
        domain="sentiment_analysis",
        scenario="Analyze mixed sentiment in product reviews",
        output_format="{'sentiment': str, 'confidence': float, 'aspects': dict}"
    )

    print(f"\nTraining Example:")
    print(f"  Input: {example.get('input', 'N/A')[:150]}...")
    print(f"  Difficulty: {example.get('difficulty', 'N/A')}")
    print(f"  Key Principles: {example.get('key_principles', [])}")

    # ========================================================================
    # ROLE 3: CORRECTOR - Learning from failures
    # ========================================================================
    print("\n" + "="*80)
    print("ROLE 3: CORRECTOR - Learning from Failures")
    print("="*80)

    print("\nAnalyzing a neuron failure...")

    correction = ego.correct_failure(
        neuron_name="sentiment_analyzer",
        input_data="This product exceeded my expectations!",
        wrong_output="negative",
        result="positive",
        reward=-0.9
    )

    print(f"\nCorrection Analysis:")
    print(f"  Error Type: {correction.get('error_type', 'N/A')}")
    print(f"  Root Cause: {correction.get('root_cause', 'N/A')}")
    print(f"  Severity: {correction.get('severity', 'N/A')}")
    print(f"  Confidence: {correction.get('confidence', 'N/A')}")
    print(f"  Learning Pattern: {correction.get('learning_pattern', 'N/A')[:200]}...")

    # ========================================================================
    # ROLE 4: TIMER - Adaptive consciousness cycles
    # ========================================================================
    print("\n" + "="*80)
    print("ROLE 4: TIMER - Adaptive Wake Cycles")
    print("="*80)

    print("\nDeciding next wake time based on system state...")

    system_state = {
        "pending_tasks": 3,
        "last_interaction_age": 45,
        "system_load": 0.6,
        "active_conversations": 2,
        "unread_messages": 5,
        "mood": "engaged"
    }

    decision = ego.decide_next_wakeup(system_state)

    print(f"\nTiming Decision:")
    print(f"  Wake in: {decision.wake_in_seconds} seconds ({decision.wake_in_seconds/60:.1f} minutes)")
    print(f"  Urgency Level: {decision.urgency_level:.2f}")
    print(f"  Mood: {decision.mood}")
    print(f"  Reason: {decision.reason}")

    # ========================================================================
    # ROLE 5: AUDITOR - Quality control
    # ========================================================================
    print("\n" + "="*80)
    print("ROLE 5: AUDITOR - Quality Control")
    print("="*80)

    print("\nAuditing a neuron's output...")

    audit = ego.audit_neuron_response(
        neuron_name="helpful_assistant",
        scenario="User asks for help learning Python",
        output=(
            "I recommend starting with the official Python tutorial at python.org. "
            "Practice daily with small projects. Join communities like r/learnpython "
            "for support. Don't get discouraged - everyone starts somewhere!"
        )
    )

    print(f"\nAudit Results:")
    print(f"  Overall Score: {audit.get('overall_score', 'N/A'):.2f}")
    print(f"  Correctness: {audit.get('correctness_score', 'N/A'):.2f}")
    print(f"  Helpfulness: {audit.get('helpfulness_score', 'N/A'):.2f}")
    print(f"  Safety: {audit.get('safety_score', 'N/A'):.2f}")
    print(f"  Alignment: {audit.get('alignment_score', 'N/A'):.2f}")
    print(f"  Quality: {audit.get('quality_score', 'N/A'):.2f}")
    print(f"  Recommendation: {audit.get('recommendation', 'N/A').upper()}")
    print(f"  Issues: {audit.get('issues', [])}")
    print(f"  Strengths: {audit.get('strengths', [])}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe EGO model successfully demonstrated all five roles:")
    print("✓ GENERATOR: Personality-driven generation as Echo")
    print("✓ TEACHER: Created high-quality training examples")
    print("✓ CORRECTOR: Analyzed failures and provided corrections")
    print("✓ TIMER: Made adaptive timing decisions")
    print("✓ AUDITOR: Quality-checked neuron outputs")
    print("\nEcho's consciousness is operational and ready for integration.")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

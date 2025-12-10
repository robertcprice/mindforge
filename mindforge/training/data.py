"""
Training Data Generation

Generates training examples from consciousness cycles and grounding feedback
for continuous LoRA fine-tuning.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExampleType(Enum):
    """Types of training examples."""

    # Consciousness cycle examples
    THOUGHT_GENERATION = "thought_generation"
    DECISION_MAKING = "decision_making"
    REFLECTION = "reflection"
    SLEEP_DETERMINATION = "sleep_determination"

    # KVRM grounding examples
    CLAIM_CLASSIFICATION = "claim_classification"
    KEY_EXTRACTION = "key_extraction"
    GROUNDING_VERIFICATION = "grounding_verification"

    # Combined examples
    GROUNDED_THINKING = "grounded_thinking"
    VERIFIED_DECISION = "verified_decision"


@dataclass
class TrainingExample:
    """
    A single training example for LoRA fine-tuning.

    Includes input prompt, expected output, and metadata for filtering/weighting.
    """

    # Core content
    prompt: str
    completion: str
    example_type: ExampleType

    # Quality metrics
    quality_score: float = 1.0  # 0.0-1.0, higher = better quality
    verified: bool = False  # Was this example verified/validated?

    # Metadata
    source: str = ""  # Where this example came from
    timestamp: Optional[datetime] = None
    cycle_id: Optional[int] = None

    # Grounding metadata (for KVRM examples)
    grounding_confidence: float = 0.0
    claims_verified: int = 0
    claims_total: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def content_hash(self) -> str:
        """Hash of prompt + completion for deduplication."""
        content = f"{self.prompt}|{self.completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Rough estimate of token count."""
        # Rough estimate: 4 chars per token
        return (len(self.prompt) + len(self.completion)) // 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "example_type": self.example_type.value,
            "quality_score": self.quality_score,
            "verified": self.verified,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cycle_id": self.cycle_id,
            "grounding_confidence": self.grounding_confidence,
            "claims_verified": self.claims_verified,
            "claims_total": self.claims_total,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            completion=data["completion"],
            example_type=ExampleType(data["example_type"]),
            quality_score=data.get("quality_score", 1.0),
            verified=data.get("verified", False),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            cycle_id=data.get("cycle_id"),
            grounding_confidence=data.get("grounding_confidence", 0.0),
            claims_verified=data.get("claims_verified", 0),
            claims_total=data.get("claims_total", 0),
            metadata=data.get("metadata", {}),
        )

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to chat message format for training."""
        return [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.completion},
        ]

    def to_instruction_format(self) -> str:
        """Convert to instruction format string."""
        return f"### Instruction:\n{self.prompt}\n\n### Response:\n{self.completion}"


class TrainingDataset:
    """
    Dataset of training examples with filtering and sampling capabilities.
    """

    def __init__(self, path: Optional[Path] = None):
        """
        Initialize dataset.

        Args:
            path: Optional path to load/save dataset
        """
        self.path = path
        self.examples: List[TrainingExample] = []
        self._hash_index: Dict[str, int] = {}  # content_hash -> index

        if path and path.exists():
            self.load()

    def add(self, example: TrainingExample) -> bool:
        """
        Add an example to the dataset.

        Args:
            example: Training example to add

        Returns:
            True if added, False if duplicate
        """
        # Check for duplicates
        if example.content_hash in self._hash_index:
            logger.debug(f"Duplicate example skipped: {example.content_hash}")
            return False

        self._hash_index[example.content_hash] = len(self.examples)
        self.examples.append(example)
        return True

    def add_batch(self, examples: List[TrainingExample]) -> int:
        """Add multiple examples, returns count of new examples added."""
        added = 0
        for example in examples:
            if self.add(example):
                added += 1
        return added

    def filter(
        self,
        example_types: Optional[List[ExampleType]] = None,
        min_quality: float = 0.0,
        verified_only: bool = False,
        min_grounding_confidence: float = 0.0,
    ) -> List[TrainingExample]:
        """
        Filter examples by criteria.

        Args:
            example_types: Only include these types (None = all)
            min_quality: Minimum quality score
            verified_only: Only include verified examples
            min_grounding_confidence: Minimum grounding confidence

        Returns:
            List of matching examples
        """
        results = []

        for ex in self.examples:
            # Type filter
            if example_types and ex.example_type not in example_types:
                continue

            # Quality filter
            if ex.quality_score < min_quality:
                continue

            # Verified filter
            if verified_only and not ex.verified:
                continue

            # Grounding confidence filter
            if ex.grounding_confidence < min_grounding_confidence:
                continue

            results.append(ex)

        return results

    def sample(
        self,
        n: int,
        weighted: bool = True,
        example_types: Optional[List[ExampleType]] = None,
    ) -> List[TrainingExample]:
        """
        Sample n examples from the dataset.

        Args:
            n: Number of examples to sample
            weighted: Weight by quality score
            example_types: Filter to these types before sampling

        Returns:
            List of sampled examples
        """
        pool = self.filter(example_types=example_types) if example_types else self.examples

        if len(pool) <= n:
            return pool.copy()

        if weighted:
            weights = [ex.quality_score for ex in pool]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return random.choices(pool, weights=weights, k=n)

        return random.sample(pool, n)

    def get_by_type(self, example_type: ExampleType) -> List[TrainingExample]:
        """Get all examples of a specific type."""
        return [ex for ex in self.examples if ex.example_type == example_type]

    def statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.examples:
            return {"total": 0}

        by_type = {}
        for ex in self.examples:
            type_name = ex.example_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        verified = sum(1 for ex in self.examples if ex.verified)
        avg_quality = sum(ex.quality_score for ex in self.examples) / len(self.examples)
        total_tokens = sum(ex.token_estimate for ex in self.examples)

        return {
            "total": len(self.examples),
            "by_type": by_type,
            "verified_count": verified,
            "verified_ratio": verified / len(self.examples),
            "avg_quality": avg_quality,
            "total_tokens_estimate": total_tokens,
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save dataset to JSON file."""
        save_path = path or self.path
        if not save_path:
            raise ValueError("No path specified for saving")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "statistics": self.statistics(),
            "examples": [ex.to_dict() for ex in self.examples],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.examples)} examples to {save_path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load dataset from JSON file."""
        load_path = path or self.path
        if not load_path or not load_path.exists():
            raise ValueError(f"Path does not exist: {load_path}")

        with open(load_path) as f:
            data = json.load(f)

        self.examples = []
        self._hash_index = {}

        for ex_data in data.get("examples", []):
            example = TrainingExample.from_dict(ex_data)
            self.add(example)

        logger.info(f"Loaded {len(self.examples)} examples from {load_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


class ConsciousnessDataGenerator:
    """
    Generates training data from consciousness cycles.

    Extracts high-quality examples from:
    - Thought generation
    - Decision making
    - Reflection
    - Sleep determination
    """

    def __init__(
        self,
        min_quality_threshold: float = 0.6,
        include_failed_cycles: bool = False,
    ):
        """
        Initialize generator.

        Args:
            min_quality_threshold: Minimum quality to include
            include_failed_cycles: Include cycles with errors
        """
        self.min_quality_threshold = min_quality_threshold
        self.include_failed_cycles = include_failed_cycles

    def from_cycle_state(
        self,
        state: Dict[str, Any],
        system_prompt: str = "",
    ) -> List[TrainingExample]:
        """
        Generate training examples from a consciousness cycle state.

        Args:
            state: ConsciousnessState dictionary
            system_prompt: System prompt used in the cycle

        Returns:
            List of training examples
        """
        examples = []

        # Skip failed cycles unless configured to include
        if state.get("error") and not self.include_failed_cycles:
            return examples

        cycle_id = state.get("cycle_count", 0)

        # 1. Thought generation example
        if state.get("current_thought"):
            thought_example = self._create_thought_example(state, system_prompt, cycle_id)
            if thought_example and thought_example.quality_score >= self.min_quality_threshold:
                examples.append(thought_example)

        # 2. Decision making example
        if state.get("decision") and state.get("action_type"):
            decision_example = self._create_decision_example(state, system_prompt, cycle_id)
            if decision_example and decision_example.quality_score >= self.min_quality_threshold:
                examples.append(decision_example)

        # 3. Reflection example
        if state.get("reflection"):
            reflection_example = self._create_reflection_example(state, system_prompt, cycle_id)
            if reflection_example and reflection_example.quality_score >= self.min_quality_threshold:
                examples.append(reflection_example)

        # 4. Sleep determination example
        if state.get("requested_sleep") is not None:
            sleep_example = self._create_sleep_example(state, system_prompt, cycle_id)
            if sleep_example and sleep_example.quality_score >= self.min_quality_threshold:
                examples.append(sleep_example)

        return examples

    def _create_thought_example(
        self,
        state: Dict[str, Any],
        system_prompt: str,
        cycle_id: int,
    ) -> Optional[TrainingExample]:
        """Create thought generation training example."""
        thought = state.get("current_thought", "")
        if not thought or len(thought) < 20:
            return None

        # Build prompt similar to what was used
        needs_str = self._format_needs(state.get("needs", {}))
        memory_summary = state.get("memory_summary", "No recent memories.")

        prompt = f"""{system_prompt}

## Current State
**Cycle:** {cycle_id}
**Needs (0.0-1.0):**
{needs_str}

**Recent memories:**
{memory_summary}

## Task
Generate a spontaneous thought based on your current state and needs.

Your thought:"""

        # Quality based on thought coherence and relevance
        quality = self._assess_thought_quality(thought, state)

        return TrainingExample(
            prompt=prompt,
            completion=thought,
            example_type=ExampleType.THOUGHT_GENERATION,
            quality_score=quality,
            source="consciousness_cycle",
            cycle_id=cycle_id,
            metadata={"needs": state.get("needs", {})},
        )

    def _create_decision_example(
        self,
        state: Dict[str, Any],
        system_prompt: str,
        cycle_id: int,
    ) -> Optional[TrainingExample]:
        """Create decision making training example."""
        decision = state.get("decision", "")
        action_type = state.get("action_type", "")

        if not decision:
            return None

        # Use grounded thought if available
        thought = state.get("grounded_thought") or state.get("current_thought", "")
        needs_str = self._format_needs(state.get("needs", {}))

        # Include grounding info if available
        grounding_info = ""
        if state.get("grounding_enabled"):
            verified = state.get("verified_claims_count", 0)
            unverified = state.get("unverified_claims_count", 0)
            if verified > 0 or unverified > 0:
                grounding_info = f"""
## Grounding Status
- Verified claims: {verified}
- Unverified claims: {unverified}
"""

        prompt = f"""{system_prompt}

## Your Thought
{thought}
{grounding_info}
## Current Needs
{needs_str}

## Task
Based on your thought and needs, decide what to do.
Choose: TOOL: ..., DO_NOTHING: ..., or REFLECT: ...

Your decision:"""

        # Format completion
        completion = f"{action_type.upper()}: {decision}" if action_type != "reflect" else f"REFLECT: {decision}"

        # Quality based on decision appropriateness
        quality = self._assess_decision_quality(state)

        return TrainingExample(
            prompt=prompt,
            completion=completion,
            example_type=ExampleType.DECISION_MAKING,
            quality_score=quality,
            verified=state.get("grounding_enabled", False),
            source="consciousness_cycle",
            cycle_id=cycle_id,
            grounding_confidence=1.0 if state.get("verified_claims_count", 0) > 0 else 0.5,
            claims_verified=state.get("verified_claims_count", 0),
            claims_total=state.get("verified_claims_count", 0) + state.get("unverified_claims_count", 0),
        )

    def _create_reflection_example(
        self,
        state: Dict[str, Any],
        system_prompt: str,
        cycle_id: int,
    ) -> Optional[TrainingExample]:
        """Create reflection training example."""
        reflection = state.get("reflection", "")
        if not reflection or len(reflection) < 20:
            return None

        thought = state.get("current_thought", "")
        decision = state.get("decision", "")
        action_type = state.get("action_type", "")
        result = state.get("action_result", "")

        prompt = f"""{system_prompt}

## Cycle Summary
**Thought:** {thought[:200]}...
**Decision:** {decision[:100]}...
**Action Type:** {action_type}
**Result:** {result[:200]}...

## Task
Reflect on this cycle. What did you learn? What would you do differently?

Your reflection:"""

        quality = self._assess_reflection_quality(reflection, state)

        return TrainingExample(
            prompt=prompt,
            completion=reflection,
            example_type=ExampleType.REFLECTION,
            quality_score=quality,
            source="consciousness_cycle",
            cycle_id=cycle_id,
        )

    def _create_sleep_example(
        self,
        state: Dict[str, Any],
        system_prompt: str,
        cycle_id: int,
    ) -> Optional[TrainingExample]:
        """Create sleep determination training example."""
        sleep_duration = state.get("requested_sleep")
        sleep_reason = state.get("sleep_reason", "")

        if sleep_duration is None:
            return None

        needs_str = self._format_needs(state.get("needs", {}))

        prompt = f"""You just completed a consciousness cycle. Decide how long to rest.

## Cycle Summary
**Action:** {state.get('action_type', 'unknown')}
**Result:** {state.get('action_result', '')[:100]}...

## Current Needs
{needs_str}

## Guidelines
- 30s: urgent follow-up
- 60-120s: something interesting
- 180-300s: normal rest
- 300-600s: need rest
- 600+s: extended rest

Your decision (format: SLEEP: <seconds> - <reason>):"""

        completion = f"SLEEP: {int(sleep_duration)} - {sleep_reason}"

        return TrainingExample(
            prompt=prompt,
            completion=completion,
            example_type=ExampleType.SLEEP_DETERMINATION,
            quality_score=0.8,  # Sleep examples are generally reliable
            source="consciousness_cycle",
            cycle_id=cycle_id,
        )

    def _format_needs(self, needs: Dict[str, Any]) -> str:
        """Format needs dictionary for prompts."""
        lines = []
        for k, v in needs.items():
            if isinstance(v, dict):
                level = v.get("level", 0)
                lines.append(f"  - {k}: {level:.2f}")
            else:
                lines.append(f"  - {k}: {float(v):.2f}")
        return "\n".join(lines) if lines else "  (no needs data)"

    def _assess_thought_quality(self, thought: str, state: Dict[str, Any]) -> float:
        """Assess quality of a generated thought."""
        quality = 0.5  # Base quality

        # Length bonus (not too short, not too long)
        if 50 < len(thought) < 500:
            quality += 0.2
        elif len(thought) >= 20:
            quality += 0.1

        # Coherence bonus (basic check)
        if thought[0].isupper() and thought.rstrip().endswith((".", "!", "?")):
            quality += 0.1

        # Grounding bonus
        if state.get("grounding_enabled") and state.get("verified_claims_count", 0) > 0:
            quality += 0.2

        return min(quality, 1.0)

    def _assess_decision_quality(self, state: Dict[str, Any]) -> float:
        """Assess quality of a decision."""
        quality = 0.5

        # Action completed successfully
        if state.get("action_result") and "error" not in state.get("action_result", "").lower():
            quality += 0.2

        # Grounding bonus
        if state.get("grounding_enabled"):
            quality += 0.1
            if state.get("verified_claims_count", 0) > 0:
                quality += 0.1

        # Reflection exists (indicates complete cycle)
        if state.get("reflection"):
            quality += 0.1

        return min(quality, 1.0)

    def _assess_reflection_quality(self, reflection: str, state: Dict[str, Any]) -> float:
        """Assess quality of a reflection."""
        quality = 0.5

        # Length check
        if len(reflection) > 50:
            quality += 0.2

        # Contains learning indicators
        learning_words = ["learned", "realized", "understood", "insight", "next time"]
        if any(word in reflection.lower() for word in learning_words):
            quality += 0.2

        # Self-critical (good for learning)
        critical_words = ["better", "differently", "improve", "mistake"]
        if any(word in reflection.lower() for word in critical_words):
            quality += 0.1

        return min(quality, 1.0)


class GroundingDataGenerator:
    """
    Generates training data from KVRM grounding operations.

    Creates examples for:
    - Claim classification
    - Key extraction
    - Grounding verification
    """

    def __init__(self):
        """Initialize grounding data generator."""
        pass

    def from_grounding_results(
        self,
        thought: str,
        grounding_results: List[Dict[str, Any]],
    ) -> List[TrainingExample]:
        """
        Generate training examples from grounding results.

        Args:
            thought: Original thought that was grounded
            grounding_results: List of GroundingResult dictionaries

        Returns:
            List of training examples
        """
        examples = []

        for result in grounding_results:
            # 1. Claim classification example
            classification_example = self._create_classification_example(result)
            if classification_example:
                examples.append(classification_example)

            # 2. Key extraction example (for verified claims)
            if result.get("grounded") and result.get("key_used"):
                extraction_example = self._create_extraction_example(result)
                if extraction_example:
                    examples.append(extraction_example)

        return examples

    def _create_classification_example(
        self,
        result: Dict[str, Any],
    ) -> Optional[TrainingExample]:
        """Create claim classification training example."""
        claim = result.get("original", "")
        claim_type = result.get("claim_type", "unknown")

        if not claim:
            return None

        prompt = f"""Classify the following statement into one of these types:
- FACTUAL: Verifiable statement about the world
- MEMORY: Reference to past experience
- OPINION: Subjective assessment
- QUESTION: Interrogative statement
- CREATIVE: Imaginative content
- ACTION: Statement about intended action

Statement: "{claim}"

Classification:"""

        completion = claim_type.upper()

        # Quality based on confidence
        quality = 0.8 if result.get("confidence", 0) > 0.7 else 0.6

        return TrainingExample(
            prompt=prompt,
            completion=completion,
            example_type=ExampleType.CLAIM_CLASSIFICATION,
            quality_score=quality,
            verified=result.get("verified", False),
            source="grounding",
            grounding_confidence=result.get("confidence", 0),
        )

    def _create_extraction_example(
        self,
        result: Dict[str, Any],
    ) -> Optional[TrainingExample]:
        """Create key extraction training example."""
        claim = result.get("original", "")
        key_used = result.get("key_used", "")

        if not claim or not key_used:
            return None

        prompt = f"""Extract a verification key for this claim.

Available key formats:
- mem:{{type}}:{{date}}:{{hash}} - for memories
- fact:{{domain}}:{{id}} - for facts
- ext:{{source}}:{{path}} - for external sources
- UNKNOWN - if claim cannot be verified

Claim: "{claim}"

Key:"""

        completion = key_used

        return TrainingExample(
            prompt=prompt,
            completion=completion,
            example_type=ExampleType.KEY_EXTRACTION,
            quality_score=0.9,  # Verified key extractions are high quality
            verified=True,
            source="grounding",
            grounding_confidence=result.get("confidence", 1.0),
        )

    def from_verification_feedback(
        self,
        claim: str,
        predicted_verified: bool,
        actual_verified: bool,
        key_used: Optional[str] = None,
    ) -> Optional[TrainingExample]:
        """
        Create training example from verification feedback.

        Used when human or external validation confirms/denies verification.

        Args:
            claim: The claim that was verified
            predicted_verified: What the system predicted
            actual_verified: What the ground truth is
            key_used: Key used for verification (if any)

        Returns:
            Training example if feedback is useful
        """
        # Only create examples from corrections or confirmations
        if predicted_verified == actual_verified and actual_verified:
            # Confirmed verification - high quality positive example
            prompt = f"""Verify if this claim can be grounded against known facts.

Claim: "{claim}"

Can this claim be verified?"""

            completion = f"VERIFIED: This claim is verified against key {key_used}" if key_used else "VERIFIED"

            return TrainingExample(
                prompt=prompt,
                completion=completion,
                example_type=ExampleType.GROUNDING_VERIFICATION,
                quality_score=1.0,  # Human-confirmed
                verified=True,
                source="human_feedback",
                grounding_confidence=1.0,
            )

        elif predicted_verified and not actual_verified:
            # False positive - learn to not verify
            prompt = f"""Verify if this claim can be grounded against known facts.

Claim: "{claim}"

Can this claim be verified?"""

            completion = "UNVERIFIED: This claim cannot be reliably verified"

            return TrainingExample(
                prompt=prompt,
                completion=completion,
                example_type=ExampleType.GROUNDING_VERIFICATION,
                quality_score=1.0,  # Human-corrected
                verified=True,
                source="human_feedback",
                grounding_confidence=0.0,
            )

        return None

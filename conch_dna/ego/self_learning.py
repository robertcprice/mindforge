"""
Conch DNA - Self-Learning Module

When the EGO's confidence is below threshold (default 70%), it:
1. Consults external AI resources (Claude API, other models)
2. Gets the correct/better response
3. Trains itself on that response for future similar queries

This creates a feedback loop where the system continuously improves
by learning from more capable models when uncertain.

Architecture:
    Low Confidence Query → Consult External AI → Get Response → Train Self
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """A learning experience from external consultation."""
    query: str
    original_response: str
    original_confidence: float
    external_response: str
    external_source: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    quality_score: float = 0.0
    used_for_training: bool = False


@dataclass
class SelfLearningConfig:
    """Configuration for self-learning behavior."""
    confidence_threshold: float = 0.70  # Below this, consult external
    min_quality_for_training: float = 0.6  # Minimum quality to use for training
    experience_buffer_size: int = 100  # Max experiences before auto-train
    training_batch_size: int = 10
    anthropic_api_key: Optional[str] = None
    external_model: str = "claude-sonnet-4-20250514"  # Default external model
    enable_auto_training: bool = True


class ExternalConsultant:
    """Consults external AI models for better responses."""

    def __init__(self, config: SelfLearningConfig):
        self.config = config
        self._anthropic_client = None

    def _get_anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic
                api_key = self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("No Anthropic API key found. External consultation disabled.")
                    return None
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed. Run: pip install anthropic")
                return None
        return self._anthropic_client

    def consult_claude(self, query: str, context: Optional[str] = None) -> Optional[str]:
        """Consult Claude API for a response.

        Args:
            query: The question or task
            context: Optional additional context

        Returns:
            Claude's response or None if failed
        """
        client = self._get_anthropic_client()
        if not client:
            return None

        try:
            system_prompt = """You are an expert AI assistant helping another AI system learn.
Provide clear, accurate, and well-structured responses. The response should be suitable
for the other AI to learn from - be precise, include reasoning, and follow best practices."""

            if context:
                system_prompt += f"\n\nContext: {context}"

            message = client.messages.create(
                model=self.config.external_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )

            response = message.content[0].text
            logger.info(f"Received response from Claude ({len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Claude consultation failed: {e}")
            return None

    def consult_any(self, query: str, context: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Consult any available external AI.

        Tries Claude first, then falls back to other sources.

        Returns:
            Dict with 'response' and 'source' keys, or None
        """
        # Try Claude first (highest quality)
        response = self.consult_claude(query, context)
        if response:
            return {"response": response, "source": f"claude:{self.config.external_model}"}

        # Future: Add other models here (OpenAI, local models, etc.)

        logger.warning("No external AI sources available")
        return None


class SelfLearner:
    """Manages self-learning from external AI consultations.

    When confidence is low:
    1. Queries external AI for better response
    2. Stores the learning experience
    3. Periodically trains on accumulated experiences
    """

    def __init__(
        self,
        config: SelfLearningConfig,
        training_callback: Optional[Callable] = None
    ):
        """Initialize self-learner.

        Args:
            config: Self-learning configuration
            training_callback: Optional callback for custom training logic
        """
        self.config = config
        self.consultant = ExternalConsultant(config)
        self.experiences: List[LearningExperience] = []
        self.training_callback = training_callback
        self.data_dir = Path("data/learning_experiences")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SelfLearner initialized (threshold: {config.confidence_threshold})")

    def should_consult_external(self, confidence: float) -> bool:
        """Determine if external consultation is needed.

        Args:
            confidence: The model's confidence (0.0-1.0)

        Returns:
            True if confidence is below threshold
        """
        return confidence < self.config.confidence_threshold

    def learn_from_external(
        self,
        query: str,
        original_response: str,
        original_confidence: float,
        context: Optional[str] = None
    ) -> Optional[LearningExperience]:
        """Consult external AI and create learning experience.

        Args:
            query: The original query
            original_response: What the EGO produced
            original_confidence: EGO's confidence
            context: Additional context

        Returns:
            LearningExperience if successful, None otherwise
        """
        if not self.should_consult_external(original_confidence):
            logger.debug(f"Confidence {original_confidence:.2f} above threshold, no consultation needed")
            return None

        logger.info(f"Low confidence ({original_confidence:.2f}), consulting external AI...")

        result = self.consultant.consult_any(query, context)
        if not result:
            logger.warning("Failed to get external response")
            return None

        # Create learning experience
        experience = LearningExperience(
            query=query,
            original_response=original_response,
            original_confidence=original_confidence,
            external_response=result["response"],
            external_source=result["source"],
            quality_score=self._assess_quality(result["response"])
        )

        self.experiences.append(experience)
        logger.info(f"Created learning experience from {result['source']} (quality: {experience.quality_score:.2f})")

        # Auto-save experiences periodically
        if len(self.experiences) % 10 == 0:
            self._save_experiences()

        # Auto-train if buffer is full
        if self.config.enable_auto_training and len(self.experiences) >= self.config.experience_buffer_size:
            self.trigger_training()

        return experience

    def _assess_quality(self, response: str) -> float:
        """Assess the quality of an external response.

        Simple heuristics for now - could be enhanced with a quality model.
        """
        quality = 0.7  # Base quality for external responses

        # Length check - reasonable responses are 50-2000 chars
        length = len(response)
        if length < 50:
            quality -= 0.2
        elif length > 2000:
            quality -= 0.1
        elif 100 < length < 1000:
            quality += 0.1

        # Structure check - JSON or code blocks indicate structured output
        if "{" in response and "}" in response:
            quality += 0.1
        if "```" in response:
            quality += 0.05

        # Coherence check - ends properly
        if response.rstrip().endswith(('.', '!', '?', '"', '`', '}')):
            quality += 0.05

        return min(1.0, max(0.0, quality))

    def get_training_data(self) -> List[Dict[str, str]]:
        """Get accumulated experiences as training data.

        Only returns experiences meeting quality threshold.

        Returns:
            List of training samples
        """
        samples = []
        for exp in self.experiences:
            if exp.quality_score >= self.config.min_quality_for_training and not exp.used_for_training:
                samples.append({
                    "text": f"""<|im_start|>user
{exp.query}
<|im_end|>
<|im_start|>assistant
{exp.external_response}<|im_end|>""",
                    "source": exp.external_source,
                    "quality": exp.quality_score
                })

        return samples

    def trigger_training(self) -> bool:
        """Trigger training on accumulated experiences.

        Returns:
            True if training was performed
        """
        training_data = self.get_training_data()
        if len(training_data) < self.config.training_batch_size:
            logger.info(f"Not enough training data ({len(training_data)} < {self.config.training_batch_size})")
            return False

        logger.info(f"Triggering self-training on {len(training_data)} samples")

        # Use callback if provided
        if self.training_callback:
            success = self.training_callback(training_data)
            if success:
                self._mark_trained()
            return success

        # Default: save training data for later use
        self._save_training_data(training_data)
        self._mark_trained()
        return True

    def _mark_trained(self) -> None:
        """Mark experiences as used for training."""
        for exp in self.experiences:
            if exp.quality_score >= self.config.min_quality_for_training:
                exp.used_for_training = True

    def _save_experiences(self) -> None:
        """Save experiences to disk."""
        path = self.data_dir / f"experiences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(path, 'w') as f:
            for exp in self.experiences:
                f.write(json.dumps({
                    "query": exp.query,
                    "original_response": exp.original_response,
                    "original_confidence": exp.original_confidence,
                    "external_response": exp.external_response,
                    "external_source": exp.external_source,
                    "quality_score": exp.quality_score,
                    "timestamp": exp.timestamp,
                    "used_for_training": exp.used_for_training
                }) + '\n')
        logger.info(f"Saved {len(self.experiences)} experiences to {path}")

    def _save_training_data(self, training_data: List[Dict]) -> Path:
        """Save training data in format ready for fine-tuning."""
        path = self.data_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(path, 'w') as f:
            for sample in training_data:
                f.write(json.dumps({"text": sample["text"]}) + '\n')
        logger.info(f"Saved {len(training_data)} training samples to {path}")
        return path

    def get_stats(self) -> Dict[str, Any]:
        """Get self-learning statistics."""
        return {
            "total_experiences": len(self.experiences),
            "high_quality": len([e for e in self.experiences if e.quality_score >= 0.7]),
            "trained_on": len([e for e in self.experiences if e.used_for_training]),
            "pending_training": len([e for e in self.experiences
                                     if e.quality_score >= self.config.min_quality_for_training
                                     and not e.used_for_training]),
            "sources": list(set(e.external_source for e in self.experiences))
        }


class ConfidenceAwareEGO:
    """EGO wrapper that automatically learns from external sources when uncertain.

    This wraps the EGO model and adds self-learning capabilities:
    - If confidence >= threshold: use EGO response directly
    - If confidence < threshold: consult external AI, use that response, and learn from it
    """

    def __init__(
        self,
        ego_model,
        ego_tokenizer,
        config: Optional[SelfLearningConfig] = None
    ):
        """Initialize confidence-aware EGO.

        Args:
            ego_model: The loaded EGO model
            ego_tokenizer: The tokenizer
            config: Self-learning configuration
        """
        self.model = ego_model
        self.tokenizer = ego_tokenizer
        self.config = config or SelfLearningConfig()
        self.learner = SelfLearner(self.config)

        logger.info(f"ConfidenceAwareEGO initialized (threshold: {self.config.confidence_threshold})")

    def generate_with_learning(
        self,
        prompt: str,
        max_tokens: int = 512,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response with automatic learning on low confidence.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            context: Additional context for consultation

        Returns:
            Dict with 'response', 'confidence', 'source', and 'learned' keys
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Generate with EGO
        sampler = make_sampler(temp=0.7)
        ego_response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )

        # Estimate confidence (simplified - could use perplexity or other metrics)
        confidence = self._estimate_confidence(ego_response)

        result = {
            "response": ego_response,
            "confidence": confidence,
            "source": "ego",
            "learned": False
        }

        # If low confidence, learn from external
        if self.learner.should_consult_external(confidence):
            experience = self.learner.learn_from_external(
                query=prompt,
                original_response=ego_response,
                original_confidence=confidence,
                context=context
            )

            if experience:
                # Use external response instead
                result["response"] = experience.external_response
                result["source"] = experience.external_source
                result["learned"] = True
                result["original_ego_response"] = ego_response

        return result

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence in response.

        This is a simplified heuristic. In production, you could:
        - Use token log-probs from the model
        - Train a separate confidence estimator
        - Use ensemble disagreement
        """
        confidence = 0.7  # Base confidence

        # Length heuristics
        if len(response) < 20:
            confidence -= 0.3
        elif len(response) > 2000:
            confidence -= 0.1

        # Repetition check
        words = response.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                confidence -= 0.2

        # Structure check
        if "{" in response and "}" in response:
            confidence += 0.1

        # Hedging language check
        hedge_words = ["maybe", "might", "possibly", "not sure", "unclear"]
        for word in hedge_words:
            if word.lower() in response.lower():
                confidence -= 0.05

        return max(0.0, min(1.0, confidence))

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.learner.get_stats()


# CLI interface for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = SelfLearningConfig(
        confidence_threshold=0.7,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    learner = SelfLearner(config)

    # Test consultation
    print("Testing external consultation...")
    experience = learner.learn_from_external(
        query="What is the best way to implement a singleton pattern in Python?",
        original_response="Use a class variable maybe?",
        original_confidence=0.3,
        context="Teaching Python design patterns"
    )

    if experience:
        print(f"\nLearned from: {experience.external_source}")
        print(f"Quality: {experience.quality_score:.2f}")
        print(f"Response preview: {experience.external_response[:200]}...")
    else:
        print("No external response received (check API key)")

    print(f"\nStats: {learner.get_stats()}")

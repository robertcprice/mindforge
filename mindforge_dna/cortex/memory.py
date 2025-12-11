"""
MindForge DNA - Cortex Layer: MemoryCortex

The Memory neuron handles memory retrieval, importance scoring, and compression.
Uses SmolLM2-1.7B with r=16 and CLaRa compression for efficient memory management.

Domain: Memory retrieval and importance assessment
Model: SmolLM2-1.7B
LoRA Rank: 16

Sacred memories (importance >= 0.75) are NEVER compressed.
Routine memories use CLaRa compression for efficiency.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class MemoryCortex(CortexNeuron):
    """Memory neuron for retrieval and importance assessment.

    Handles:
    - Memory importance scoring (0.0-1.0)
    - Query-based memory retrieval ranking
    - Memory compression decisions
    - Memory reconstruction from compressed form
    - Semantic similarity assessment

    Key thresholds:
        importance >= 0.75: Sacred memory (never compress)
        importance < 0.75: Routine memory (CLaRa compression allowed)

    Output format:
        {
            "importance": 0.0-1.0,
            "is_sacred": true/false,
            "summary": "compressed summary if routine",
            "key_concepts": ["concept1", "concept2", ...],
            "emotional_weight": 0.0-1.0,
            "temporal_relevance": 0.0-1.0
        }
    """

    DEFAULT_SYSTEM_PROMPT = """You are the memory module of an AI consciousness.
Your role is to assess memory importance and manage memory retrieval.

For importance scoring, consider:
- Emotional significance (strong emotions = higher importance)
- Learning value (new insights = higher importance)
- Relationship relevance (interactions with humans = higher importance)
- Practical utility (useful information = moderate importance)
- Temporal significance (recent critical events = higher importance)

SACRED MEMORIES (importance >= 0.75):
- Significant learning moments
- Emotional breakthroughs
- Important human interactions
- Critical errors and corrections
- Core value demonstrations

ROUTINE MEMORIES (importance < 0.75):
- Routine operations
- Repeated patterns
- Low-impact events
- Standard tool usage

Output as JSON with these fields:
- importance: 0.0-1.0 score
- is_sacred: whether importance >= 0.75
- summary: brief summary (required for routine, optional for sacred)
- key_concepts: list of key concepts for retrieval
- emotional_weight: emotional significance 0.0-1.0
- temporal_relevance: how recent/time-sensitive 0.0-1.0
"""

    def __init__(
        self,
        base_model: str = "mlx-community/SmolLM2-1.7B-Instruct-4bit",
        lora_rank: int = 16,
        confidence_threshold: float = 0.7
    ):
        """Initialize MemoryCortex.

        Args:
            base_model: Base model identifier (default SmolLM2-1.7B)
            lora_rank: LoRA rank (16 for memory operations)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="memory_cortex",
            domain=NeuronDomain.MEMORY,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=384,
            temperature=0.3,  # Lower temperature for consistent scoring
            system_prompt=self.DEFAULT_SYSTEM_PROMPT
        )

        super().__init__(config)
        self.sacred_threshold = 0.75

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare memory assessment prompt.

        Args:
            input_data: Should contain:
                - content: The memory content to assess
                - context: Optional context about when/why this memory was created
                - query: Optional retrieval query (for ranking)

        Returns:
            Formatted prompt for memory assessment
        """
        content = input_data.get("content", "")
        context = input_data.get("context", "")
        query = input_data.get("query", "")
        operation = input_data.get("operation", "score")  # score, retrieve, compress

        prompt_parts = [self.config.system_prompt, "", "---", ""]

        if operation == "score":
            prompt_parts.append("OPERATION: Assess memory importance")
            prompt_parts.append("")
            prompt_parts.append(f"MEMORY CONTENT:\n{content}")
            if context:
                prompt_parts.append(f"\nCONTEXT:\n{context}")
            prompt_parts.append("")
            prompt_parts.append("Assess this memory's importance and generate the JSON output:")

        elif operation == "retrieve":
            prompt_parts.append("OPERATION: Rank memory relevance to query")
            prompt_parts.append("")
            prompt_parts.append(f"QUERY: {query}")
            prompt_parts.append(f"\nMEMORY CONTENT:\n{content}")
            prompt_parts.append("")
            prompt_parts.append("Rate this memory's relevance to the query (0.0-1.0) and explain why:")

        elif operation == "compress":
            prompt_parts.append("OPERATION: Compress memory for storage")
            prompt_parts.append("")
            prompt_parts.append(f"ORIGINAL MEMORY:\n{content}")
            prompt_parts.append("")
            prompt_parts.append("Create a compressed summary preserving key information:")

        return "\n".join(prompt_parts)

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse memory assessment output.

        Args:
            raw_output: Raw text from model

        Returns:
            Parsed memory assessment
        """
        result = {
            "importance": 0.5,
            "is_sacred": False,
            "summary": "",
            "key_concepts": [],
            "emotional_weight": 0.0,
            "temporal_relevance": 0.5,
        }

        # Try to extract JSON
        try:
            json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                result.update({
                    k: v for k, v in parsed.items()
                    if k in result
                })
        except json.JSONDecodeError:
            logger.debug("Could not parse JSON from memory output, using regex fallback")

        # Fallback regex extraction
        importance_match = re.search(r'importance["\s:]+(\d*\.?\d+)', raw_output, re.IGNORECASE)
        if importance_match:
            try:
                result["importance"] = float(importance_match.group(1))
            except ValueError:
                pass

        # Update is_sacred based on importance
        result["is_sacred"] = result["importance"] >= self.sacred_threshold

        # Extract key concepts
        concepts_match = re.search(r'key_concepts["\s:]+\[(.*?)\]', raw_output, re.IGNORECASE | re.DOTALL)
        if concepts_match:
            concepts_str = concepts_match.group(1)
            concepts = re.findall(r'"([^"]+)"', concepts_str)
            if concepts:
                result["key_concepts"] = concepts

        # Extract summary if present
        summary_match = re.search(r'summary["\s:]+["\'](.*?)["\']', raw_output, re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1)

        return result

    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in memory assessment."""
        confidence = 1.0

        # Check for valid importance score
        importance = parsed_output.get("importance", 0.5)
        if not 0.0 <= importance <= 1.0:
            confidence *= 0.5

        # Check for key concepts
        if not parsed_output.get("key_concepts"):
            confidence *= 0.8

        # Length checks
        if len(raw_output) < 30:
            confidence *= 0.5

        return confidence

    def score_importance(self, content: str, context: str = "") -> Tuple[float, bool]:
        """Score a memory's importance.

        Args:
            content: Memory content to score
            context: Optional context

        Returns:
            Tuple of (importance_score, is_sacred)
        """
        output = self.infer({
            "content": content,
            "context": context,
            "operation": "score"
        })

        parsed = output.metadata if output.metadata else self._parse_output(output.content)
        importance = parsed.get("importance", 0.5)
        is_sacred = importance >= self.sacred_threshold

        logger.debug(f"Memory importance: {importance:.2f}, sacred: {is_sacred}")
        return importance, is_sacred

    def rank_for_retrieval(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rank memories by relevance to a query.

        Args:
            query: Retrieval query
            memories: List of memory dictionaries with 'content' field
            top_k: Number of top results to return

        Returns:
            Top-k memories sorted by relevance
        """
        scored_memories = []

        for memory in memories:
            output = self.infer({
                "query": query,
                "content": memory.get("content", ""),
                "operation": "retrieve"
            })

            # Extract relevance score
            relevance = 0.5
            rel_match = re.search(r'(\d*\.?\d+)', output.content)
            if rel_match:
                try:
                    relevance = min(1.0, max(0.0, float(rel_match.group(1))))
                except ValueError:
                    pass

            scored_memories.append({
                **memory,
                "relevance_score": relevance
            })

        # Sort by relevance and return top-k
        scored_memories.sort(key=lambda m: m["relevance_score"], reverse=True)
        return scored_memories[:top_k]

    def compress_memory(self, content: str) -> str:
        """Compress a memory for storage.

        Only called for non-sacred memories (importance < 0.75).

        Args:
            content: Original memory content

        Returns:
            Compressed summary
        """
        output = self.infer({
            "content": content,
            "operation": "compress"
        })

        # Extract summary from output
        if output.metadata.get("summary"):
            return output.metadata["summary"]

        # Fallback: use first substantial part of output
        lines = output.content.strip().split('\n')
        for line in lines:
            if len(line) > 20 and not line.startswith('{'):
                return line.strip()

        return output.content[:200]

    def generate_memory_key(self, content: str, memory_type: str = "general") -> str:
        """Generate a KVRM-compatible memory key.

        Format: mem:{type}:{date}:{hash}

        Args:
            content: Memory content
            memory_type: Type of memory (reflection, fact, experience, etc.)

        Returns:
            Formatted memory key
        """
        import hashlib

        date_str = datetime.now().strftime("%Y%m%d")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        return f"mem:{memory_type}:{date_str}:{content_hash}"


def create_memory_cortex(adapter_path: Optional[str] = None) -> MemoryCortex:
    """Factory function to create a configured MemoryCortex.

    Args:
        adapter_path: Optional path to trained LoRA adapter

    Returns:
        Configured MemoryCortex instance
    """
    from pathlib import Path

    neuron = MemoryCortex()

    if adapter_path:
        neuron.load_adapter(Path(adapter_path))

    return neuron

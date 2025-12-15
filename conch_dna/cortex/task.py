"""
Conch DNA - Cortex Layer: TaskCortex

The task neuron extracts and prioritizes tasks from thoughts.
Uses Qwen3-1.7B with r=8 for efficiency.

Domain: Task extraction and prioritization
Model: Qwen3-1.7B
LoRA Rank: 8
"""

import json
import logging
from typing import Any, Dict, List

from .base import CortexNeuron, NeuronConfig, NeuronDomain, NeuronOutput

logger = logging.getLogger(__name__)


class TaskCortex(CortexNeuron):
    """Task extraction and prioritization neuron.

    Analyzes thoughts and extracts actionable tasks, ranking them by:
    - Urgency (time-sensitive)
    - Importance (impact on goals)
    - Dependencies (what must happen first)
    - Resource requirements

    Output format:
        {
            "new_tasks": [
                {
                    "description": "task description",
                    "priority": "high|medium|low",
                    "urgency": 0.0-1.0,
                    "importance": 0.0-1.0,
                    "dependencies": ["task_id1", ...] or null,
                    "estimated_effort": "quick|moderate|substantial"
                },
                ...
            ],
            "ranked_task_ids": ["id1", "id2", ...]  # For reranking pending
        }
    """

    DEFAULT_SYSTEM_PROMPT = """/no_think
You output ONLY valid JSON. No explanations. No markdown. No prose.

Your response MUST start with { and end with }

Required JSON structure:
{"new_tasks": [{"description": "task text", "priority": "high|medium|low", "urgency": 0.0-1.0, "importance": 0.0-1.0, "dependencies": [], "estimated_effort": "quick|moderate|substantial"}], "ranked_task_ids": []}

Example:
{"new_tasks": [{"description": "Draft architecture overview", "priority": "high", "urgency": 0.8, "importance": 0.9, "dependencies": [], "estimated_effort": "moderate"}], "ranked_task_ids": []}"""

    def __init__(
        self,
        base_model: str = "mlx-community/Qwen3-1.7B-4bit",
        lora_rank: int = 8,
        confidence_threshold: float = 0.65
    ):
        """Initialize TaskCortex.

        Args:
            base_model: Base model identifier
            lora_rank: LoRA rank (default 8, sufficient for task extraction)
            confidence_threshold: Minimum confidence before EGO fallback
        """
        config = NeuronConfig(
            name="task_cortex",
            domain=NeuronDomain.TASK,
            base_model=base_model,
            lora_rank=lora_rank,
            confidence_threshold=confidence_threshold,
            max_tokens=512,  # Capped for controlled output
            temperature=0.2,  # Low for deterministic JSON
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            repetition_penalty=1.3,
        )
        super().__init__(config)
        # Expected fields for confidence estimation
        self._expected_fields = ["new_tasks", "ranked_task_ids"]

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """Prepare task extraction prompt.

        Args:
            input_data: Must contain:
                - thought: str - thought to extract tasks from
                - pending_tasks: list - existing tasks (optional, for reranking)
                - context: str - additional context (optional)

        Returns:
            Formatted prompt
        """
        thought = input_data.get("thought", "")
        pending_tasks = input_data.get("pending_tasks", [])
        context = input_data.get("context", "")

        # Format pending tasks
        pending_str = ""
        if pending_tasks:
            task_items = []
            for i, task in enumerate(pending_tasks[:10]):  # Limit to 10
                desc = task.get("description", str(task))
                task_items.append(f"{i}. {desc}")
            pending_str = "Pending tasks:\n" + "\n".join(task_items)

        prompt = f"""{self.config.system_prompt}

THOUGHT:
{thought}

{pending_str}

{f"CONTEXT:\n{context}" if context else ""}

Respond with JSON only. Begin your response with {{"""

        return prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse task extraction output.

        Args:
            raw_output: Raw model output

        Returns:
            Parsed task structure
        """
        try:
            json_text = self._extract_first_json(raw_output)
            if json_text:
                task_data = json.loads(json_text)

                # Ensure required fields exist
                if "new_tasks" not in task_data:
                    task_data["new_tasks"] = []
                if "ranked_task_ids" not in task_data:
                    task_data["ranked_task_ids"] = []

                # Validate task structure
                for task in task_data["new_tasks"]:
                    if "description" not in task:
                        task["description"] = "Unnamed task"
                    if "priority" not in task:
                        task["priority"] = "medium"
                    if "urgency" not in task:
                        task["urgency"] = 0.5
                    if "importance" not in task:
                        task["importance"] = 0.5

                return task_data

        except Exception as e:
            logger.error(f"{self.config.name}: JSON parse error: {e}")

        # Heuristic fallback: extract bullet lines as tasks
        tasks = []
        seen = set()
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("the tasks"):
                continue
            if stripped[0] in "-•*":
                desc = stripped.lstrip("-•* ").strip().strip(".")
                if desc and desc not in seen:
                    seen.add(desc)
                    tasks.append({
                        "description": desc,
                        "priority": "high" if not tasks else "medium",
                        "urgency": 0.8 if not tasks else 0.5,
                        "importance": 0.8 if not tasks else 0.6,
                        "dependencies": [],
                        "estimated_effort": "moderate"
                    })
                if len(tasks) >= 5:
                    break
        if tasks:
            return {
                "new_tasks": tasks,
                "ranked_task_ids": []
            }

        logger.warning(f"{self.config.name}: Could not parse tasks; returning empty set")
        return {
            "new_tasks": [],
            "ranked_task_ids": [],
            "error": "no tasks parsed"
        }


    def _estimate_confidence(
        self,
        input_data: Dict[str, Any],
        raw_output: str,
        parsed_output: Dict[str, Any]
    ) -> float:
        """Estimate confidence in task extraction quality.

        Args:
            input_data: Input provided
            raw_output: Raw model output
            parsed_output: Parsed task structure

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = super()._estimate_confidence(input_data, raw_output, parsed_output)

        # Check if we got any tasks
        new_tasks = parsed_output.get("new_tasks", [])
        if not new_tasks and not parsed_output.get("ranked_task_ids"):
            confidence *= 0.5  # No output is suspicious

        # Check task completeness
        if new_tasks:
            complete_tasks = 0
            for task in new_tasks:
                required = ["description", "priority", "urgency", "importance"]
                if all(k in task for k in required):
                    complete_tasks += 1

            completeness_ratio = complete_tasks / len(new_tasks)
            confidence *= (0.5 + 0.5 * completeness_ratio)

        # Check for reasonable task descriptions
        if new_tasks:
            avg_length = sum(len(t.get("description", "")) for t in new_tasks) / len(new_tasks)
            if avg_length < 10:
                confidence *= 0.6  # Too brief
            elif avg_length > 200:
                confidence *= 0.8  # Too verbose

        # Check priority distribution (not everything should be high)
        if len(new_tasks) > 1:
            high_count = sum(1 for t in new_tasks if t.get("priority") == "high")
            if high_count == len(new_tasks):
                confidence *= 0.7  # Everything high is suspicious

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def extract_tasks(
        self,
        thought: str,
        pending_tasks: List[Dict[str, Any]] = None,
        context: str = ""
    ) -> NeuronOutput:
        """High-level task extraction interface.

        Args:
            thought: Thought text to extract tasks from
            pending_tasks: Existing tasks for reranking (optional)
            context: Additional context (optional)

        Returns:
            NeuronOutput with task structure in metadata
        """
        input_data = {
            "thought": thought,
            "pending_tasks": pending_tasks or [],
            "context": context
        }

        output = self.infer(input_data)
        output.metadata["input_data"] = input_data

        return output

    def get_new_tasks(self, output: NeuronOutput) -> List[Dict[str, Any]]:
        """Extract new tasks from output.

        Args:
            output: NeuronOutput from extract_tasks()

        Returns:
            List of new task dictionaries
        """
        return output.metadata.get("new_tasks", [])

    def get_ranked_ids(self, output: NeuronOutput) -> List[str]:
        """Extract ranked task IDs from output.

        Args:
            output: NeuronOutput from extract_tasks()

        Returns:
            List of task IDs in priority order
        """
        return output.metadata.get("ranked_task_ids", [])

    def prioritize(
        self,
        tasks: List[Dict[str, Any]],
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """Prioritize a list of tasks.

        Args:
            tasks: List of task dictionaries
            context: Additional context for prioritization

        Returns:
            Tasks sorted by priority
        """
        # Create thought from tasks
        thought = f"Prioritize these {len(tasks)} tasks based on urgency and importance."

        output = self.extract_tasks(thought, pending_tasks=tasks, context=context)

        # Get ranked IDs
        ranked_ids = self.get_ranked_ids(output)

        # Reorder tasks based on ranking
        if ranked_ids and len(ranked_ids) == len(tasks):
            id_to_task = {i: task for i, task in enumerate(tasks)}
            try:
                return [id_to_task[int(task_id)] for task_id in ranked_ids]
            except (KeyError, ValueError):
                logger.warning("Could not reorder tasks, returning original order")

        # Fallback: sort by priority and urgency
        priority_map = {"high": 3, "medium": 2, "low": 1}
        return sorted(
            tasks,
            key=lambda t: (
                priority_map.get(t.get("priority", "medium"), 2),
                t.get("urgency", 0.5)
            ),
            reverse=True
        )


# Factory function
def create_task_cortex(
    base_model: str = "mlx-community/Qwen3-1.7B-4bit",
    adapter_path: str = None
) -> TaskCortex:
    """Create and optionally load TaskCortex.

    Args:
        base_model: Base model identifier
        adapter_path: Optional path to LoRA adapter

    Returns:
        Configured TaskCortex
    """
    from pathlib import Path

    cortex = TaskCortex(base_model=base_model)

    if adapter_path:
        cortex.load_adapter(Path(adapter_path))

    return cortex

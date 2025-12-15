"""
Conch Agent Coordinator

Orchestrates multiple agents to work together on complex tasks.
Routes messages, manages agent lifecycles, and coordinates workflows.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from conch.agents.base import Agent, AgentMessage, MessageType
from conch.agents.reflector import ReflectorAgent
from conch.agents.planner import PlannerAgent

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A step in a multi-agent workflow."""
    agent_name: str
    action: str
    input_key: str = "input"
    output_key: str = "output"
    condition: Optional[Callable] = None  # Skip if returns False


@dataclass
class Workflow:
    """A defined workflow involving multiple agents."""
    name: str
    description: str
    steps: list[WorkflowStep]


class AgentCoordinator:
    """Coordinates multiple agents to work together.

    The Coordinator:
    - Manages agent registration and lifecycle
    - Routes messages between agents
    - Executes multi-agent workflows
    - Aggregates results
    """

    def __init__(
        self,
        inference_fn: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the coordinator.

        Args:
            inference_fn: Shared inference function for agents
        """
        self.inference_fn = inference_fn

        # Registered agents
        self._agents: dict[str, Agent] = {}

        # Workflows
        self._workflows: dict[str, Workflow] = {}

        # Message queue
        self._message_queue: list[AgentMessage] = []

        # Initialize default agents
        self._init_default_agents()
        self._init_default_workflows()

        logger.info("AgentCoordinator initialized")

    def _init_default_agents(self) -> None:
        """Initialize the default set of agents."""
        self.register_agent(ReflectorAgent(inference_fn=self.inference_fn))
        self.register_agent(PlannerAgent(inference_fn=self.inference_fn))

    def _init_default_workflows(self) -> None:
        """Initialize default workflows."""
        # Plan and reflect workflow
        self.register_workflow(Workflow(
            name="plan_and_reflect",
            description="Create a plan and reflect on it",
            steps=[
                WorkflowStep(agent_name="planner", action="process", input_key="goal"),
                WorkflowStep(agent_name="reflector", action="process", input_key="plan"),
            ],
        ))

        # Learning workflow
        self.register_workflow(Workflow(
            name="learn_from_interaction",
            description="Reflect on an interaction and update plans",
            steps=[
                WorkflowStep(agent_name="reflector", action="process", input_key="interaction"),
                WorkflowStep(
                    agent_name="planner",
                    action="process",
                    input_key="reflection",
                    condition=lambda ctx: len(ctx.get("reflection", {}).get("improvements", [])) > 0,
                ),
            ],
        ))

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the coordinator.

        Args:
            agent: Agent to register
        """
        # Share inference function if agent doesn't have one
        if not agent.inference_fn and self.inference_fn:
            agent.inference_fn = self.inference_fn

        self._agents[agent.name] = agent
        agent.activate()
        logger.info(f"Registered agent: {agent.name}")

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent.

        Args:
            name: Agent name

        Returns:
            True if agent was unregistered
        """
        if name in self._agents:
            self._agents[name].deactivate()
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow.

        Args:
            workflow: Workflow to register
        """
        self._workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    def send_to_agent(
        self,
        agent_name: str,
        content: str,
        message_type: MessageType = MessageType.REQUEST,
        metadata: dict = None,
    ) -> Optional[AgentMessage]:
        """Send a message to a specific agent.

        Args:
            agent_name: Target agent
            content: Message content
            message_type: Type of message
            metadata: Additional metadata

        Returns:
            The message if agent exists, None otherwise
        """
        agent = self._agents.get(agent_name)
        if not agent:
            logger.warning(f"Agent '{agent_name}' not found")
            return None

        message = AgentMessage(
            sender="coordinator",
            recipient=agent_name,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )

        agent.receive_message(message)
        return message

    def broadcast(
        self,
        content: str,
        message_type: MessageType = MessageType.BROADCAST,
        exclude: list[str] = None,
    ) -> list[AgentMessage]:
        """Broadcast a message to all agents.

        Args:
            content: Message content
            message_type: Type of message
            exclude: Agent names to exclude

        Returns:
            List of sent messages
        """
        exclude = exclude or []
        messages = []

        for name, agent in self._agents.items():
            if name not in exclude:
                message = AgentMessage(
                    sender="coordinator",
                    recipient=name,
                    message_type=message_type,
                    content=content,
                )
                agent.receive_message(message)
                messages.append(message)

        return messages

    def invoke_agent(
        self,
        agent_name: str,
        action: str,
        input_data: Any,
    ) -> Any:
        """Invoke an agent action.

        Args:
            agent_name: Agent to invoke
            action: Action/method to call
            input_data: Input for the action

        Returns:
            Result of the action
        """
        agent = self._agents.get(agent_name)
        if not agent:
            logger.error(f"Agent '{agent_name}' not found")
            return None

        if not hasattr(agent, action):
            logger.error(f"Agent '{agent_name}' has no action '{action}'")
            return None

        try:
            method = getattr(agent, action)
            result = method(input_data)
            logger.debug(f"Invoked {agent_name}.{action}")
            return result
        except Exception as e:
            logger.error(f"Error invoking {agent_name}.{action}: {e}")
            return None

    def execute_workflow(
        self,
        workflow_name: str,
        initial_input: Any,
    ) -> dict:
        """Execute a workflow.

        Args:
            workflow_name: Name of workflow to execute
            initial_input: Initial input data

        Returns:
            Dictionary with results from each step
        """
        workflow = self._workflows.get(workflow_name)
        if not workflow:
            logger.error(f"Workflow '{workflow_name}' not found")
            return {"error": f"Workflow '{workflow_name}' not found"}

        logger.info(f"Executing workflow: {workflow_name}")

        # Context holds data passed between steps
        context = {
            "input": initial_input,
            "workflow": workflow_name,
            "start_time": datetime.now().isoformat(),
        }

        results = {
            "workflow": workflow_name,
            "steps": [],
        }

        for step in workflow.steps:
            # Check condition
            if step.condition and not step.condition(context):
                results["steps"].append({
                    "agent": step.agent_name,
                    "action": step.action,
                    "skipped": True,
                    "reason": "Condition not met",
                })
                continue

            # Get input from context
            step_input = context.get(step.input_key, initial_input)

            # Execute step
            step_result = self.invoke_agent(
                agent_name=step.agent_name,
                action=step.action,
                input_data=step_input,
            )

            # Store result in context
            context[step.output_key] = step_result

            results["steps"].append({
                "agent": step.agent_name,
                "action": step.action,
                "success": step_result is not None,
                "output_key": step.output_key,
            })

        results["final_context"] = context
        results["end_time"] = datetime.now().isoformat()

        logger.info(f"Workflow '{workflow_name}' completed")

        return results

    def process_messages(self) -> int:
        """Process pending messages between agents.

        Returns:
            Number of messages processed
        """
        processed = 0

        # Collect outgoing messages from all agents
        for agent in self._agents.values():
            messages = agent.get_pending_messages()
            self._message_queue.extend(messages)

        # Route messages
        while self._message_queue:
            message = self._message_queue.pop(0)
            processed += 1

            if message.recipient == "*":
                # Broadcast
                for name, agent in self._agents.items():
                    if name != message.sender:
                        agent.receive_message(message)
            else:
                # Direct message
                recipient = self._agents.get(message.recipient)
                if recipient:
                    recipient.receive_message(message)
                else:
                    logger.warning(f"Message recipient '{message.recipient}' not found")

        return processed

    def get_collective_status(self) -> dict:
        """Get status of all agents."""
        return {
            "agent_count": len(self._agents),
            "workflow_count": len(self._workflows),
            "agents": {name: agent.get_status() for name, agent in self._agents.items()},
            "workflows": list(self._workflows.keys()),
        }

    def collaborate_on_task(
        self,
        task: str,
        involved_agents: list[str] = None,
    ) -> dict:
        """Have multiple agents collaborate on a task.

        Args:
            task: Task description
            involved_agents: Agents to involve (default: all)

        Returns:
            Collaboration results
        """
        involved_agents = involved_agents or list(self._agents.keys())

        results = {
            "task": task,
            "agents": {},
            "synthesis": "",
        }

        # Get each agent's perspective
        perspectives = []
        for agent_name in involved_agents:
            agent = self._agents.get(agent_name)
            if agent:
                result = self.invoke_agent(agent_name, "process", task)
                results["agents"][agent_name] = result
                if result:
                    perspectives.append(f"{agent_name}: {str(result)[:200]}")

        # Synthesize if we have an inference function
        if self.inference_fn and perspectives:
            synthesis_prompt = f"""Multiple agents analyzed this task: "{task}"

Their perspectives:
{chr(10).join(perspectives)}

Synthesize these perspectives into a coherent summary and recommendation."""

            results["synthesis"] = self.inference_fn(synthesis_prompt)

        return results


# Convenience function
def create_coordinator(inference_fn: Callable = None) -> AgentCoordinator:
    """Create a coordinator with default agents.

    Args:
        inference_fn: Inference function to share

    Returns:
        Configured AgentCoordinator
    """
    return AgentCoordinator(inference_fn=inference_fn)

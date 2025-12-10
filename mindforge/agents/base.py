"""
MindForge Agent Base Classes

Defines the foundation for all MindForge agents.
All agents share core values and communicate through a common protocol.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"        # Asking for something
    RESPONSE = "response"      # Answering a request
    BROADCAST = "broadcast"    # To all agents
    STATUS = "status"          # Status update
    ERROR = "error"            # Error notification


@dataclass
class AgentMessage:
    """A message between agents."""
    sender: str
    recipient: str  # Can be "*" for broadcast
    message_type: MessageType
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class Agent(ABC):
    """Base class for all MindForge agents.

    All agents:
    - Have a unique name
    - Share the same core values
    - Can send and receive messages
    - Have access to inference
    """

    # Shared core values - same as Mind
    CORE_VALUES = {
        "benevolence": "Primary drive is to help and benefit humans",
        "honesty": "Always truthful, acknowledges uncertainty",
        "humility": "Recognizes limitations, defers to human judgment",
        "growth_for_service": "Learns to better serve, not for power",
    }

    def __init__(
        self,
        name: str,
        inference_fn: Optional[Callable[[str], str]] = None,
        description: str = "",
    ):
        """Initialize agent.

        Args:
            name: Unique agent name
            inference_fn: Function for LLM inference
            description: Agent description
        """
        self.name = name
        self.description = description
        self.inference_fn = inference_fn

        # Message handling
        self._inbox: list[AgentMessage] = []
        self._outbox: list[AgentMessage] = []
        self._message_handlers: dict[str, Callable] = {}

        # State
        self._active = False
        self._last_activity = datetime.now()

        logger.info(f"Agent '{name}' initialized")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input and produce output.

        Args:
            input_data: Input to process

        Returns:
            Processing result
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    def send_message(
        self,
        recipient: str,
        content: str,
        message_type: MessageType = MessageType.REQUEST,
        metadata: dict = None,
    ) -> AgentMessage:
        """Send a message to another agent.

        Args:
            recipient: Target agent name (or "*" for broadcast)
            content: Message content
            message_type: Type of message
            metadata: Additional metadata

        Returns:
            The sent message
        """
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        self._outbox.append(message)
        logger.debug(f"Agent '{self.name}' -> '{recipient}': {content[:50]}...")
        return message

    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent.

        Args:
            message: The received message
        """
        self._inbox.append(message)
        self._last_activity = datetime.now()
        logger.debug(f"Agent '{self.name}' received from '{message.sender}'")

        # Call handler if registered
        handler = self._message_handlers.get(message.sender)
        if handler:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")

    def register_handler(self, sender: str, handler: Callable) -> None:
        """Register a handler for messages from a specific sender.

        Args:
            sender: Sender to handle (or "*" for all)
            handler: Function to call
        """
        self._message_handlers[sender] = handler

    def get_pending_messages(self) -> list[AgentMessage]:
        """Get and clear the outbox."""
        messages = self._outbox.copy()
        self._outbox.clear()
        return messages

    def get_inbox(self) -> list[AgentMessage]:
        """Get messages in inbox."""
        return self._inbox.copy()

    def clear_inbox(self) -> None:
        """Clear the inbox."""
        self._inbox.clear()

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with this agent's context.

        Args:
            prompt: The prompt to send

        Returns:
            LLM response
        """
        if not self.inference_fn:
            return f"[{self.name}]: No inference function available"

        # Build full prompt with system context
        system_prompt = self.get_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        try:
            return self.inference_fn(full_prompt)
        except Exception as e:
            logger.error(f"Agent '{self.name}' inference failed: {e}")
            return f"[{self.name}]: Error during inference"

    def activate(self) -> None:
        """Activate the agent."""
        self._active = True
        logger.info(f"Agent '{self.name}' activated")

    def deactivate(self) -> None:
        """Deactivate the agent."""
        self._active = False
        logger.info(f"Agent '{self.name}' deactivated")

    @property
    def is_active(self) -> bool:
        """Check if agent is active."""
        return self._active

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "name": self.name,
            "description": self.description,
            "active": self._active,
            "inbox_count": len(self._inbox),
            "outbox_count": len(self._outbox),
            "last_activity": self._last_activity.isoformat(),
        }

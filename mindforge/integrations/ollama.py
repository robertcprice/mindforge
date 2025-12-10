"""
MindForge Ollama Integration

Interface with Ollama for local model serving and inference.
Serves as fallback/alternative to MLX for inference.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Union

import httpx

from mindforge.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Information about an Ollama model."""

    name: str
    modified_at: datetime
    size: int
    digest: str
    details: Optional[dict] = None


@dataclass
class ChatMessage:
    """A chat message for Ollama."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class GenerateResponse:
    """Response from Ollama generation."""

    response: str
    model: str
    done: bool
    context: Optional[list] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """Client for interacting with Ollama API.

    Ollama provides fast local inference and is used as a fallback
    when MLX is not available or for specific use cases.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server URL
        """
        self.host = host.rstrip("/")
        self._client = httpx.Client(
            base_url=self.host,
            timeout=300.0,  # Long timeout for generation
        )

    def is_healthy(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self._client.get("/")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> list[OllamaModel]:
        """List all available models."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("models", []):
                # Parse modified_at - handle nanoseconds by truncating to microseconds
                modified_str = m["modified_at"].replace("Z", "+00:00")
                # Truncate nanoseconds to 6 digits (microseconds)
                import re
                modified_str = re.sub(r'(\.\d{6})\d+', r'\1', modified_str)
                try:
                    modified_at = datetime.fromisoformat(modified_str)
                except ValueError:
                    modified_at = datetime.now()  # Fallback

                models.append(OllamaModel(
                    name=m["name"],
                    modified_at=modified_at,
                    size=m["size"],
                    digest=m["digest"],
                    details=m.get("details"),
                ))
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists."""
        models = self.list_models()
        return any(m.name == model_name or m.name.startswith(f"{model_name}:") for m in models)

    def pull_model(self, model_name: str, stream: bool = True) -> Iterator[dict]:
        """Pull a model from the Ollama registry.

        Args:
            model_name: Model to pull
            stream: Stream progress updates

        Yields:
            Progress updates
        """
        try:
            with self._client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name, "stream": stream},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            yield {"error": str(e)}

    def create_model(
        self,
        name: str,
        modelfile: str,
        stream: bool = True,
    ) -> Iterator[dict]:
        """Create a custom model from a Modelfile.

        Args:
            name: Name for the new model
            modelfile: Modelfile content
            stream: Stream progress updates

        Yields:
            Progress updates
        """
        try:
            with self._client.stream(
                "POST",
                "/api/create",
                json={"name": name, "modelfile": modelfile, "stream": stream},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            yield {"error": str(e)}

    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            response = self._client.delete(
                "/api/delete",
                json={"name": model_name},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[list] = None,
        options: Optional[dict] = None,
        stream: bool = False,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate a response from a model.

        Args:
            model: Model name
            prompt: User prompt
            system: System prompt
            context: Context from previous generation
            options: Generation options (temperature, top_p, etc.)
            stream: Stream the response

        Returns:
            GenerateResponse or iterator of responses if streaming
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options

        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_sync(payload)

    def _generate_sync(self, payload: dict) -> GenerateResponse:
        """Synchronous generation."""
        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            return GenerateResponse(
                response=data.get("response", ""),
                model=data.get("model", ""),
                done=data.get("done", True),
                context=data.get("context"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration"),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerateResponse(
                response="",
                model=payload.get("model", ""),
                done=True,
            )

    def _generate_stream(self, payload: dict) -> Iterator[GenerateResponse]:
        """Streaming generation."""
        try:
            with self._client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        yield GenerateResponse(
                            response=data.get("response", ""),
                            model=data.get("model", ""),
                            done=data.get("done", False),
                            context=data.get("context"),
                            total_duration=data.get("total_duration"),
                            eval_count=data.get("eval_count"),
                        )
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield GenerateResponse(response="", model="", done=True)

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: Optional[dict] = None,
        stream: bool = False,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Chat with a model using message history.

        Args:
            model: Model name
            messages: List of chat messages
            options: Generation options
            stream: Stream the response

        Returns:
            GenerateResponse or iterator
        """
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
        }

        if options:
            payload["options"] = options

        if stream:
            return self._chat_stream(payload)
        else:
            return self._chat_sync(payload)

    def _chat_sync(self, payload: dict) -> GenerateResponse:
        """Synchronous chat."""
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return GenerateResponse(
                response=data.get("message", {}).get("content", ""),
                model=data.get("model", ""),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count"),
            )
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return GenerateResponse(response="", model=payload.get("model", ""), done=True)

    def _chat_stream(self, payload: dict) -> Iterator[GenerateResponse]:
        """Streaming chat."""
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        yield GenerateResponse(
                            response=data.get("message", {}).get("content", ""),
                            model=data.get("model", ""),
                            done=data.get("done", False),
                            total_duration=data.get("total_duration"),
                            eval_count=data.get("eval_count"),
                        )
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield GenerateResponse(response="", model="", done=True)

    def embeddings(
        self,
        model: str,
        prompt: str,
    ) -> list[float]:
        """Generate embeddings for text.

        Args:
            model: Model name (e.g., "nomic-embed-text")
            prompt: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self._client.post(
                "/api/embeddings",
                json={"model": model, "prompt": prompt},
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Embeddings failed: {e}")
            return []

    def show_model(self, model_name: str) -> Optional[dict]:
        """Get detailed info about a model."""
        try:
            response = self._client.post(
                "/api/show",
                json={"name": model_name},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to show model {model_name}: {e}")
            return None

    def close(self):
        """Close the HTTP client."""
        self._client.close()


class OllamaTool(Tool):
    """Tool for interacting with Ollama."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "echo_assistant",
    ):
        """Initialize Ollama tool.

        Args:
            host: Ollama server URL
            default_model: Default model to use
        """
        super().__init__(
            name="ollama",
            description="Interact with Ollama for local model inference",
            requires_confirmation=False,
        )

        self.client = OllamaClient(host=host)
        self.default_model = default_model

    def execute(
        self,
        operation: str,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        system: Optional[str] = None,
        modelfile: Optional[str] = None,
        options: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute an Ollama operation.

        Operations:
            - health: Check if Ollama is running
            - list: List available models
            - pull: Pull a model
            - create: Create a custom model
            - delete: Delete a model
            - generate: Generate text
            - chat: Chat with history
            - embed: Generate embeddings
            - show: Show model details

        Args:
            operation: Operation to perform
            model: Model name
            prompt: Text prompt
            messages: Chat messages (for chat operation)
            system: System prompt
            modelfile: Modelfile content (for create)
            options: Generation options
        """
        import time
        start_time = time.time()

        model = model or self.default_model

        try:
            if operation == "health":
                healthy = self.client.is_healthy()
                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Ollama: {'healthy' if healthy else 'unreachable'}",
                    metadata={"healthy": healthy},
                )

            elif operation == "list":
                models = self.client.list_models()
                output_lines = [f"Available models ({len(models)}):"]
                for m in models:
                    size_gb = m.size / (1024**3)
                    output_lines.append(f"  • {m.name} ({size_gb:.1f}GB)")

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="\n".join(output_lines),
                    metadata={"models": [m.name for m in models]},
                )

            elif operation == "pull":
                if not model:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="model required for 'pull' operation",
                    )

                # Consume the stream to complete the pull
                for progress in self.client.pull_model(model):
                    if "error" in progress:
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            output="",
                            error=progress["error"],
                        )

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Model {model} pulled successfully",
                )

            elif operation == "create":
                if not model or not modelfile:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="model and modelfile required for 'create' operation",
                    )

                for progress in self.client.create_model(model, modelfile):
                    if "error" in progress:
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            output="",
                            error=progress["error"],
                        )

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Model {model} created successfully",
                )

            elif operation == "delete":
                if not model:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="model required for 'delete' operation",
                    )

                success = self.client.delete_model(model)
                result = ToolResult(
                    status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                    output=f"Model {model} deleted" if success else "",
                    error=None if success else f"Failed to delete {model}",
                )

            elif operation == "generate":
                if not prompt:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="prompt required for 'generate' operation",
                    )

                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    options=options,
                    stream=False,
                )

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=response.response,
                    metadata={
                        "model": response.model,
                        "eval_count": response.eval_count,
                        "total_duration_ms": response.total_duration // 1_000_000 if response.total_duration else None,
                    },
                )

            elif operation == "chat":
                if not messages:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="messages required for 'chat' operation",
                    )

                chat_messages = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in messages
                ]

                response = self.client.chat(
                    model=model,
                    messages=chat_messages,
                    options=options,
                    stream=False,
                )

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=response.response,
                    metadata={
                        "model": response.model,
                        "eval_count": response.eval_count,
                    },
                )

            elif operation == "embed":
                if not prompt:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="prompt required for 'embed' operation",
                    )

                embedding = self.client.embeddings(model=model, prompt=prompt)
                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Generated {len(embedding)}-dimensional embedding",
                    metadata={"embedding": embedding, "dimensions": len(embedding)},
                )

            elif operation == "show":
                if not model:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="model required for 'show' operation",
                    )

                info = self.client.show_model(model)
                if info:
                    result = ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Model: {model}\nParameters: {info.get('parameters', 'N/A')}\nTemplate: {info.get('template', 'N/A')[:100]}...",
                        metadata=info,
                    )
                else:
                    result = ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=f"Model {model} not found",
                    )

            else:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"Unknown operation: {operation}. Valid: health, list, pull, create, delete, generate, chat, embed, show",
                )

        except Exception as e:
            logger.exception(f"Ollama operation failed: {operation}")
            result = ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

        result.execution_time = time.time() - start_time
        self._record_result(result)
        return result


# Global singleton
_ollama_tool: Optional[OllamaTool] = None


def get_ollama(
    host: str = "http://localhost:11434",
    default_model: str = "echo_assistant",
) -> OllamaTool:
    """Get the global Ollama tool instance."""
    global _ollama_tool
    if _ollama_tool is None:
        _ollama_tool = OllamaTool(host=host, default_model=default_model)
    return _ollama_tool

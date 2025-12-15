"""
Conch Integrations

External service integrations for the consciousness engine:
- n8n: Workflow automation control
- Ollama: Local model serving
"""

from conch.integrations.n8n import N8NTool, N8NClient, get_n8n
from conch.integrations.ollama import OllamaTool, OllamaClient, get_ollama

__all__ = [
    "N8NTool",
    "N8NClient",
    "get_n8n",
    "OllamaTool",
    "OllamaClient",
    "get_ollama",
]

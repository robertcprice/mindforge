"""
Conch Web API

FastAPI-based REST API for interacting with Conch consciousness engine.
Provides endpoints for:
- Chat interactions
- Memory operations
- KVRM grounding
- System status
- Training management
"""

from conch.api.main import app, create_app

__all__ = ["app", "create_app"]

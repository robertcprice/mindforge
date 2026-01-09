#!/usr/bin/env python3
"""
PM Agent Orchestrator Dashboard - Backend

A real-time web dashboard for managing and visualizing the PM Agent Orchestrator system.
This dashboard provides a comprehensive interface for AI-powered project management.

Key Features:
- Kanban board for task lifecycle management (backlog → planning → in_progress → review → done)
- Real-time monitoring of 25 specialized execution agents across 7 categories:
  * Discovery & Requirements (requirements analyst, business analyst, user researcher)
  * Architecture & Design (solution architect, data architect, API designer, UX designer)
  * Backend Implementation (backend, database, integration, realtime engineers)
  * Frontend Implementation (frontend engineer, UI specialist, accessibility engineer)
  * Data & ML (data engineer, analytics engineer, ML engineer)
  * Quality Assurance (QA, test automation, performance, security engineers)
  * Operations & Infrastructure (DevOps, infrastructure, SRE engineers)
  * Support & Maintenance (technical writer, code reviewer, refactoring, maintenance)
- PM-1000 model integration via Ollama for intelligent task planning and agent dispatch
- Claude Code session spawning for autonomous task execution
- WebSocket-based real-time state broadcasting
- Project and task management with in-memory state storage
- System metrics monitoring (CPU, memory, uptime)
- Knowledge base integration for learning and pattern recognition

Architecture:
- Flask backend with Flask-SocketIO for real-time communication
- CORS-enabled REST API endpoints
- Integration with Ollama (PM-1000 model) and Claude Code CLI
- React frontend served from ../frontend directory
"""

import json
import asyncio
import os
import psutil
import signal
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import subprocess
import threading
import requests
import time

# Import robustness modules
from logging_config import init_default_logger, log, set_request_id, get_request_id
from config import get_config, server, ollama, session as session_config, pm as pm_config
from retry_utils import (
    retry_with_backoff, get_circuit, CircuitOpenError,
    OLLAMA_RETRY, OLLAMA_CIRCUIT, AGENT_MONITOR_RETRY, AGENT_MONITOR_CIRCUIT
)
from session_manager import get_session_manager, init_session_manager, SessionStatus
from state_store import get_state_store, init_state_store
from knowledge_base import get_knowledge_base, ObsidianKnowledgeBase

# Import Phase 2 autonomy components
try:
    from autonomy_api import register_autonomy_routes
    from coding_agents import get_agent_manager, init_agent_manager
    AUTONOMY_AVAILABLE = True
except ImportError as e:
    AUTONOMY_AVAILABLE = False
    print(f"Warning: Autonomy components not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize structured logging
config = get_config()
logger = init_default_logger(
    log_level=config.logging.level,
    log_dir=config.logging.log_dir,
    json_logs=config.logging.json_format,
)

# Track server start time for uptime calculation
SERVER_START_TIME = datetime.now()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.config['SECRET_KEY'] = config.server.secret_key
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Rate limiting
_limiter_available = False
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    if config.rate_limit.enabled:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[config.rate_limit.default_limit],
            storage_uri=config.rate_limit.storage_uri,
            strategy="fixed-window",
        )
        _limiter_available = True
        logger.info(f"Rate limiting enabled: {config.rate_limit.default_limit}")
    else:
        limiter = None
        logger.info("Rate limiting disabled by configuration")
except ImportError:
    limiter = None
    logger.warning("flask-limiter not installed, rate limiting disabled")


def rate_limit(limit_string: str):
    """Decorator to apply rate limiting if available."""
    def decorator(f):
        if _limiter_available and limiter:
            return limiter.limit(limit_string)(f)
        return f
    return decorator

# PM-1000 model configuration (from config module)
PM_MODEL = config.ollama.model_name
OLLAMA_BASE = config.ollama.base_url
AGENT_MONITOR_BASE = config.agent_monitor.base_url

# Initialize session manager, state store, and autonomous worker
session_manager = None
state_store = None
autonomous_worker = None
task_queue = None

def init_services():
    """Initialize background services."""
    global session_manager, state_store, autonomous_worker, task_queue

    # Initialize state persistence
    if config.state.persistence_enabled:
        state_store = init_state_store()
        logger.info("State persistence initialized")

    # Initialize session manager with callback
    def on_session_complete(session):
        """Handle completed sessions."""
        logger.info(f"Session {session.task_id} completed with status {session.status}")
        # Save to state store
        if state_store:
            state_store.save_session(session.to_dict())
        # Broadcast update
        socketio.emit('cc_session_update', session.to_dict())

    session_manager = init_session_manager(on_session_complete=on_session_complete)
    logger.info("Session manager initialized")

    # Initialize task queue and autonomous worker
    try:
        from task_queue import init_task_queue
        from autonomous_worker import init_autonomous_worker

        task_queue = init_task_queue()
        task_queue.start()
        logger.info("Task queue initialized")

        # Initialize autonomous worker (disabled by default, enable via API)
        autonomous_worker = init_autonomous_worker(
            task_queue=task_queue,
            session_manager=session_manager,
            max_concurrent=3,
        )
        # Don't auto-start - let user enable via API
        logger.info("Autonomous worker initialized (not started)")
    except Exception as e:
        logger.warning(f"Failed to initialize autonomous components: {e}")

    # Initialize Phase 2 autonomy components
    if AUTONOMY_AVAILABLE:
        try:
            # Initialize coding agent manager
            agent_manager = init_agent_manager()
            logger.info("Coding agent manager initialized")

            # Register autonomy API routes
            register_autonomy_routes(app)
            logger.info("Autonomy API routes registered")
        except Exception as e:
            logger.warning(f"Failed to initialize Phase 2 autonomy: {e}")

# Active Claude Code sessions tracked by this PM instance (legacy, kept for compatibility)
claude_code_sessions: Dict[str, dict] = {}  # task_id -> session info

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"


# The 25 specialized execution agents
EXECUTION_AGENTS = {
    # Discovery & Requirements (3)
    "requirements_analyst": {"role": "Requirements Analyst", "category": "discovery", "color": "#8B5CF6"},
    "business_analyst": {"role": "Business Analyst", "category": "discovery", "color": "#8B5CF6"},
    "user_researcher": {"role": "User Researcher", "category": "discovery", "color": "#8B5CF6"},
    # Architecture & Design (4)
    "solution_architect": {"role": "Solution Architect", "category": "architecture", "color": "#3B82F6"},
    "data_architect": {"role": "Data Architect", "category": "architecture", "color": "#3B82F6"},
    "api_designer": {"role": "API Designer", "category": "architecture", "color": "#3B82F6"},
    "ux_designer": {"role": "UX Designer", "category": "architecture", "color": "#3B82F6"},
    # Backend Implementation (4)
    "backend_engineer": {"role": "Backend Engineer", "category": "backend", "color": "#10B981"},
    "database_engineer": {"role": "Database Engineer", "category": "backend", "color": "#10B981"},
    "integration_engineer": {"role": "Integration Engineer", "category": "backend", "color": "#10B981"},
    "realtime_engineer": {"role": "Realtime Engineer", "category": "backend", "color": "#10B981"},
    # Frontend Implementation (3)
    "frontend_engineer": {"role": "Frontend Engineer", "category": "frontend", "color": "#F59E0B"},
    "ui_specialist": {"role": "UI Specialist", "category": "frontend", "color": "#F59E0B"},
    "accessibility_engineer": {"role": "Accessibility Engineer", "category": "frontend", "color": "#F59E0B"},
    # Data & ML (3)
    "data_engineer": {"role": "Data Engineer", "category": "data", "color": "#EC4899"},
    "analytics_engineer": {"role": "Analytics Engineer", "category": "data", "color": "#EC4899"},
    "ml_engineer": {"role": "ML Engineer", "category": "data", "color": "#EC4899"},
    # Quality Assurance (4)
    "qa_engineer": {"role": "QA Engineer", "category": "quality", "color": "#EF4444"},
    "test_automation_engineer": {"role": "Test Automation Engineer", "category": "quality", "color": "#EF4444"},
    "performance_engineer": {"role": "Performance Engineer", "category": "quality", "color": "#EF4444"},
    "security_engineer": {"role": "Security Engineer", "category": "quality", "color": "#EF4444"},
    # Operations & Infrastructure (3)
    "devops_engineer": {"role": "DevOps Engineer", "category": "operations", "color": "#6366F1"},
    "infrastructure_engineer": {"role": "Infrastructure Engineer", "category": "operations", "color": "#6366F1"},
    "sre_engineer": {"role": "Site Reliability Engineer", "category": "operations", "color": "#6366F1"},
    # Support & Maintenance (4)
    "technical_writer": {"role": "Technical Writer", "category": "support", "color": "#78716C"},
    "code_reviewer": {"role": "Code Reviewer", "category": "support", "color": "#78716C"},
    "refactoring_engineer": {"role": "Refactoring Engineer", "category": "support", "color": "#78716C"},
    "maintenance_engineer": {"role": "Maintenance Engineer", "category": "support", "color": "#78716C"},
}

# Planning capabilities
PLANNING_CAPABILITIES = [
    {"id": "strategic_analysis", "name": "Strategic Analysis", "icon": "strategy"},
    {"id": "knowledge_synthesis", "name": "Knowledge Synthesis", "icon": "research"},
    {"id": "technical_specification", "name": "Technical Specification", "icon": "code"},
    {"id": "security_review", "name": "Security Review", "icon": "shield"},
    {"id": "creative_alternatives", "name": "Creative Alternatives", "icon": "lightbulb"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# In-Memory State (replace with database in production)
# ═══════════════════════════════════════════════════════════════════════════════

# Projects
projects: Dict[str, dict] = {}

# Tasks
tasks: Dict[str, dict] = {}

# Agent states
agent_states: Dict[str, dict] = {
    agent_id: {
        "id": agent_id,
        "status": AgentStatus.IDLE.value,
        "current_task": None,
        "task_progress": 0,
        "last_activity": None,
        **info
    }
    for agent_id, info in EXECUTION_AGENTS.items()
}

# Activity log
activity_log: List[dict] = []

# PM Orchestrator state
pm_state = {
    "status": "idle",
    "current_thinking": None,
    "current_plan": None,
    "dispatched_tasks": [],
    "current_goal": None,
    "active_cc_sessions": [],  # Claude Code sessions spawned by PM
}


# ═══════════════════════════════════════════════════════════════════════════════
# PM-1000 Model Integration
# ═══════════════════════════════════════════════════════════════════════════════

@retry_with_backoff(config=OLLAMA_RETRY, circuit="ollama")
def _call_ollama_api(prompt: str, stream: bool = False) -> requests.Response:
    """Internal function to call Ollama API with retry."""
    return requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": PM_MODEL,
            "prompt": prompt,
            "stream": stream,
        },
        timeout=config.ollama.timeout,
    )


def call_pm1000(prompt: str, stream: bool = False) -> str:
    """
    Call PM-1000 model via Ollama with retry and circuit breaker.

    Features:
    - Exponential backoff on failure
    - Circuit breaker to prevent cascading failures
    - Structured logging
    """
    try:
        response = _call_ollama_api(prompt, stream)
        if response.status_code == 200:
            result = response.json().get("response", "")
            logger.debug(f"PM-1000 response: {result[:100]}...")
            return result
        logger.warning(f"PM-1000 returned status {response.status_code}")
        return f"Error: {response.status_code}"
    except CircuitOpenError as e:
        logger.error(f"PM-1000 circuit open: {e}")
        return "Error: PM-1000 service temporarily unavailable (circuit open)"
    except Exception as e:
        logger.error(f"PM-1000 call failed: {e}")
        return f"Error calling PM-1000: {str(e)}"


def spawn_claude_code_session(task_id: str, task_title: str, working_dir: str, prompt: str) -> Optional[dict]:
    """Spawn a Claude Code session for a task."""
    try:
        # Build the claude command
        cmd = [
            "claude",
            "--print",  # Non-interactive mode
            "--permission-mode", "acceptEdits",  # Auto-accept file edits
            "--max-turns", "10",  # Limit turns to prevent runaway
            prompt,
        ]

        # Start the process
        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        session_info = {
            "task_id": task_id,
            "task_title": task_title,
            "working_dir": working_dir,
            "prompt": prompt,
            "pid": process.pid,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "output": "",
        }

        claude_code_sessions[task_id] = session_info

        # Start a thread to monitor the process
        def monitor_process():
            stdout, stderr = process.communicate()
            session_info["output"] = stdout or stderr
            session_info["status"] = "completed" if process.returncode == 0 else "failed"
            session_info["return_code"] = process.returncode
            session_info["completed_at"] = datetime.now().isoformat()

            # Update task status
            if task_id in tasks:
                tasks[task_id]["status"] = TaskStatus.REVIEW.value if process.returncode == 0 else TaskStatus.IN_PROGRESS.value

            # Broadcast update
            socketio.emit('cc_session_update', session_info)
            broadcast_state()

        thread = threading.Thread(target=monitor_process, daemon=True)
        thread.start()

        return session_info

    except Exception as e:
        return {"error": str(e)}


def process_goal_with_pm1000(goal: str, project_dir: str = "."):
    """Process a goal using PM-1000 model in a background thread."""
    global pm_state

    try:
        # Phase 1: Analyze the goal
        pm_state["status"] = "thinking"
        pm_state["current_thinking"] = f"Analyzing goal: {goal}"
        pm_state["current_goal"] = goal
        broadcast_state()

        analysis_prompt = f"""You are PM-1000, an AI project manager. Analyze this goal and determine the best approach.

Goal: {goal}
Working Directory: {project_dir}

Respond in this format:
COMPLEXITY: [simple|moderate|complex]
APPROACH: [direct|decompose|escalate]
REASONING: <brief explanation>
TASKS: <list of specific tasks to execute, one per line>"""

        analysis = call_pm1000(analysis_prompt)
        pm_state["current_thinking"] = analysis
        broadcast_state()
        log_activity("pm_analysis_complete", {"analysis": analysis[:200]})

        # Phase 2: Generate plan
        pm_state["status"] = "planning"
        pm_state["current_plan"] = analysis
        broadcast_state()

        # Parse tasks from analysis
        task_lines = []
        in_tasks = False
        for line in analysis.split('\n'):
            if line.strip().startswith('TASKS:'):
                in_tasks = True
                continue
            if in_tasks and line.strip() and line.strip().startswith('-'):
                task_lines.append(line.strip()[1:].strip())

        # If no tasks found, create a single task
        if not task_lines:
            task_lines = [goal]

        # Phase 3: Dispatch tasks
        pm_state["status"] = "dispatching"
        pm_state["dispatched_tasks"] = []
        broadcast_state()

        for task_desc in task_lines[:5]:  # Limit to 5 tasks
            task_id = str(uuid.uuid4())[:8]
            task = {
                "id": task_id,
                "project_id": None,
                "title": task_desc[:50],
                "description": task_desc,
                "status": TaskStatus.IN_PROGRESS.value,
                "assigned_agent": "claude_code",
                "priority": "high",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "phase": None,
                "subtasks": [],
            }
            tasks[task_id] = task
            pm_state["dispatched_tasks"].append({
                "task_id": task_id,
                "title": task_desc[:50],
                "status": "dispatching",
            })
            broadcast_state()

            # Spawn Claude Code session for this task
            session = spawn_claude_code_session(
                task_id=task_id,
                task_title=task_desc[:50],
                working_dir=project_dir,
                prompt=task_desc,
            )

            if session and "error" not in session:
                pm_state["active_cc_sessions"].append(session)
                log_activity("cc_session_spawned", {
                    "task_id": task_id,
                    "pid": session.get("pid"),
                })

            time.sleep(0.5)  # Small delay between spawns

        # Complete
        pm_state["status"] = "monitoring"
        broadcast_state()
        log_activity("pm_dispatch_complete", {"task_count": len(task_lines)})

        # Log to knowledge base
        try:
            kb = get_knowledge_base()
            kb.log_agent_activity(
                agent_name="PM-1000",
                task=goal[:100],
                result=f"Dispatched {len(task_lines)} tasks to Claude Code agents",
                status="dispatched",
            )
        except Exception:
            pass  # KB logging is non-critical

    except Exception as e:
        pm_state["status"] = "error"
        pm_state["current_thinking"] = f"Error: {str(e)}"
        broadcast_state()
        log_activity("pm_error", {"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def log_activity(action: str, details: dict):
    """Log an activity and broadcast to clients."""
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details,
    }
    activity_log.append(entry)
    if len(activity_log) > 100:
        activity_log.pop(0)
    socketio.emit('activity', entry)
    return entry


def broadcast_state():
    """Broadcast full state to all clients."""
    socketio.emit('state_update', {
        "projects": list(projects.values()),
        "tasks": list(tasks.values()),
        "agents": list(agent_states.values()),
        "pm_state": pm_state,
        "cc_sessions": list(claude_code_sessions.values()),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Static Files
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the dashboard."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(app.static_folder, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/state')
def get_state():
    """Get full dashboard state."""
    return jsonify({
        "projects": list(projects.values()),
        "tasks": list(tasks.values()),
        "agents": list(agent_states.values()),
        "pm_state": pm_state,
        "activity": activity_log[-20:],
        "cc_sessions": list(claude_code_sessions.values()),
    })


@app.route('/api/ping')
def ping():
    """Simple ping endpoint for connectivity checks."""
    return jsonify({"pong": True})


@app.route('/api/version')
def get_version():
    """Get the application version."""
    return jsonify({
        "version": "2.0.0",
        "name": "PM Agent Orchestrator Dashboard",
        "features": {
            "autonomy_available": AUTONOMY_AVAILABLE,
            "phase": "Phase 2 - Intelligence Generation" if AUTONOMY_AVAILABLE else "Phase 1",
            "components": [
                "Decision Engine",
                "Temporal Intelligence",
                "Opportunity Scout",
                "Anti-Goal Generator",
                "Configurable Coding Agents"
            ] if AUTONOMY_AVAILABLE else []
        }
    })


def _check_ollama_health() -> dict:
    """Check Ollama service health."""
    try:
        start = time.time()
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        latency = int((time.time() - start) * 1000)
        if response.status_code == 200:
            models = response.json().get("models", [])
            has_pm1000 = any(m.get("name", "").startswith(PM_MODEL) for m in models)
            return {
                "status": "up",
                "latency_ms": latency,
                "model_available": has_pm1000,
            }
        return {"status": "degraded", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "down", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"status": "down", "error": "Timeout"}
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _check_agent_monitor_health() -> dict:
    """Check agent monitor service health."""
    if not config.agent_monitor.enabled:
        return {"status": "disabled"}
    try:
        start = time.time()
        response = requests.get(f"{AGENT_MONITOR_BASE}/health", timeout=3)
        latency = int((time.time() - start) * 1000)
        if response.status_code == 200:
            return {"status": "up", "latency_ms": latency}
        return {"status": "degraded", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "down", "error": "Connection refused"}
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _check_knowledge_base_health() -> dict:
    """Check knowledge base health."""
    try:
        kb = get_knowledge_base()
        return {
            "status": "up",
            "note_count": len(kb._index),
            "vault_path": str(kb.vault_path),
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _check_state_store_health() -> dict:
    """Check state store health."""
    if not config.state.persistence_enabled or state_store is None:
        return {"status": "disabled"}
    try:
        stats = state_store.get_stats()
        return {
            "status": "up",
            "db_size_bytes": stats.get("db_size_bytes", 0),
            "tasks_count": stats.get("tasks_count", 0),
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


@app.route('/api/health')
def health_check():
    """
    Comprehensive health check with dependency status.

    Returns overall health status based on critical dependencies:
    - healthy: All critical services up
    - degraded: Some non-critical services down
    - unhealthy: Critical services down
    """
    now = datetime.now()
    uptime_delta = now - SERVER_START_TIME
    uptime_seconds = int(uptime_delta.total_seconds())

    # Check all dependencies
    checks = {
        "ollama": _check_ollama_health(),
        "agent_monitor": _check_agent_monitor_health(),
        "knowledge_base": _check_knowledge_base_health(),
        "state_store": _check_state_store_health(),
    }

    # Get circuit breaker states
    ollama_circuit = get_circuit("ollama")
    circuits = {
        "ollama": ollama_circuit.get_stats(),
    }

    # Determine overall status
    critical_down = checks["ollama"]["status"] == "down"
    any_down = any(c["status"] == "down" for c in checks.values())

    if critical_down:
        overall_status = "unhealthy"
    elif any_down:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Get memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    # Get session manager stats
    sm_stats = session_manager.get_stats() if session_manager else {}

    return jsonify({
        "status": overall_status,
        "timestamp": now.isoformat(),
        "version": "1.1.0",
        "uptime": {
            "seconds": uptime_seconds,
            "human": str(uptime_delta).split('.')[0],
        },
        "checks": checks,
        "circuits": circuits,
        "memory": {
            "process": {
                "rss_bytes": memory_info.rss,
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "vms_bytes": memory_info.vms,
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            },
            "system": {
                "total_bytes": system_memory.total,
                "total_gb": round(system_memory.total / (1024 ** 3), 2),
                "available_bytes": system_memory.available,
                "available_gb": round(system_memory.available / (1024 ** 3), 2),
                "percent_used": system_memory.percent,
            },
        },
        "stats": {
            "projects": len(projects),
            "tasks": len(tasks),
            "active_agents": len([a for a in agent_states.values() if a.get("status") == "working"]),
            "cc_sessions": len(claude_code_sessions),
        },
        "sessions": sm_stats,
    })


@app.route('/api/agents')
def get_agents():
    """Get all agent states."""
    return jsonify(list(agent_states.values()))


@app.route('/api/agents/<agent_id>')
def get_agent(agent_id: str):
    """Get specific agent state."""
    if agent_id not in agent_states:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify(agent_states[agent_id])


@app.route('/api/capabilities')
def get_capabilities():
    """Get planning capabilities."""
    return jsonify(PLANNING_CAPABILITIES)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Projects
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get all projects."""
    return jsonify(list(projects.values()))


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    data = request.get_json()
    project_id = str(uuid.uuid4())[:8]

    # Get directory path, default to current directory
    directory = data.get("directory", ".")
    if not directory:
        directory = "."

    project = {
        "id": project_id,
        "name": data.get("name", "Untitled Project"),
        "directory": directory,
        "description": data.get("description", ""),
        "created_at": datetime.now().isoformat(),
        "status": "active",
    }
    projects[project_id] = project

    log_activity("project_created", {"project": project})
    broadcast_state()

    return jsonify(project), 201


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """Delete a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404

    del projects[project_id]
    # Also delete associated tasks
    task_ids_to_delete = [tid for tid, t in tasks.items() if t.get("project_id") == project_id]
    for tid in task_ids_to_delete:
        del tasks[tid]

    log_activity("project_deleted", {"project_id": project_id})
    broadcast_state()

    return jsonify({"status": "deleted"})


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Tasks
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks, optionally filtered by project."""
    project_id = request.args.get('project_id')
    if project_id:
        filtered = [t for t in tasks.values() if t.get("project_id") == project_id]
        return jsonify(filtered)
    return jsonify(list(tasks.values()))


@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task."""
    data = request.get_json()
    task_id = str(uuid.uuid4())[:8]

    task = {
        "id": task_id,
        "project_id": data.get("project_id"),
        "title": data.get("title", "Untitled Task"),
        "description": data.get("description", ""),
        "status": TaskStatus.BACKLOG.value,
        "assigned_agent": None,
        "priority": data.get("priority", "medium"),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "phase": None,
        "subtasks": [],
    }
    tasks[task_id] = task

    log_activity("task_created", {"task": task})
    broadcast_state()

    return jsonify(task), 201


@app.route('/api/tasks/<task_id>', methods=['PATCH'])
def update_task(task_id: str):
    """Update a task."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.get_json()
    task = tasks[task_id]

    # Update allowed fields
    for field in ["title", "description", "status", "priority", "assigned_agent", "phase"]:
        if field in data:
            task[field] = data[field]

    task["updated_at"] = datetime.now().isoformat()

    # If assigning to agent, update agent state
    if "assigned_agent" in data and data["assigned_agent"]:
        agent_id = data["assigned_agent"]
        if agent_id in agent_states:
            agent_states[agent_id]["status"] = AgentStatus.WORKING.value
            agent_states[agent_id]["current_task"] = task_id
            agent_states[agent_id]["last_activity"] = datetime.now().isoformat()

    log_activity("task_updated", {"task_id": task_id, "changes": data})
    broadcast_state()

    return jsonify(task)


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id: str):
    """Delete a task."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404

    del tasks[task_id]

    log_activity("task_deleted", {"task_id": task_id})
    broadcast_state()

    return jsonify({"status": "deleted"})


@app.route('/api/tasks/<task_id>/move', methods=['POST'])
def move_task(task_id: str):
    """Move a task to a different status column."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.get_json()
    new_status = data.get("status")

    if new_status not in [s.value for s in TaskStatus]:
        return jsonify({"error": "Invalid status"}), 400

    tasks[task_id]["status"] = new_status
    tasks[task_id]["updated_at"] = datetime.now().isoformat()

    log_activity("task_moved", {"task_id": task_id, "new_status": new_status})
    broadcast_state()

    return jsonify(tasks[task_id])


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - PM Orchestrator Actions
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/pm/submit', methods=['POST'])
@rate_limit("5/minute")
def submit_to_pm():
    """Submit a goal/task to the PM Orchestrator for planning. Rate limited: 5/minute."""
    data = request.get_json()
    goal = data.get("goal", "")
    project_dir = data.get("directory", ".")

    if not goal:
        return jsonify({"error": "Goal required"}), 400

    # Get project directory if project is selected
    project_id = data.get("project_id")
    if project_id and project_id in projects:
        project_dir = projects[project_id].get("directory", ".")

    log_activity("pm_goal_submitted", {"goal": goal, "directory": project_dir})
    broadcast_state()

    # Start PM-1000 processing in background thread
    thread = threading.Thread(
        target=process_goal_with_pm1000,
        args=(goal, project_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "submitted",
        "message": "Goal submitted to PM-1000 for analysis",
    })


@app.route('/api/pm/state')
def get_pm_state():
    """Get current PM Orchestrator state."""
    return jsonify(pm_state)


@app.route('/api/pm/reset', methods=['POST'])
def reset_pm():
    """Reset PM Orchestrator to idle state."""
    global pm_state
    pm_state = {
        "status": "idle",
        "current_thinking": None,
        "current_plan": None,
        "dispatched_tasks": [],
        "current_goal": None,
        "active_cc_sessions": [],
    }
    log_activity("pm_reset", {})
    broadcast_state()
    return jsonify({"status": "reset"})


@app.route('/api/pm/sessions')
def get_pm_sessions():
    """Get Claude Code sessions spawned by this PM instance."""
    return jsonify({
        "sessions": list(claude_code_sessions.values()),
        "active": [s for s in claude_code_sessions.values() if s.get("status") == "running"],
        "completed": [s for s in claude_code_sessions.values() if s.get("status") in ["completed", "failed"]],
    })


@app.route('/api/pm/sessions/<task_id>')
def get_pm_session(task_id: str):
    """Get a specific Claude Code session."""
    if task_id not in claude_code_sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(claude_code_sessions[task_id])


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Agent Actions
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/agents/<agent_id>/dispatch', methods=['POST'])
def dispatch_to_agent(agent_id: str):
    """Dispatch a task to a specific agent."""
    if agent_id not in agent_states:
        return jsonify({"error": "Agent not found"}), 404

    data = request.get_json()
    task_id = data.get("task_id")

    if not task_id or task_id not in tasks:
        return jsonify({"error": "Valid task_id required"}), 400

    # Update agent state
    agent_states[agent_id]["status"] = AgentStatus.WORKING.value
    agent_states[agent_id]["current_task"] = task_id
    agent_states[agent_id]["task_progress"] = 0
    agent_states[agent_id]["last_activity"] = datetime.now().isoformat()

    # Update task
    tasks[task_id]["assigned_agent"] = agent_id
    tasks[task_id]["status"] = TaskStatus.IN_PROGRESS.value

    log_activity("agent_dispatched", {
        "agent_id": agent_id,
        "task_id": task_id,
        "agent_role": agent_states[agent_id]["role"],
    })
    broadcast_state()

    return jsonify({
        "status": "dispatched",
        "agent": agent_states[agent_id],
        "task": tasks[task_id],
    })


@app.route('/api/agents/<agent_id>/complete', methods=['POST'])
def complete_agent_task(agent_id: str):
    """Mark an agent's current task as complete."""
    if agent_id not in agent_states:
        return jsonify({"error": "Agent not found"}), 404

    agent = agent_states[agent_id]
    task_id = agent["current_task"]

    if not task_id:
        return jsonify({"error": "Agent has no current task"}), 400

    # Update agent state
    agent["status"] = AgentStatus.IDLE.value
    agent["current_task"] = None
    agent["task_progress"] = 0
    agent["last_activity"] = datetime.now().isoformat()

    # Update task
    if task_id in tasks:
        tasks[task_id]["status"] = TaskStatus.REVIEW.value
        tasks[task_id]["assigned_agent"] = None

    log_activity("agent_completed", {
        "agent_id": agent_id,
        "task_id": task_id,
    })
    broadcast_state()

    return jsonify({"status": "completed"})


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Activity Log
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/activity')
def get_activity():
    """Get recent activity log."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(activity_log[-limit:])


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Events
# ═══════════════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'connected'})
    emit('state_update', {
        "projects": list(projects.values()),
        "tasks": list(tasks.values()),
        "agents": list(agent_states.values()),
        "pm_state": pm_state,
        "cc_sessions": list(claude_code_sessions.values()),
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    pass


@socketio.on('request_state')
def handle_request_state():
    """Handle state request from client."""
    emit('state_update', {
        "projects": list(projects.values()),
        "tasks": list(tasks.values()),
        "agents": list(agent_states.values()),
        "pm_state": pm_state,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Demo Data
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/demo/setup', methods=['POST'])
def setup_demo():
    """Set up demo data for testing the dashboard."""
    global projects, tasks

    # Create demo project
    project_id = "demo-001"
    projects[project_id] = {
        "id": project_id,
        "name": "Authentication System",
        "description": "Build a complete authentication system with OAuth, MFA, and session management",
        "created_at": datetime.now().isoformat(),
        "status": "active",
    }

    # Create demo tasks
    demo_tasks = [
        {"title": "Design auth architecture", "status": "done", "agent": "solution_architect"},
        {"title": "Create database schema", "status": "done", "agent": "data_architect"},
        {"title": "Implement JWT token service", "status": "in_progress", "agent": "backend_engineer"},
        {"title": "Build login UI components", "status": "in_progress", "agent": "frontend_engineer"},
        {"title": "Add OAuth providers", "status": "planning", "agent": None},
        {"title": "Implement MFA flow", "status": "backlog", "agent": None},
        {"title": "Write security tests", "status": "backlog", "agent": None},
        {"title": "Performance testing", "status": "backlog", "agent": None},
    ]

    for i, t in enumerate(demo_tasks):
        task_id = f"task-{i+1:03d}"
        tasks[task_id] = {
            "id": task_id,
            "project_id": project_id,
            "title": t["title"],
            "description": "",
            "status": t["status"],
            "assigned_agent": t["agent"],
            "priority": "high" if i < 3 else "medium",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "phase": None,
            "subtasks": [],
        }

        # Update agent states for assigned tasks
        if t["agent"] and t["status"] == "in_progress":
            agent_states[t["agent"]]["status"] = AgentStatus.WORKING.value
            agent_states[t["agent"]]["current_task"] = task_id
            agent_states[t["agent"]]["task_progress"] = 45

    log_activity("demo_setup", {"project_id": project_id, "task_count": len(demo_tasks)})
    broadcast_state()

    return jsonify({"status": "demo data created"})


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Claude Code Session Monitoring (via agent-monitor)
# ═══════════════════════════════════════════════════════════════════════════════

# Agent monitor client (lazy loaded)
_agent_monitor_client = None


def get_agent_monitor_client():
    """Get or create agent monitor client."""
    global _agent_monitor_client
    if _agent_monitor_client is None:
        import aiohttp
        _agent_monitor_client = {
            "base_url": "http://127.0.0.1:8420",
            "available": False,
        }
    return _agent_monitor_client


@app.route('/api/cc/sessions')
def get_cc_sessions():
    """Get active Claude Code sessions from agent-monitor."""
    import requests

    try:
        client = get_agent_monitor_client()
        response = requests.get(
            f"{client['base_url']}/api/sessions",
            params={"limit": 50, "active_only": "false"},
            timeout=5,
        )

        if response.status_code != 200:
            return jsonify({"error": "Agent monitor not responding", "sessions": []}), 200

        data = response.json()
        sessions = data.get("sessions", [])

        # Add PM task correlation
        for session in sessions:
            session["pm_task_id"] = None  # TODO: Correlate with PM tasks

        return jsonify({
            "sessions": sessions,
            "total": len(sessions),
            "monitor_available": True,
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Agent monitor not running",
            "sessions": [],
            "monitor_available": False,
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "sessions": [],
            "monitor_available": False,
        }), 200


@app.route('/api/cc/sessions/<session_id>')
def get_cc_session(session_id: str):
    """Get a specific Claude Code session."""
    import requests

    try:
        client = get_agent_monitor_client()
        response = requests.get(
            f"{client['base_url']}/api/sessions/{session_id}",
            timeout=5,
        )

        if response.status_code != 200:
            return jsonify({"error": "Session not found"}), 404

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cc/sessions/<session_id>/events')
def get_cc_session_events(session_id: str):
    """Get events for a Claude Code session."""
    import requests

    try:
        client = get_agent_monitor_client()
        limit = request.args.get('limit', 100, type=int)

        response = requests.get(
            f"{client['base_url']}/api/sessions/{session_id}/events",
            params={"limit": limit},
            timeout=5,
        )

        if response.status_code != 200:
            return jsonify({"error": "Session not found", "events": []}), 404

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e), "events": []}), 500


@app.route('/api/cc/metrics')
def get_cc_metrics():
    """Get Claude Code usage metrics."""
    import requests

    try:
        client = get_agent_monitor_client()

        # Get sessions to compute metrics
        response = requests.get(
            f"{client['base_url']}/api/sessions",
            params={"limit": 100},
            timeout=5,
        )

        if response.status_code != 200:
            return jsonify({"error": "Agent monitor not responding"}), 200

        data = response.json()
        sessions = data.get("sessions", [])

        # Compute metrics
        active_sessions = [s for s in sessions if s.get("status") == "active"]
        total_tokens = sum(
            (s.get("tokens_input", 0) or 0) + (s.get("tokens_output", 0) or 0)
            for s in sessions
        )
        total_cost = sum(s.get("estimated_cost", 0) or 0 for s in sessions)

        return jsonify({
            "total_sessions": len(sessions),
            "active_sessions": len(active_sessions),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "monitor_available": True,
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "total_sessions": 0,
            "active_sessions": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "monitor_available": False,
        }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Knowledge Base (Obsidian Integration)
# ═══════════════════════════════════════════════════════════════════════════════

from knowledge_base import get_knowledge_base, ObsidianKnowledgeBase

# Knowledge base instance (lazy loaded)
_kb: Optional[ObsidianKnowledgeBase] = None


def get_kb() -> ObsidianKnowledgeBase:
    """Get or create knowledge base instance."""
    global _kb
    if _kb is None:
        # Default vault path - can be configured
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
        _kb = get_knowledge_base(vault_path)
    return _kb


@app.route('/api/kb/info')
def get_kb_info():
    """Get knowledge base information."""
    kb = get_kb()
    return jsonify(kb.to_dict())


@app.route('/api/kb/notes')
def get_kb_notes():
    """Get all notes, optionally filtered."""
    kb = get_kb()
    folder = request.args.get('folder')
    notes = kb.get_all_notes(folder)
    return jsonify({
        "notes": [
            {
                "id": n.id,
                "title": n.title,
                "tags": n.tags,
                "path": n.path,
                "updated_at": n.updated_at,
            }
            for n in notes
        ],
        "total": len(notes),
    })


@app.route('/api/kb/notes/<note_id>')
def get_kb_note(note_id: str):
    """Get a specific note."""
    kb = get_kb()
    note = kb.get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404
    return jsonify({
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "tags": note.tags,
        "links": note.links,
        "path": note.path,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
        "metadata": note.metadata,
    })


@app.route('/api/kb/notes', methods=['POST'])
def create_kb_note():
    """Create a new note."""
    kb = get_kb()
    data = request.get_json()

    title = data.get("title", "Untitled")
    content = data.get("content", "")
    folder = data.get("folder", "")
    tags = data.get("tags", [])
    metadata = data.get("metadata", {})

    note = kb.create_note(
        title=title,
        content=content,
        folder=folder,
        tags=tags,
        metadata=metadata,
    )

    log_activity("kb_note_created", {"title": title, "folder": folder})

    return jsonify({
        "id": note.id,
        "title": note.title,
        "path": note.path,
    }), 201


@app.route('/api/kb/search')
def search_kb():
    """Search the knowledge base."""
    kb = get_kb()
    query = request.args.get('q', '')
    tags = request.args.getlist('tags')
    folder = request.args.get('folder')
    limit = request.args.get('limit', 20, type=int)

    notes = kb.search(query, tags=tags or None, folder=folder, limit=limit)

    return jsonify({
        "query": query,
        "results": [
            {
                "id": n.id,
                "title": n.title,
                "tags": n.tags,
                "path": n.path,
                "snippet": n.content[:200] + "..." if len(n.content) > 200 else n.content,
            }
            for n in notes
        ],
        "total": len(notes),
    })


@app.route('/api/kb/tags')
def get_kb_tags():
    """Get all tags with counts."""
    kb = get_kb()
    return jsonify(kb.get_tags())


@app.route('/api/kb/daily')
def get_daily_note():
    """Get or create today's daily note."""
    kb = get_kb()
    note = kb.get_daily_note()
    return jsonify({
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "path": note.path,
    })


@app.route('/api/kb/log-activity', methods=['POST'])
def log_to_kb():
    """Log agent activity to knowledge base."""
    kb = get_kb()
    data = request.get_json()

    agent_name = data.get("agent_name", "Unknown Agent")
    task = data.get("task", "")
    result = data.get("result", "")
    status = data.get("status", "completed")

    kb.log_agent_activity(agent_name, task, result, status)

    return jsonify({"status": "logged"})


@app.route('/api/kb/config', methods=['POST'])
def configure_kb():
    """Configure knowledge base vault path."""
    global _kb
    data = request.get_json()
    vault_path = data.get("vault_path")

    if vault_path:
        os.environ["OBSIDIAN_VAULT_PATH"] = vault_path
        _kb = get_knowledge_base(vault_path)
        log_activity("kb_configured", {"vault_path": vault_path})
        return jsonify({"status": "configured", "vault_path": vault_path})

    return jsonify({"error": "vault_path required"}), 400


# ═══════════════════════════════════════════════════════════════════════════════
# Routes - Autonomous Worker & Task Queue
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/autonomous/status')
def get_autonomous_status():
    """Get autonomous worker status."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503
    return jsonify(autonomous_worker.get_status())


@app.route('/api/autonomous/start', methods=['POST'])
def start_autonomous_worker():
    """Start the autonomous worker for continuous task processing."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503

    autonomous_worker.start()
    log_activity("autonomous_started", {})
    return jsonify({"status": "started", "message": "Autonomous worker is now processing tasks"})


@app.route('/api/autonomous/stop', methods=['POST'])
def stop_autonomous_worker():
    """Stop the autonomous worker."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503

    autonomous_worker.stop()
    log_activity("autonomous_stopped", {})
    return jsonify({"status": "stopped", "message": "Autonomous worker stopped"})


@app.route('/api/autonomous/results')
def get_recent_results():
    """Get recent aggregated results from autonomous tasks."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503

    limit = request.args.get("limit", 5, type=int)
    results = autonomous_worker.get_recent_results(limit)
    return jsonify({"results": results, "count": len(results)})


@app.route('/api/autonomous/results/<task_id>')
def get_task_result(task_id):
    """Get aggregated result for a specific task."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503

    result = autonomous_worker.get_result(task_id)
    if result is None:
        return jsonify({"error": "Result not found"}), 404
    return jsonify(result)


@app.route('/api/autonomous/context')
def get_pm1000_context():
    """Get PM-1000 context from recent results."""
    if autonomous_worker is None:
        return jsonify({"error": "Autonomous worker not initialized"}), 503

    context = autonomous_worker.get_pm1000_context()
    return jsonify({"context": context})


@app.route('/api/queue/submit', methods=['POST'])
@rate_limit("10/minute")
def submit_to_queue():
    """Submit a task to the queue for autonomous processing."""
    if autonomous_worker is None or task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503

    data = request.get_json()
    goal = data.get("goal", "")
    project_dir = data.get("project_dir", ".")
    priority = data.get("priority", 2)

    if not goal:
        return jsonify({"error": "Goal required"}), 400

    task = autonomous_worker.submit_task(
        goal=goal,
        project_dir=project_dir,
        priority=priority,
        metadata=data.get("metadata", {}),
    )

    log_activity("task_queued", {"task_id": task.id, "goal": goal[:100]})

    return jsonify({
        "status": "queued",
        "task_id": task.id,
        "message": "Task added to queue for processing",
    })


@app.route('/api/queue/status')
def get_queue_status():
    """Get task queue status."""
    if task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503
    return jsonify(task_queue.get_queue_status())


@app.route('/api/queue/pending')
def get_pending_tasks():
    """Get pending tasks in the queue."""
    if task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503

    limit = request.args.get("limit", 20, type=int)
    tasks = task_queue.get_pending_tasks(limit=limit)
    return jsonify({"tasks": [t.to_dict() for t in tasks]})


@app.route('/api/queue/history')
def get_queue_history():
    """Get recent task execution history."""
    if task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503

    limit = request.args.get("limit", 20, type=int)
    history = task_queue.get_recent_history(limit=limit)
    return jsonify({"history": history})


@app.route('/api/queue/cancel/<task_id>', methods=['POST'])
def cancel_queued_task(task_id: str):
    """Cancel a pending task."""
    if task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503

    if task_queue.cancel_task(task_id):
        log_activity("task_cancelled", {"task_id": task_id})
        return jsonify({"status": "cancelled", "task_id": task_id})

    return jsonify({"error": "Task not found or not cancellable"}), 404


@app.route('/api/queue/patterns')
def get_learned_patterns():
    """Get learned outcome patterns."""
    if task_queue is None:
        return jsonify({"error": "Task queue not initialized"}), 503

    # Get sample of patterns for a test goal
    test_goal = request.args.get("goal", "add endpoint")
    patterns = task_queue.get_similar_patterns(test_goal, limit=5)

    return jsonify({
        "patterns": [
            {
                "pattern_hash": p.pattern_hash,
                "goal_pattern": p.goal_pattern,
                "project_type": p.project_type,
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "avg_duration": p.avg_duration,
                "success_rate": p.success_count / max(1, p.success_count + p.failure_count),
            }
            for p in patterns
        ]
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  PM Agent Orchestrator Dashboard")
    print("  Manage your AI agents like a pro")
    print("=" * 60)
    print(f"\n  Open: http://localhost:8080\n")

    socketio.run(app, host='0.0.0.0', port=8080, debug=True)

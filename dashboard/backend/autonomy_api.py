#!/usr/bin/env python3
"""
Autonomy API - RESTful endpoints for PM-1000 autonomy system.

Exposes Phase 2 Intelligence Generation components:
- Decision Engine: Multi-criteria opportunity evaluation
- Temporal Intelligence: Time-based scheduling
- Opportunity Scout: Codebase scanning
- Anti-Goal Generator: Prevention-focused goals
- Coding Agents: Configurable agent routing

Also includes integrated PM-1000 autonomy loop controls.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from flask import Blueprint, jsonify, request

from logging_config import get_logger

# Import autonomy components
from autonomy import (
    # Phase 0: Foundation
    get_safety_controller,
    get_config_manager,
    get_state_manager,
    is_safe_to_operate,
    emergency_stop,
    heartbeat,

    # Phase 1: Core Loop
    get_autonomous_loop,
    get_task_router,
    get_goal_manager,
    ModelTier,
    TaskComplexity as RouterTaskComplexity,
    GoalLevel,
    GoalStatus,
    GoalSource,

    # Phase 2: Intelligence
    create_decision_engine,
    create_temporal_intelligence,
    create_opportunity_scout,
    get_anti_goal_generator,
    OpportunityType,
    RiskLevel,
    TimeWindow,
    ActivityLevel,
    DayPart,
)

from coding_agents import (
    get_agent_manager,
    AgentType,
    TaskComplexity,
    AgentRequest,
)

logger = get_logger("pm1000.autonomy_api")

# Create Flask blueprint
autonomy_bp = Blueprint('autonomy', __name__, url_prefix='/api/autonomy')

# Lazy-initialized components
_decision_engine = None
_temporal_intel = None
_opportunity_scout = None
_anti_goal_gen = None


def get_decision_engine():
    """Get or create decision engine."""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = create_decision_engine()
    return _decision_engine


def get_temporal_intel():
    """Get or create temporal intelligence."""
    global _temporal_intel
    if _temporal_intel is None:
        _temporal_intel = create_temporal_intelligence()
    return _temporal_intel


def get_opportunity_scout_instance(project_path: str = "."):
    """Get or create opportunity scout."""
    global _opportunity_scout
    if _opportunity_scout is None:
        _opportunity_scout = create_opportunity_scout(project_path)
    return _opportunity_scout


def get_anti_goal_gen():
    """Get or create anti-goal generator."""
    global _anti_goal_gen
    if _anti_goal_gen is None:
        _anti_goal_gen = get_anti_goal_generator()
    return _anti_goal_gen


# ═══════════════════════════════════════════════════════════════════════════════
# Safety & Status Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/safety/status')
def safety_status():
    """Get current safety status."""
    try:
        safety = get_safety_controller()
        safe, reason = is_safe_to_operate()

        return jsonify({
            "safe_to_operate": safe,
            "reason": reason,
            "safety_level": safety.current_level.name if hasattr(safety, 'current_level') else "UNKNOWN",
            "kill_switches": safety.get_kill_switch_status() if hasattr(safety, 'get_kill_switch_status') else {},
            "constraints": safety.get_constraint_status() if hasattr(safety, 'get_constraint_status') else {}
        })
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/safety/emergency-stop', methods=['POST'])
def trigger_emergency_stop():
    """Trigger emergency stop."""
    try:
        reason = request.json.get("reason", "Manual emergency stop from dashboard")
        emergency_stop(reason)
        return jsonify({"status": "stopped", "reason": reason})
    except Exception as e:
        logger.error(f"Error triggering emergency stop: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/safety/heartbeat', methods=['POST'])
def send_heartbeat():
    """Send heartbeat to keep system alive."""
    try:
        heartbeat()
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Engine Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/decisions/evaluate', methods=['POST'])
def evaluate_opportunity():
    """Evaluate a single opportunity."""
    try:
        engine = get_decision_engine()
        opportunity = request.json

        decision = engine.evaluate_opportunity(opportunity)
        return jsonify(decision.to_dict())
    except Exception as e:
        logger.error(f"Error evaluating opportunity: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/decisions/rank', methods=['POST'])
def rank_opportunities():
    """Rank multiple opportunities."""
    try:
        engine = get_decision_engine()
        opportunities = request.json.get("opportunities", [])

        ranked = engine.rank_opportunities(opportunities)
        return jsonify({
            "count": len(ranked),
            "decisions": [d.to_dict() for d in ranked]
        })
    except Exception as e:
        logger.error(f"Error ranking opportunities: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/decisions/best', methods=['POST'])
def select_best_action():
    """Select the best action from opportunities."""
    try:
        engine = get_decision_engine()
        opportunities = request.json.get("opportunities", [])

        ranked = engine.rank_opportunities(opportunities)
        best = engine.select_best_action(ranked)

        if best:
            return jsonify({
                "found": True,
                "decision": best.to_dict()
            })
        else:
            return jsonify({
                "found": False,
                "reason": "No suitable action found"
            })
    except Exception as e:
        logger.error(f"Error selecting best action: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/decisions/metrics')
def decision_metrics():
    """Get decision engine metrics."""
    try:
        engine = get_decision_engine()
        return jsonify(engine.get_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/decisions/goals', methods=['POST'])
def set_decision_goals():
    """Set current goals for decision alignment."""
    try:
        engine = get_decision_engine()
        goals = request.json.get("goals", [])
        engine.set_current_goals(goals)
        return jsonify({"status": "ok", "goals": goals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Intelligence Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/temporal/status')
def temporal_status():
    """Get temporal intelligence status and metrics."""
    try:
        ti = get_temporal_intel()
        return jsonify(ti.get_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/schedule', methods=['POST'])
def recommend_schedule():
    """Get scheduling recommendation for a task."""
    try:
        ti = get_temporal_intel()
        data = request.json

        recommendation = ti.recommend_schedule(
            task_id=data.get("task_id", "unknown"),
            task_type=data.get("task_type", "general"),
            estimated_minutes=data.get("estimated_minutes", 30),
            deadline_id=data.get("deadline_id"),
            constraints=data.get("constraints", {})
        )

        return jsonify({
            "task_id": recommendation.task_id,
            "recommended_start": recommendation.recommended_start.isoformat(),
            "window": recommendation.recommended_window.value,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning,
            "alternatives": [
                {"time": t.isoformat(), "score": s}
                for t, s in recommendation.alternatives
            ]
        })
    except Exception as e:
        logger.error(f"Error getting schedule recommendation: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/deadlines', methods=['GET'])
def list_deadlines():
    """List all active deadlines."""
    try:
        ti = get_temporal_intel()
        ranking = ti.get_urgency_ranking()
        return jsonify({
            "count": len(ranking),
            "deadlines": ranking
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/deadlines', methods=['POST'])
def add_deadline():
    """Add a new deadline."""
    try:
        ti = get_temporal_intel()
        data = request.json

        due_at = datetime.fromisoformat(data["due_at"])

        result = ti.add_deadline(
            id=data.get("id", f"dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            description=data.get("description", ""),
            due_at=due_at,
            priority=data.get("priority", 3),
            is_hard=data.get("is_hard", False),
            warning_hours=data.get("warning_hours", 24.0)
        )

        return jsonify({"success": result})
    except Exception as e:
        logger.error(f"Error adding deadline: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/activity', methods=['POST'])
def record_activity():
    """Record an activity for pattern learning."""
    try:
        ti = get_temporal_intel()
        data = request.json

        level = ActivityLevel[data.get("level", "MODERATE").upper()]

        ti.record_activity(
            activity_type=data.get("activity_type", "coding"),
            level=level,
            duration_minutes=data.get("duration_minutes", 30),
            metadata=data.get("metadata", {})
        )

        return jsonify({"status": "recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/patterns')
def get_patterns():
    """Get learned temporal patterns."""
    try:
        ti = get_temporal_intel()
        patterns = ti.analyze_patterns()

        return jsonify({
            "count": len(patterns),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "day_part": p.day_part.value if p.day_part else None,
                    "day_of_week": p.day_of_week,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "observations": p.observations,
                    "metadata": p.metadata
                }
                for p in patterns
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/optimal-times')
def get_optimal_times():
    """Get optimal execution times."""
    try:
        ti = get_temporal_intel()
        count = request.args.get("count", 5, type=int)

        times = ti.get_optimal_execution_times(task_count=count)

        return jsonify({
            "optimal_times": [t.isoformat() for t in times]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/temporal/slots')
def get_available_slots():
    """Get available time slots."""
    try:
        ti = get_temporal_intel()
        hours = request.args.get("hours", 24, type=int)

        slots = ti.get_available_slots(hours_ahead=hours)

        return jsonify({
            "count": len(slots),
            "slots": [s.to_dict() for s in slots]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Opportunity Scout Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/opportunities/scan', methods=['POST'])
def scan_opportunities():
    """Scan codebase for opportunities."""
    try:
        project_path = request.json.get("project_path", os.getcwd())
        scout = get_opportunity_scout_instance(project_path)

        opportunities = scout.scan_all()

        return jsonify({
            "count": len(opportunities),
            "opportunities": [
                {
                    "id": opp.id,
                    "type": opp.type,
                    "description": opp.description,
                    "priority": opp.priority.name if hasattr(opp.priority, 'name') else str(opp.priority),
                    "source": getattr(opp, 'source', None),
                    "file_path": getattr(opp, 'file_path', None),
                    "line": getattr(opp, 'line', None),
                    "metadata": getattr(opp, 'metadata', {})
                }
                for opp in opportunities
            ]
        })
    except Exception as e:
        logger.error(f"Error scanning opportunities: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/opportunities/types')
def opportunity_types():
    """List available opportunity types."""
    return jsonify({
        "types": [t.value for t in OpportunityType]
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-Goal Generator Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/anti-goals/check', methods=['POST'])
def check_anti_goals():
    """Check for anti-goal triggers."""
    try:
        gen = get_anti_goal_gen()
        context = request.json.get("context", {})

        triggers = gen.check_anti_goals(context)

        return jsonify({
            "count": len(triggers),
            "triggers": [t.to_dict() for t in triggers]
        })
    except Exception as e:
        logger.error(f"Error checking anti-goals: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/anti-goals/generate', methods=['POST'])
def generate_prevention_goals():
    """Generate prevention goals from triggers."""
    try:
        gen = get_anti_goal_gen()
        context = request.json.get("context", {})

        # First check for triggers
        triggers = gen.check_anti_goals(context)

        if triggers:
            # Generate prevention goals
            goals = gen.generate_prevention_goals(triggers)
            return jsonify({
                "triggers_found": len(triggers),
                "goals_generated": len(goals),
                "goals": [
                    {
                        "id": g.id,
                        "level": g.level.name,
                        "title": g.title,
                        "description": g.description,
                        "status": g.status.name,
                        "priority": g.priority.name if hasattr(g.priority, 'name') else str(g.priority)
                    }
                    for g in goals
                ]
            })
        else:
            return jsonify({
                "triggers_found": 0,
                "goals_generated": 0,
                "goals": []
            })
    except Exception as e:
        logger.error(f"Error generating prevention goals: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/anti-goals/list')
def list_anti_goals():
    """List all configured anti-goals."""
    try:
        gen = get_anti_goal_gen()

        # Get registered anti-goals
        anti_goals = list(gen._anti_goals.values())

        return jsonify({
            "count": len(anti_goals),
            "anti_goals": [ag.to_dict() for ag in anti_goals]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Coding Agent Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/agents/status')
def agents_status():
    """Get status of all coding agents."""
    try:
        manager = get_agent_manager()
        return jsonify(manager.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/<agent_type>/config', methods=['PATCH'])
def update_agent_config(agent_type: str):
    """Update agent configuration."""
    try:
        manager = get_agent_manager()
        agent_enum = AgentType(agent_type)

        updates = request.json
        success = manager.update_agent_config(agent_enum, updates)

        if success:
            agent = manager.get_agent(agent_enum)
            return jsonify({"status": "updated", "config": agent.config.to_dict()})
        else:
            return jsonify({"error": "Agent not found"}), 404
    except ValueError:
        return jsonify({"error": f"Unknown agent type: {agent_type}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/force', methods=['POST'])
def force_agent():
    """Force all tasks to use a specific agent."""
    try:
        manager = get_agent_manager()
        agent_type = request.json.get("agent_type")

        if agent_type:
            agent_enum = AgentType(agent_type)
            manager.set_force_agent(agent_enum)
        else:
            manager.set_force_agent(None)

        return jsonify({
            "status": "ok",
            "force_agent": agent_type
        })
    except ValueError:
        return jsonify({"error": f"Unknown agent type: {agent_type}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/force', methods=['DELETE'])
def clear_force_agent():
    """Clear forced agent override, return to auto-selection."""
    try:
        manager = get_agent_manager()
        manager.set_force_agent(None)
        return jsonify({"status": "ok", "force_agent": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/reset-limits', methods=['POST'])
def reset_agent_limits():
    """Reset agent usage limits and counters."""
    try:
        manager = get_agent_manager()
        manager.reset_usage()
        return jsonify({"status": "ok", "message": "Usage limits reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/recommend', methods=['POST'])
def recommend_agent():
    """Get agent recommendations for a task."""
    try:
        manager = get_agent_manager()
        task_type = request.json.get("task_type", "general")
        complexity = request.json.get("complexity", "MODERATE")

        recommendations = manager.get_recommendations(task_type, complexity)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/agents/execute', methods=['POST'])
def execute_with_agent():
    """Execute a task with agent routing."""
    try:
        manager = get_agent_manager()
        data = request.json

        # Build request
        request_obj = AgentRequest(
            task_id=data.get("task_id", f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            task_type=data.get("task_type", "general"),
            prompt=data.get("prompt", ""),
            working_directory=data.get("working_directory", os.getcwd()),
            complexity=TaskComplexity[data.get("complexity", "MODERATE").upper()],
            timeout_seconds=data.get("timeout_seconds", 300),
            metadata=data.get("metadata", {})
        )

        # Execute
        if "agent_type" in data:
            # Use specific agent
            agent_enum = AgentType(data["agent_type"])
            response = manager.execute_with_agent(agent_enum, request_obj)
        else:
            # Auto-select agent
            response = manager.execute(request_obj)

        return jsonify(response.to_dict())
    except Exception as e:
        logger.error(f"Error executing with agent: {e}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Task Router Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/router/status')
def router_status():
    """Get task router status and metrics."""
    try:
        router = get_task_router()
        return jsonify(router.get_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/router/analyze', methods=['POST'])
def analyze_task():
    """Analyze a task for routing."""
    try:
        router = get_task_router()
        data = request.json

        # Create a task for analysis
        from autonomy import Task
        task = Task(
            id=data.get("id", "analysis_task"),
            type=data.get("type", "general"),
            description=data.get("description", ""),
            priority=data.get("priority", "MEDIUM"),
            source=data.get("source", "manual")
        )

        analysis = router.analyze_task(task)

        return jsonify({
            "complexity": analysis.complexity.name,
            "recommended_tier": analysis.recommended_tier.name,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Goal Manager Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/goals', methods=['GET'])
def list_goals():
    """List all goals."""
    try:
        gm = get_goal_manager()
        goals = gm.get_all_goals()

        return jsonify({
            "count": len(goals),
            "goals": [g.to_dict() for g in goals]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/goals', methods=['POST'])
def create_goal():
    """Create a new goal."""
    try:
        gm = get_goal_manager()
        data = request.json

        goal = gm.create_goal(
            title=data.get("title"),
            description=data.get("description", ""),
            level=GoalLevel[data.get("level", "GOAL").upper()],
            source=GoalSource[data.get("source", "USER").upper()],
            parent_id=data.get("parent_id")
        )

        return jsonify(goal.to_dict())
    except Exception as e:
        logger.error(f"Error creating goal: {e}")
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/goals/<goal_id>/status', methods=['PATCH'])
def update_goal_status(goal_id: str):
    """Update goal status."""
    try:
        gm = get_goal_manager()
        status = GoalStatus[request.json.get("status", "ACTIVE").upper()]

        success = gm.update_goal_status(goal_id, status)

        if success:
            goal = gm.get_goal(goal_id)
            return jsonify(goal.to_dict())
        else:
            return jsonify({"error": "Goal not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/goals/metrics')
def goal_metrics():
    """Get goal metrics."""
    try:
        gm = get_goal_manager()
        return jsonify(gm.get_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Integrated Autonomy Loop Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/loop/status')
def loop_status():
    """Get autonomy loop status."""
    try:
        loop = get_autonomous_loop()
        return jsonify({
            "running": loop.is_running if hasattr(loop, 'is_running') else False,
            "phase": loop.current_phase.name if hasattr(loop, 'current_phase') else "UNKNOWN",
            "metrics": loop.get_metrics() if hasattr(loop, 'get_metrics') else {},
            "iteration_count": getattr(loop, 'iteration_count', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/loop/start', methods=['POST'])
def start_loop():
    """Start the autonomy loop."""
    try:
        loop = get_autonomous_loop()
        loop.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@autonomy_bp.route('/loop/stop', methods=['POST'])
def stop_loop():
    """Stop the autonomy loop."""
    try:
        loop = get_autonomous_loop()
        loop.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Summary Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@autonomy_bp.route('/dashboard')
def dashboard_summary():
    """Get comprehensive dashboard summary."""
    try:
        # Gather all component statuses
        summary = {
            "timestamp": datetime.now().isoformat(),
            "safety": {},
            "agents": {},
            "decisions": {},
            "temporal": {},
            "opportunities": {},
            "goals": {},
            "loop": {}
        }

        # Safety status
        try:
            safe, reason = is_safe_to_operate()
            summary["safety"] = {
                "safe_to_operate": safe,
                "is_nominal": safe,  # Frontend expects this
                "reason": reason
            }
        except Exception as e:
            summary["safety"] = {"error": str(e), "is_nominal": False}

        # Agent status
        try:
            manager = get_agent_manager()
            status = manager.get_status()
            summary["agents"] = {
                "total_agents": len(status.get("agents", {})),
                "default_agent": status.get("default_agent"),
                "force_override": status.get("force_agent"),  # Frontend expects this name
                "total_cost": status.get("total_cost", 0),
                "total_cost_today": status.get("total_cost", 0),  # Frontend expects this
                "max_cost_daily": status.get("max_cost_daily", 10.0),
                "total_requests": status.get("total_requests", 0),
                "total_requests_hour": status.get("requests_this_hour", 0),  # Frontend expects this
                "max_requests_hour": status.get("max_requests_hour", 50)
            }
        except Exception as e:
            summary["agents"] = {"error": str(e)}

        # Decision engine metrics
        try:
            engine = get_decision_engine()
            metrics = engine.get_metrics()
            summary["decisions"] = metrics
        except Exception as e:
            summary["decisions"] = {"error": str(e)}

        # Temporal intelligence metrics
        try:
            ti = get_temporal_intel()
            metrics = ti.get_metrics()
            # Count upcoming deadlines
            ranking = ti.get_urgency_ranking()
            summary["temporal"] = {
                **metrics,
                "upcoming_deadlines": len(ranking)  # Frontend expects this
            }
        except Exception as e:
            summary["temporal"] = {"error": str(e), "upcoming_deadlines": 0}

        # Opportunities (count pending decisions)
        try:
            summary["opportunities"] = {
                "pending_decisions": summary.get("decisions", {}).get("pending_count", 0)
            }
        except Exception as e:
            summary["opportunities"] = {"error": str(e), "pending_decisions": 0}

        # Goal metrics
        try:
            gm = get_goal_manager()
            summary["goals"] = gm.get_metrics()
        except Exception as e:
            summary["goals"] = {"error": str(e)}

        # Loop status
        try:
            loop = get_autonomous_loop()
            summary["loop"] = {
                "running": loop.is_running if hasattr(loop, 'is_running') else False,
                "iteration_count": getattr(loop, 'iteration_count', 0)
            }
        except Exception as e:
            summary["loop"] = {"error": str(e)}

        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error generating dashboard summary: {e}")
        return jsonify({"error": str(e)}), 500


def register_autonomy_routes(app):
    """Register autonomy blueprint with Flask app."""
    app.register_blueprint(autonomy_bp)
    logger.info("Autonomy API routes registered")

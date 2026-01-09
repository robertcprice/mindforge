#!/usr/bin/env python3
"""
Autonomy Package for PM-1000

Phase 0 Foundation Components:
- ConfigManager: Enhanced configuration with autonomy dials and safety constraints
- ErrorFramework: Comprehensive error handling with graceful degradation
- SystemStateManager: Checkpointing, crash recovery, transaction support
- SafetyController: THE MOST CRITICAL - kill switches, constraints, emergency stop

Phase 1 Core Loop Components:
- AutonomousLoop: Main sense-think-act loop for autonomous operation
- TaskRouter: Multi-tier model routing for optimal resource usage
- GoalManager: Goal generation, decomposition, and lifecycle management

Phase 2 Intelligence Generation Components:
- OpportunityScout: Advanced opportunity detection with multiple scanners
- AntiGoalGenerator: Prevention-focused goal generation
- DecisionEngine: Multi-criteria decision making and planning
- TemporalIntelligence: Time-based optimization and scheduling

Usage:
    from autonomy import (
        get_safety_controller,
        get_state_manager,
        get_config_manager,
        get_error_handler,
        get_autonomous_loop,
        get_task_router,
        get_goal_manager,
        create_decision_engine,
        create_temporal_intelligence,
        is_safe_to_operate,
        emergency_stop,
    )

    # Always check safety before operations
    safe, reason = is_safe_to_operate()
    if not safe:
        raise RuntimeError(f"Not safe to operate: {reason}")

    # Start the autonomous loop
    loop = get_autonomous_loop()
    loop.run()
"""

# Config Manager
from .config_manager import (
    ConfigChangeType,
    ResourceBudget,
    AutonomyDial,
    AutonomyDialsConfig,
    SafetyConstraint,
    SafetyConstraintsConfig,
    KillSwitchConfig,
    CommunicationConfig,
    LearningConfig,
    AutonomyConfig,
    ConfigChangeListener,
    AutonomyConfigManager,
    get_config_manager,
    init_config_manager,
    get_resource_budget,
    get_autonomy_dials,
    get_safety_constraints,
    get_kill_switch_config,
    get_communication_config,
    get_learning_config,
    is_autonomous_mode,
)

# Error Framework
from .error_framework import (
    ErrorSeverity,
    ErrorCategory,
    RecoveryAction,
    DegradationLevel,
    ErrorContext,
    ErrorRecord,
    RecoveryStrategy,
    PM1000Error,
    NetworkError,
    APIError,
    ResourceExhaustedError,
    BudgetExceededError,
    ValidationError,
    ExecutionError,
    StateCorruptionError,
    SecurityViolationError,
    TimeoutError,
    DependencyError,
    SafetyConstraintViolation as ErrorFrameworkSafetyViolation,
    ErrorAggregator,
    ErrorHandler,
    get_error_handler,
    handle_error,
    error_context,
    with_error_handling,
    error_boundary,
    is_degraded,
    get_degradation_level,
)

# System State Manager
from .system_state_manager import (
    StateVersion,
    TransactionStatus,
    CheckpointType,
    AutonomousLoopState,
    ResourceUsageState,
    LearningState,
    SafetyState,
    Checkpoint,
    TransactionLog,
    SystemStateManager,
    get_state_manager,
    init_state_manager,
)

# Safety Controller - THE MOST CRITICAL
from .safety_controller import (
    SafetyLevel,
    ConstraintSeverity,
    KillSwitchType,
    ActionType,
    Action,
    ConstraintViolation,
    SafetyCheckResult,
    KillSwitchStatus,
    Constraint,
    NoDataDestructionConstraint,
    TestsMustPassConstraint,
    NoSecretsInCodeConstraint,
    BudgetLimitsConstraint,
    MaxFileChangesConstraint,
    NoProductionDeployConstraint,
    KillSwitch,
    FileBasedKillSwitch,
    TimeBasedKillSwitch,
    BudgetBasedKillSwitch,
    FailureBasedKillSwitch,
    HeartbeatBasedKillSwitch,
    SafetyController,
    get_safety_controller,
    init_safety_controller,
    is_safe_to_operate,
    validate_action,
    emergency_stop,
    heartbeat,
    safety_check,
    SafetyConstraintViolation,
)

# =========================================================================
# Phase 1: Core Loop Components
# =========================================================================

# Autonomous Loop
from .autonomous_loop import (
    LoopPhase,
    TaskPriority,
    Opportunity,
    Task,
    LoopMetrics,
    OpportunityScanner,
    TaskExecutor,
    AutonomousLoop,
    get_autonomous_loop,
    init_autonomous_loop,
)

# Task Router
from .task_router import (
    ModelTier,
    TaskComplexity,
    ModelConfig,
    TaskAnalysis,
    RoutingDecision,
    RoutingOutcome,
    RouterMetrics,
    ComplexityAnalyzer,
    TaskRouter,
    get_task_router,
    init_task_router,
)

# Goal Manager
from .goal_manager import (
    GoalLevel,
    GoalStatus,
    GoalSource,
    GoalPriority,
    GoalMetadata,
    Goal,
    GoalMetrics,
    GoalGenerator,
    GoalDecomposer,
    GoalManager,
    get_goal_manager,
    init_goal_manager,
)

# =========================================================================
# Phase 2: Intelligence Generation Components
# =========================================================================

# Opportunity Scout
from .opportunity_scout import (
    OpportunityType,
    ScanResult,
    OpportunityScout,
    TodoScanner,
    TestCoverageScanner,
    DocGapScanner,
    CodeSmellScanner,
    TypeAnnotationScanner,
    create_opportunity_scout,
)

# Anti-Goal Generator
from .anti_goal_generator import (
    AntiGoalType,
    TriggerSeverity,
    AntiGoalTrigger,
    AntiGoal,
    AntiGoalDetector,
    TechDebtDetector,
    TestBrittlenessDetector,
    DocDriftDetector,
    SecurityRegressionDetector,
    PerformanceDegradationDetector,
    AntiGoalGenerator,
    get_anti_goal_generator,
    init_anti_goal_generator,
)

# Decision Engine
from .decision_engine import (
    RiskLevel,
    DecisionConfidence,
    ExecutionStrategy,
    ExecutionStep,
    ExecutionPlan,
    RiskAssessment,
    CriteriaScore,
    Decision,
    CriteriaEvaluator,
    ExecutionPlanner,
    RiskAnalyzer,
    DecisionEngine,
    create_decision_engine,
)

# Temporal Intelligence
from .temporal_intelligence import (
    TimeWindow,
    ActivityLevel,
    DayPart,
    TimeSlot,
    Deadline,
    TemporalPattern,
    ScheduleRecommendation,
    ActivityTracker,
    DeadlineManager,
    ScheduleOptimizer,
    TemporalIntelligence,
    create_temporal_intelligence,
    get_day_part,
)

__all__ = [
    # Config Manager
    "ConfigChangeType",
    "ResourceBudget",
    "AutonomyDial",
    "AutonomyDialsConfig",
    "SafetyConstraint",
    "SafetyConstraintsConfig",
    "KillSwitchConfig",
    "CommunicationConfig",
    "LearningConfig",
    "AutonomyConfig",
    "ConfigChangeListener",
    "AutonomyConfigManager",
    "get_config_manager",
    "init_config_manager",
    "get_resource_budget",
    "get_autonomy_dials",
    "get_safety_constraints",
    "get_kill_switch_config",
    "get_communication_config",
    "get_learning_config",
    "is_autonomous_mode",

    # Error Framework
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryAction",
    "DegradationLevel",
    "ErrorContext",
    "ErrorRecord",
    "RecoveryStrategy",
    "PM1000Error",
    "NetworkError",
    "APIError",
    "ResourceExhaustedError",
    "BudgetExceededError",
    "ValidationError",
    "ExecutionError",
    "StateCorruptionError",
    "SecurityViolationError",
    "TimeoutError",
    "DependencyError",
    "ErrorFrameworkSafetyViolation",
    "ErrorAggregator",
    "ErrorHandler",
    "get_error_handler",
    "handle_error",
    "error_context",
    "with_error_handling",
    "error_boundary",
    "is_degraded",
    "get_degradation_level",

    # System State Manager
    "StateVersion",
    "TransactionStatus",
    "CheckpointType",
    "AutonomousLoopState",
    "ResourceUsageState",
    "LearningState",
    "SafetyState",
    "Checkpoint",
    "TransactionLog",
    "SystemStateManager",
    "get_state_manager",
    "init_state_manager",

    # Safety Controller
    "SafetyLevel",
    "ConstraintSeverity",
    "KillSwitchType",
    "ActionType",
    "Action",
    "ConstraintViolation",
    "SafetyCheckResult",
    "KillSwitchStatus",
    "Constraint",
    "NoDataDestructionConstraint",
    "TestsMustPassConstraint",
    "NoSecretsInCodeConstraint",
    "BudgetLimitsConstraint",
    "MaxFileChangesConstraint",
    "NoProductionDeployConstraint",
    "KillSwitch",
    "FileBasedKillSwitch",
    "TimeBasedKillSwitch",
    "BudgetBasedKillSwitch",
    "FailureBasedKillSwitch",
    "HeartbeatBasedKillSwitch",
    "SafetyController",
    "get_safety_controller",
    "init_safety_controller",
    "is_safe_to_operate",
    "validate_action",
    "emergency_stop",
    "heartbeat",
    "safety_check",
    "SafetyConstraintViolation",

    # Autonomous Loop
    "LoopPhase",
    "TaskPriority",
    "Opportunity",
    "Task",
    "LoopMetrics",
    "OpportunityScanner",
    "TaskExecutor",
    "AutonomousLoop",
    "get_autonomous_loop",
    "init_autonomous_loop",

    # Task Router
    "ModelTier",
    "TaskComplexity",
    "ModelConfig",
    "TaskAnalysis",
    "RoutingDecision",
    "RoutingOutcome",
    "RouterMetrics",
    "ComplexityAnalyzer",
    "TaskRouter",
    "get_task_router",
    "init_task_router",

    # Goal Manager
    "GoalLevel",
    "GoalStatus",
    "GoalSource",
    "GoalPriority",
    "GoalMetadata",
    "Goal",
    "GoalMetrics",
    "GoalGenerator",
    "GoalDecomposer",
    "GoalManager",
    "get_goal_manager",
    "init_goal_manager",

    # Opportunity Scout
    "OpportunityType",
    "ScanResult",
    "OpportunityScout",
    "TodoScanner",
    "TestCoverageScanner",
    "DocGapScanner",
    "CodeSmellScanner",
    "TypeAnnotationScanner",
    "create_opportunity_scout",

    # Anti-Goal Generator
    "AntiGoalType",
    "TriggerSeverity",
    "AntiGoalTrigger",
    "AntiGoal",
    "AntiGoalDetector",
    "TechDebtDetector",
    "TestBrittlenessDetector",
    "DocDriftDetector",
    "SecurityRegressionDetector",
    "PerformanceDegradationDetector",
    "AntiGoalGenerator",
    "get_anti_goal_generator",
    "init_anti_goal_generator",

    # Decision Engine
    "RiskLevel",
    "DecisionConfidence",
    "ExecutionStrategy",
    "ExecutionStep",
    "ExecutionPlan",
    "RiskAssessment",
    "CriteriaScore",
    "Decision",
    "CriteriaEvaluator",
    "ExecutionPlanner",
    "RiskAnalyzer",
    "DecisionEngine",
    "create_decision_engine",

    # Temporal Intelligence
    "TimeWindow",
    "ActivityLevel",
    "DayPart",
    "TimeSlot",
    "Deadline",
    "TemporalPattern",
    "ScheduleRecommendation",
    "ActivityTracker",
    "DeadlineManager",
    "ScheduleOptimizer",
    "TemporalIntelligence",
    "create_temporal_intelligence",
    "get_day_part",
]

__version__ = "0.3.0"  # Phase 2 complete
__author__ = "PM-1000 Autonomous System"

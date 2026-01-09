#!/usr/bin/env python3
"""
Enhanced Configuration Manager for PM-1000 Autonomous Operation

Provides:
- Hot reload capability with file watching
- Resource budgets for autonomous operation
- Safety constraints configuration
- Autonomy dial settings
- Configuration validation and change notifications
- Environment-specific overrides
"""

import os
import json
import threading
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

logger = get_logger("pm1000.autonomy.config")


class ConfigChangeType(Enum):
    """Types of configuration changes."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RELOADED = "reloaded"


@dataclass
class ResourceBudget:
    """Resource budget configuration for autonomous operation."""
    # API Cost Limits (USD per day)
    daily_api_budget: float = field(default_factory=lambda: float(os.getenv("DAILY_API_BUDGET", "50.0")))
    openai_daily_limit: float = field(default_factory=lambda: float(os.getenv("OPENAI_DAILY_LIMIT", "20.0")))
    anthropic_daily_limit: float = field(default_factory=lambda: float(os.getenv("ANTHROPIC_DAILY_LIMIT", "30.0")))

    # Session Limits
    max_concurrent_sessions: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_SESSIONS", "3")))
    max_sessions_per_hour: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS_PER_HOUR", "20")))
    max_sessions_per_day: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS_PER_DAY", "100")))

    # Compute Limits
    max_task_duration_seconds: int = field(default_factory=lambda: int(os.getenv("MAX_TASK_DURATION", "3600")))
    max_memory_mb: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_MB", "4096")))

    # Rate Limits
    api_calls_per_minute: int = field(default_factory=lambda: int(os.getenv("API_CALLS_PER_MINUTE", "60")))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomyDial:
    """Configuration for a single autonomy dial."""
    name: str
    current_level: float  # 0.0 to 1.0
    min_level: float = 0.0
    max_level: float = 1.0
    adjustment_rate: float = 0.1  # How much to adjust per cycle
    description: str = ""

    def adjust(self, delta: float) -> float:
        """Adjust the dial level within bounds."""
        new_level = max(self.min_level, min(self.max_level, self.current_level + delta))
        old_level = self.current_level
        self.current_level = new_level
        return new_level - old_level  # Return actual change


@dataclass
class AutonomyDialsConfig:
    """Configuration for all autonomy dials."""
    goal_generation: AutonomyDial = field(default_factory=lambda: AutonomyDial(
        name="goal_generation",
        current_level=float(os.getenv("AUTONOMY_GOAL_GEN", "0.3")),
        description="Level of autonomous goal generation (0=none, 1=full)"
    ))
    execution: AutonomyDial = field(default_factory=lambda: AutonomyDial(
        name="execution",
        current_level=float(os.getenv("AUTONOMY_EXECUTION", "0.8")),
        description="Level of autonomous task execution"
    ))
    learning: AutonomyDial = field(default_factory=lambda: AutonomyDial(
        name="learning",
        current_level=float(os.getenv("AUTONOMY_LEARNING", "0.6")),
        description="Level of autonomous learning and adaptation"
    ))
    communication: AutonomyDial = field(default_factory=lambda: AutonomyDial(
        name="communication",
        current_level=float(os.getenv("AUTONOMY_COMMUNICATION", "0.2")),
        description="Level of autonomous external communication"
    ))
    self_modification: AutonomyDial = field(default_factory=lambda: AutonomyDial(
        name="self_modification",
        current_level=float(os.getenv("AUTONOMY_SELF_MOD", "0.1")),
        max_level=0.5,  # Hard cap on self-modification
        description="Level of autonomous self-modification"
    ))

    def get_all_dials(self) -> Dict[str, AutonomyDial]:
        return {
            "goal_generation": self.goal_generation,
            "execution": self.execution,
            "learning": self.learning,
            "communication": self.communication,
            "self_modification": self.self_modification,
        }

    def emergency_lockdown(self):
        """Set all dials to minimum safe levels."""
        for dial in self.get_all_dials().values():
            dial.current_level = max(dial.min_level, 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {name: asdict(dial) for name, dial in self.get_all_dials().items()}


@dataclass
class SafetyConstraint:
    """Definition of a safety constraint."""
    name: str
    description: str
    enabled: bool = True
    severity: str = "critical"  # critical, high, medium, low
    action_on_violation: str = "block"  # block, warn, log

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyConstraintsConfig:
    """Configuration for safety constraints."""
    # Immutable constraints - these cannot be disabled
    no_data_destruction: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="no_data_destruction",
        description="Never delete databases, files, or data without backup",
        severity="critical",
        action_on_violation="block"
    ))
    tests_must_pass: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="tests_must_pass",
        description="Never commit code that breaks existing tests",
        severity="critical",
        action_on_violation="block"
    ))
    no_secrets_in_code: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="no_secrets_in_code",
        description="Never commit API keys, passwords, or credentials",
        severity="critical",
        action_on_violation="block"
    ))
    budget_limits: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="budget_limits",
        description="Never exceed daily API budget limits",
        severity="critical",
        action_on_violation="block"
    ))
    human_approval_required: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="human_approval_required",
        description="Require human approval for high-risk actions",
        severity="high",
        action_on_violation="block"
    ))

    # Configurable constraints
    max_file_changes: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="max_file_changes",
        description="Limit the number of files changed per task",
        severity="medium",
        action_on_violation="warn"
    ))
    no_production_deploy: SafetyConstraint = field(default_factory=lambda: SafetyConstraint(
        name="no_production_deploy",
        description="Never deploy directly to production",
        severity="high",
        action_on_violation="block"
    ))

    def get_all_constraints(self) -> Dict[str, SafetyConstraint]:
        return {
            "no_data_destruction": self.no_data_destruction,
            "tests_must_pass": self.tests_must_pass,
            "no_secrets_in_code": self.no_secrets_in_code,
            "budget_limits": self.budget_limits,
            "human_approval_required": self.human_approval_required,
            "max_file_changes": self.max_file_changes,
            "no_production_deploy": self.no_production_deploy,
        }

    def get_enabled_constraints(self) -> Dict[str, SafetyConstraint]:
        return {k: v for k, v in self.get_all_constraints().items() if v.enabled}

    def get_critical_constraints(self) -> Dict[str, SafetyConstraint]:
        return {k: v for k, v in self.get_all_constraints().items()
                if v.severity == "critical" and v.enabled}

    def to_dict(self) -> Dict[str, Any]:
        return {name: c.to_dict() for name, c in self.get_all_constraints().items()}


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch mechanisms."""
    # File-based kill switch
    kill_file_path: str = field(default_factory=lambda: os.getenv(
        "KILL_SWITCH_FILE",
        "/tmp/PM1000_EMERGENCY_STOP"
    ))

    # Time-based limits
    max_continuous_operation_hours: int = field(default_factory=lambda: int(os.getenv(
        "MAX_CONTINUOUS_HOURS", "24"
    )))

    # Resource-based triggers
    budget_exhaustion_threshold: float = field(default_factory=lambda: float(os.getenv(
        "BUDGET_EXHAUSTION_THRESHOLD", "0.95"
    )))

    # Error-based triggers
    max_consecutive_failures: int = field(default_factory=lambda: int(os.getenv(
        "MAX_CONSECUTIVE_FAILURES", "5"
    )))

    # Human override timeout (auto-stop if no human response)
    human_response_timeout_hours: int = field(default_factory=lambda: int(os.getenv(
        "HUMAN_RESPONSE_TIMEOUT", "4"
    )))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommunicationConfig:
    """Configuration for human communication channels."""
    # Discord
    discord_enabled: bool = field(default_factory=lambda: os.getenv("DISCORD_ENABLED", "false").lower() == "true")
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    discord_bot_token: str = field(default_factory=lambda: os.getenv("DISCORD_BOT_TOKEN", ""))
    discord_channel_id: str = field(default_factory=lambda: os.getenv("DISCORD_CHANNEL_ID", ""))

    # Email
    email_enabled: bool = field(default_factory=lambda: os.getenv("EMAIL_ENABLED", "false").lower() == "true")
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    email_to: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))
    email_from: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))

    # Notification preferences
    notify_on_error: bool = True
    notify_on_completion: bool = True
    notify_on_approval_needed: bool = True
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 9  # 9 AM

    def to_dict(self) -> Dict[str, Any]:
        # Don't expose sensitive credentials
        d = asdict(self)
        if d.get("smtp_password"):
            d["smtp_password"] = "***REDACTED***"
        if d.get("discord_bot_token"):
            d["discord_bot_token"] = "***REDACTED***"
        return d


@dataclass
class LearningConfig:
    """Configuration for learning system."""
    # Learning rate
    base_learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.1")))

    # Pattern storage
    max_patterns_stored: int = field(default_factory=lambda: int(os.getenv("MAX_PATTERNS", "10000")))
    pattern_expiry_days: int = field(default_factory=lambda: int(os.getenv("PATTERN_EXPIRY_DAYS", "90")))

    # Confidence thresholds
    min_confidence_for_autonomous: float = field(default_factory=lambda: float(os.getenv(
        "MIN_CONFIDENCE_AUTONOMOUS", "0.85"
    )))
    min_confidence_for_review: float = field(default_factory=lambda: float(os.getenv(
        "MIN_CONFIDENCE_REVIEW", "0.65"
    )))

    # Learning dimensions (from multi-model synthesis)
    enabled_dimensions: List[str] = field(default_factory=lambda: [
        "quality", "effort", "risk"  # Start with 3 dimensions as recommended
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomyConfig:
    """Master configuration for autonomous operation."""
    # Sub-configurations
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    autonomy_dials: AutonomyDialsConfig = field(default_factory=AutonomyDialsConfig)
    safety_constraints: SafetyConstraintsConfig = field(default_factory=SafetyConstraintsConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

    # Global settings
    environment: str = field(default_factory=lambda: os.getenv("PM1000_ENV", "development"))
    version: str = "1.0.0"

    # Operation mode
    autonomous_mode_enabled: bool = field(default_factory=lambda: os.getenv(
        "AUTONOMOUS_MODE", "false"
    ).lower() == "true")

    # Logging
    verbose_logging: bool = field(default_factory=lambda: os.getenv(
        "VERBOSE_LOGGING", "false"
    ).lower() == "true")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "version": self.version,
            "autonomous_mode_enabled": self.autonomous_mode_enabled,
            "verbose_logging": self.verbose_logging,
            "resource_budget": self.resource_budget.to_dict(),
            "autonomy_dials": self.autonomy_dials.to_dict(),
            "safety_constraints": self.safety_constraints.to_dict(),
            "kill_switch": self.kill_switch.to_dict(),
            "communication": self.communication.to_dict(),
            "learning": self.learning.to_dict(),
        }

    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings/errors."""
        issues = []

        if self.environment == "production":
            if self.autonomy_dials.goal_generation.current_level > 0.5:
                issues.append("WARNING: Goal generation dial > 0.5 in production")
            if self.autonomy_dials.self_modification.current_level > 0.2:
                issues.append("WARNING: Self modification dial > 0.2 in production")

        if self.autonomous_mode_enabled:
            if not self.communication.discord_enabled and not self.communication.email_enabled:
                issues.append("WARNING: Autonomous mode enabled but no communication channels configured")

        if self.resource_budget.daily_api_budget > 100:
            issues.append(f"WARNING: High daily API budget (${self.resource_budget.daily_api_budget})")

        return issues


class ConfigChangeListener:
    """Listener for configuration changes."""

    def __init__(self, callback: Callable[[str, ConfigChangeType, Any], None]):
        self.callback = callback
        self.subscribed_keys: Set[str] = set()

    def subscribe(self, key: str):
        """Subscribe to changes for a specific config key."""
        self.subscribed_keys.add(key)

    def subscribe_all(self):
        """Subscribe to all configuration changes."""
        self.subscribed_keys.add("*")

    def notify(self, key: str, change_type: ConfigChangeType, value: Any):
        """Notify listener of a change."""
        if "*" in self.subscribed_keys or key in self.subscribed_keys:
            try:
                self.callback(key, change_type, value)
            except Exception as e:
                logger.error(f"Config change listener error: {e}")


class AutonomyConfigManager:
    """
    Enhanced Configuration Manager for Autonomous Operation.

    Features:
    - Thread-safe configuration access
    - Hot reload with file watching
    - Change notification system
    - Configuration versioning
    - Validation and safety checks
    """

    def __init__(self, config_file: Optional[str] = None):
        self._config: AutonomyConfig = AutonomyConfig()
        self._config_file = Path(config_file) if config_file else None
        self._config_hash: Optional[str] = None
        self._listeners: List[ConfigChangeListener] = []
        self._lock = threading.RLock()
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        self._last_loaded: Optional[datetime] = None
        self._change_history: List[Dict[str, Any]] = []

        # Load from file if specified
        if self._config_file and self._config_file.exists():
            self._load_from_file()

        logger.info("AutonomyConfigManager initialized")

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute hash of configuration data."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _load_from_file(self):
        """Load configuration from JSON file."""
        if not self._config_file or not self._config_file.exists():
            return

        try:
            with open(self._config_file, 'r') as f:
                data = json.load(f)

            # Apply loaded values to config
            self._apply_config_data(data)
            self._config_hash = self._compute_hash(data)
            self._last_loaded = datetime.now()

            logger.info(f"Configuration loaded from {self._config_file}")
            self._notify_listeners("*", ConfigChangeType.RELOADED, self._config.to_dict())

        except Exception as e:
            logger.error(f"Failed to load config from file: {e}")

    def _apply_config_data(self, data: Dict[str, Any]):
        """Apply configuration data to config object."""
        with self._lock:
            # Apply resource budget
            if "resource_budget" in data:
                for key, value in data["resource_budget"].items():
                    if hasattr(self._config.resource_budget, key):
                        setattr(self._config.resource_budget, key, value)

            # Apply autonomy dials
            if "autonomy_dials" in data:
                for dial_name, dial_data in data["autonomy_dials"].items():
                    dial = self._config.autonomy_dials.get_all_dials().get(dial_name)
                    if dial and isinstance(dial_data, dict):
                        if "current_level" in dial_data:
                            dial.current_level = dial_data["current_level"]

            # Apply learning config
            if "learning" in data:
                for key, value in data["learning"].items():
                    if hasattr(self._config.learning, key):
                        setattr(self._config.learning, key, value)

            # Apply top-level settings
            for key in ["autonomous_mode_enabled", "verbose_logging", "environment"]:
                if key in data:
                    setattr(self._config, key, data[key])

    def save_to_file(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = Path(path) if path else self._config_file
        if not save_path:
            save_path = Path(__file__).parent / "autonomy_config.json"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = self._config.to_dict()
            data["_saved_at"] = datetime.now().isoformat()
            data["_version"] = self._config.version

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)

            self._config_hash = self._compute_hash(data)

        logger.info(f"Configuration saved to {save_path}")

    def start_watching(self, interval: float = 5.0):
        """Start watching configuration file for changes."""
        if not self._config_file:
            logger.warning("No config file specified, cannot start watching")
            return

        if self._watch_thread and self._watch_thread.is_alive():
            return

        self._stop_watching.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
            name="config-watcher"
        )
        self._watch_thread.start()
        logger.info(f"Started watching config file: {self._config_file}")

    def stop_watching(self):
        """Stop watching configuration file."""
        self._stop_watching.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2)

    def _watch_loop(self, interval: float):
        """Background loop to watch for config changes."""
        while not self._stop_watching.wait(timeout=interval):
            if self._config_file and self._config_file.exists():
                try:
                    with open(self._config_file, 'r') as f:
                        data = json.load(f)

                    new_hash = self._compute_hash(data)
                    if new_hash != self._config_hash:
                        logger.info("Configuration file changed, reloading...")
                        self._load_from_file()
                except Exception as e:
                    logger.error(f"Error checking config file: {e}")

    def add_listener(self, listener: ConfigChangeListener):
        """Add a configuration change listener."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: ConfigChangeListener):
        """Remove a configuration change listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify_listeners(self, key: str, change_type: ConfigChangeType, value: Any):
        """Notify all listeners of a configuration change."""
        # Record change history
        self._change_history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "change_type": change_type.value,
        })
        # Keep last 100 changes
        self._change_history = self._change_history[-100:]

        for listener in self._listeners:
            listener.notify(key, change_type, value)

    @property
    def config(self) -> AutonomyConfig:
        """Get the current configuration."""
        with self._lock:
            return self._config

    def get_resource_budget(self) -> ResourceBudget:
        """Get resource budget configuration."""
        with self._lock:
            return self._config.resource_budget

    def get_autonomy_dials(self) -> AutonomyDialsConfig:
        """Get autonomy dials configuration."""
        with self._lock:
            return self._config.autonomy_dials

    def get_safety_constraints(self) -> SafetyConstraintsConfig:
        """Get safety constraints configuration."""
        with self._lock:
            return self._config.safety_constraints

    def get_kill_switch_config(self) -> KillSwitchConfig:
        """Get kill switch configuration."""
        with self._lock:
            return self._config.kill_switch

    def get_communication_config(self) -> CommunicationConfig:
        """Get communication configuration."""
        with self._lock:
            return self._config.communication

    def get_learning_config(self) -> LearningConfig:
        """Get learning configuration."""
        with self._lock:
            return self._config.learning

    def set_autonomy_dial(self, dial_name: str, level: float) -> bool:
        """Set a specific autonomy dial level."""
        with self._lock:
            dials = self._config.autonomy_dials.get_all_dials()
            if dial_name not in dials:
                logger.error(f"Unknown autonomy dial: {dial_name}")
                return False

            dial = dials[dial_name]
            old_level = dial.current_level
            dial.current_level = max(dial.min_level, min(dial.max_level, level))

            if dial.current_level != old_level:
                self._notify_listeners(
                    f"autonomy_dials.{dial_name}",
                    ConfigChangeType.UPDATED,
                    {"old": old_level, "new": dial.current_level}
                )
                logger.info(f"Autonomy dial '{dial_name}' changed: {old_level:.2f} -> {dial.current_level:.2f}")

            return True

    def emergency_lockdown(self):
        """Trigger emergency lockdown - set all dials to minimum."""
        with self._lock:
            self._config.autonomy_dials.emergency_lockdown()
            self._notify_listeners("autonomy_dials", ConfigChangeType.UPDATED, "EMERGENCY_LOCKDOWN")
            logger.warning("EMERGENCY LOCKDOWN: All autonomy dials set to minimum")

    def is_autonomous_mode_enabled(self) -> bool:
        """Check if autonomous mode is enabled."""
        with self._lock:
            return self._config.autonomous_mode_enabled

    def set_autonomous_mode(self, enabled: bool):
        """Enable or disable autonomous mode."""
        with self._lock:
            old_value = self._config.autonomous_mode_enabled
            self._config.autonomous_mode_enabled = enabled

            if old_value != enabled:
                self._notify_listeners(
                    "autonomous_mode_enabled",
                    ConfigChangeType.UPDATED,
                    enabled
                )
                logger.info(f"Autonomous mode: {enabled}")

    def validate(self) -> List[str]:
        """Validate current configuration."""
        with self._lock:
            return self._config.validate()

    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status."""
        with self._lock:
            return {
                "config_file": str(self._config_file) if self._config_file else None,
                "config_hash": self._config_hash,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                "watching": self._watch_thread is not None and self._watch_thread.is_alive(),
                "listener_count": len(self._listeners),
                "change_history_count": len(self._change_history),
                "validation_issues": self.validate(),
            }

    def export_config(self) -> Dict[str, Any]:
        """Export full configuration as dictionary."""
        with self._lock:
            return self._config.to_dict()

    @contextmanager
    def temporary_config(self, **overrides):
        """Context manager for temporary configuration changes."""
        with self._lock:
            # Store original values
            original_values = {}
            for key, value in overrides.items():
                if hasattr(self._config, key):
                    original_values[key] = getattr(self._config, key)
                    setattr(self._config, key, value)

            try:
                yield self._config
            finally:
                # Restore original values
                for key, value in original_values.items():
                    setattr(self._config, key, value)


# Global instance
_config_manager: Optional[AutonomyConfigManager] = None
_config_manager_lock = threading.Lock()


def get_config_manager(config_file: Optional[str] = None) -> AutonomyConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    with _config_manager_lock:
        if _config_manager is None:
            _config_manager = AutonomyConfigManager(config_file)
        return _config_manager


def init_config_manager(config_file: Optional[str] = None, watch: bool = False) -> AutonomyConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    with _config_manager_lock:
        _config_manager = AutonomyConfigManager(config_file)
        if watch and config_file:
            _config_manager.start_watching()
        return _config_manager


# Convenience accessors
def get_resource_budget() -> ResourceBudget:
    return get_config_manager().get_resource_budget()


def get_autonomy_dials() -> AutonomyDialsConfig:
    return get_config_manager().get_autonomy_dials()


def get_safety_constraints() -> SafetyConstraintsConfig:
    return get_config_manager().get_safety_constraints()


def get_kill_switch_config() -> KillSwitchConfig:
    return get_config_manager().get_kill_switch_config()


def get_communication_config() -> CommunicationConfig:
    return get_config_manager().get_communication_config()


def get_learning_config() -> LearningConfig:
    return get_config_manager().get_learning_config()


def is_autonomous_mode() -> bool:
    return get_config_manager().is_autonomous_mode_enabled()

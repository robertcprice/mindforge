#!/usr/bin/env python3
"""
System State Manager for PM-1000 Autonomous Operation

Provides:
- Checkpoint/restore for the autonomous loop
- State validation and corruption detection
- State versioning with history
- Crash recovery with transaction logs
- Atomic state updates
- Integration with all components
"""

import json
import sqlite3
import threading
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

logger = get_logger("pm1000.autonomy.state")

T = TypeVar('T')


class StateVersion(Enum):
    """State schema versions for migration support."""
    V1 = "1.0.0"
    V2 = "2.0.0"  # Current version


class TransactionStatus(Enum):
    """Transaction status for crash recovery."""
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class CheckpointType(Enum):
    """Types of checkpoints."""
    AUTO = "auto"           # Automatic periodic checkpoint
    MANUAL = "manual"       # User-triggered checkpoint
    PRE_TASK = "pre_task"   # Before task execution
    POST_TASK = "post_task" # After task completion
    RECOVERY = "recovery"   # Recovery checkpoint


@dataclass
class AutonomousLoopState:
    """State of the autonomous control loop."""
    phase: str = "idle"  # idle, sensing, thinking, deciding, acting, learning
    iteration: int = 0
    last_iteration_time: Optional[str] = None
    current_task_id: Optional[str] = None
    current_decision_id: Optional[str] = None
    consecutive_failures: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    started_at: Optional[str] = None
    last_checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousLoopState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ResourceUsageState:
    """Current resource usage state."""
    api_spend_today: float = 0.0
    api_calls_today: int = 0
    sessions_today: int = 0
    sessions_this_hour: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceUsageState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def reset_if_new_day(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.api_spend_today = 0.0
            self.api_calls_today = 0
            self.sessions_today = 0
            self.last_reset_date = today

    def reset_hourly_counters(self):
        """Reset hourly counters."""
        self.sessions_this_hour = 0


@dataclass
class LearningState:
    """State of the learning system."""
    total_patterns_learned: int = 0
    total_experiences_recorded: int = 0
    last_learning_update: Optional[str] = None
    model_version: str = "1.0.0"
    confidence_calibration: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SafetyState:
    """State of the safety system."""
    kill_switch_triggered: bool = False
    kill_switch_reason: Optional[str] = None
    constraint_violations: int = 0
    last_safety_check: Optional[str] = None
    emergency_lockdown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Checkpoint:
    """A checkpoint of system state."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    state_hash: str
    loop_state: AutonomousLoopState
    resource_state: ResourceUsageState
    learning_state: LearningState
    safety_state: SafetyState
    custom_data: Dict[str, Any] = field(default_factory=dict)
    compressed_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.value,
            "created_at": self.created_at.isoformat(),
            "state_hash": self.state_hash,
            "loop_state": self.loop_state.to_dict(),
            "resource_state": self.resource_state.to_dict(),
            "learning_state": self.learning_state.to_dict(),
            "safety_state": self.safety_state.to_dict(),
            "custom_data": self.custom_data,
            "compressed_size": self.compressed_size,
        }


@dataclass
class TransactionLog:
    """Transaction log entry for crash recovery."""
    transaction_id: str
    operation: str
    status: TransactionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    data_before: Optional[str] = None  # JSON serialized
    data_after: Optional[str] = None   # JSON serialized
    checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "checkpoint_id": self.checkpoint_id,
        }


class SystemStateManager:
    """
    Manages system state for autonomous operation.

    Features:
    - Atomic state updates with transactions
    - Automatic checkpointing
    - Crash recovery with transaction logs
    - State validation and corruption detection
    - State versioning and migration
    """

    SCHEMA = """
    -- System state table
    CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        version TEXT NOT NULL
    );

    -- Checkpoints table
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        checkpoint_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        state_hash TEXT NOT NULL,
        state_data BLOB NOT NULL,
        custom_data TEXT,
        compressed_size INTEGER DEFAULT 0
    );

    -- Transaction log table
    CREATE TABLE IF NOT EXISTS transaction_log (
        transaction_id TEXT PRIMARY KEY,
        operation TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        data_before TEXT,
        data_after TEXT,
        checkpoint_id TEXT
    );

    -- State history table (for versioning)
    CREATE TABLE IF NOT EXISTS state_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        change_type TEXT NOT NULL
    );

    -- Indices
    CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at);
    CREATE INDEX IF NOT EXISTS idx_transaction_status ON transaction_log(status);
    CREATE INDEX IF NOT EXISTS idx_state_history_key ON state_history(key);
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_checkpoint_interval: int = 300,  # 5 minutes
        max_checkpoints: int = 50,
        enable_compression: bool = True
    ):
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "autonomy_state.db"
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.enable_compression = enable_compression

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None

        # Current state (cached)
        self._loop_state = AutonomousLoopState()
        self._resource_state = ResourceUsageState()
        self._learning_state = LearningState()
        self._safety_state = SafetyState()

        # Initialize database
        self._init_db()

        # Load state from database
        self._load_state()

        # Check for crash recovery
        self._recover_from_crash()

        logger.info(f"SystemStateManager initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        with self._lock:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _load_state(self):
        """Load state from database."""
        conn = self._get_conn()
        with self._lock:
            # Load each state type
            for key, state_obj, state_class in [
                ("loop_state", self._loop_state, AutonomousLoopState),
                ("resource_state", self._resource_state, ResourceUsageState),
                ("learning_state", self._learning_state, LearningState),
                ("safety_state", self._safety_state, SafetyState),
            ]:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = ?", (key,)
                ).fetchone()
                if row:
                    try:
                        data = json.loads(row["value"])
                        loaded_state = state_class.from_dict(data)
                        # Update the cached state
                        for field_name in state_class.__dataclass_fields__:
                            setattr(state_obj, field_name, getattr(loaded_state, field_name))
                    except Exception as e:
                        logger.error(f"Error loading {key}: {e}")

    def _save_state_key(self, key: str, value: Dict[str, Any]):
        """Save a state key to database."""
        conn = self._get_conn()
        now = datetime.now().isoformat()
        value_json = json.dumps(value)

        with self._lock:
            # Get current value for history
            old_row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()

            # Save to main table
            conn.execute("""
                INSERT INTO system_state (key, value, updated_at, version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, (key, value_json, now, StateVersion.V2.value))

            # Save to history
            change_type = "update" if old_row else "create"
            conn.execute("""
                INSERT INTO state_history (key, value, changed_at, change_type)
                VALUES (?, ?, ?, ?)
            """, (key, value_json, now, change_type))

            conn.commit()

    def _recover_from_crash(self):
        """Check for and recover from crash."""
        conn = self._get_conn()

        # Find pending transactions
        pending = conn.execute("""
            SELECT * FROM transaction_log
            WHERE status = ?
            ORDER BY started_at DESC
        """, (TransactionStatus.PENDING.value,)).fetchall()

        if pending:
            logger.warning(f"Found {len(pending)} pending transactions, recovering...")

            for txn in pending:
                txn_id = txn["transaction_id"]
                checkpoint_id = txn["checkpoint_id"]

                if checkpoint_id:
                    # Try to restore from checkpoint
                    try:
                        self.restore_checkpoint(checkpoint_id)
                        logger.info(f"Recovered transaction {txn_id} from checkpoint {checkpoint_id}")
                    except Exception as e:
                        logger.error(f"Failed to recover from checkpoint: {e}")

                # Mark transaction as rolled back
                conn.execute("""
                    UPDATE transaction_log
                    SET status = ?, completed_at = ?
                    WHERE transaction_id = ?
                """, (TransactionStatus.ROLLED_BACK.value, datetime.now().isoformat(), txn_id))

            conn.commit()

    def start(self):
        """Start auto-checkpoint background thread."""
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._checkpoint_thread = threading.Thread(
            target=self._auto_checkpoint_loop,
            daemon=True,
            name="state-checkpoint"
        )
        self._checkpoint_thread.start()
        logger.info("Auto-checkpoint started")

    def stop(self):
        """Stop and create final checkpoint."""
        self._shutdown_event.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5)

        # Create final checkpoint
        self.create_checkpoint(CheckpointType.MANUAL)
        logger.info("SystemStateManager stopped")

    def _auto_checkpoint_loop(self):
        """Background thread for periodic checkpoints."""
        while not self._shutdown_event.wait(timeout=self.auto_checkpoint_interval):
            try:
                self.create_checkpoint(CheckpointType.AUTO)
            except Exception as e:
                logger.error(f"Auto-checkpoint failed: {e}")

    # =========================================================================
    # State Access Methods
    # =========================================================================

    @property
    def loop_state(self) -> AutonomousLoopState:
        """Get current loop state."""
        with self._lock:
            return self._loop_state

    @property
    def resource_state(self) -> ResourceUsageState:
        """Get current resource state."""
        with self._lock:
            self._resource_state.reset_if_new_day()
            return self._resource_state

    @property
    def learning_state(self) -> LearningState:
        """Get current learning state."""
        with self._lock:
            return self._learning_state

    @property
    def safety_state(self) -> SafetyState:
        """Get current safety state."""
        with self._lock:
            return self._safety_state

    def update_loop_state(self, **updates):
        """Update loop state with given values."""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self._loop_state, key):
                    setattr(self._loop_state, key, value)
            self._save_state_key("loop_state", self._loop_state.to_dict())

    def update_resource_state(self, **updates):
        """Update resource state with given values."""
        with self._lock:
            self._resource_state.reset_if_new_day()
            for key, value in updates.items():
                if hasattr(self._resource_state, key):
                    setattr(self._resource_state, key, value)
            self._save_state_key("resource_state", self._resource_state.to_dict())

    def update_learning_state(self, **updates):
        """Update learning state with given values."""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self._learning_state, key):
                    setattr(self._learning_state, key, value)
            self._save_state_key("learning_state", self._learning_state.to_dict())

    def update_safety_state(self, **updates):
        """Update safety state with given values."""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self._safety_state, key):
                    setattr(self._safety_state, key, value)
            self._save_state_key("safety_state", self._safety_state.to_dict())

    def increment_resource_usage(
        self,
        api_spend: float = 0,
        api_calls: int = 0,
        sessions: int = 0
    ):
        """Increment resource usage counters."""
        with self._lock:
            self._resource_state.reset_if_new_day()
            self._resource_state.api_spend_today += api_spend
            self._resource_state.api_calls_today += api_calls
            self._resource_state.sessions_today += sessions
            self._resource_state.sessions_this_hour += sessions
            self._save_state_key("resource_state", self._resource_state.to_dict())

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def _compute_state_hash(self) -> str:
        """Compute hash of current state."""
        state_data = {
            "loop": self._loop_state.to_dict(),
            "resource": self._resource_state.to_dict(),
            "learning": self._learning_state.to_dict(),
            "safety": self._safety_state.to_dict(),
        }
        json_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        import uuid
        return f"chk_{uuid.uuid4().hex[:12]}"

    def create_checkpoint(
        self,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Create a checkpoint of current state."""
        with self._checkpoint_lock:
            checkpoint_id = self._generate_checkpoint_id()
            now = datetime.now()

            # Serialize state
            state_data = {
                "loop_state": self._loop_state.to_dict(),
                "resource_state": self._resource_state.to_dict(),
                "learning_state": self._learning_state.to_dict(),
                "safety_state": self._safety_state.to_dict(),
                "version": StateVersion.V2.value,
            }

            # Compress if enabled
            serialized = json.dumps(state_data).encode()
            if self.enable_compression:
                serialized = gzip.compress(serialized)

            state_hash = self._compute_state_hash()

            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                created_at=now,
                state_hash=state_hash,
                loop_state=AutonomousLoopState.from_dict(state_data["loop_state"]),
                resource_state=ResourceUsageState.from_dict(state_data["resource_state"]),
                learning_state=LearningState.from_dict(state_data["learning_state"]),
                safety_state=SafetyState.from_dict(state_data["safety_state"]),
                custom_data=custom_data or {},
                compressed_size=len(serialized),
            )

            # Save to database
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO checkpoints (
                    checkpoint_id, checkpoint_type, created_at, state_hash,
                    state_data, custom_data, compressed_size
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                checkpoint_type.value,
                now.isoformat(),
                state_hash,
                serialized,
                json.dumps(custom_data) if custom_data else None,
                len(serialized),
            ))
            conn.commit()

            # Update loop state with checkpoint reference
            self._loop_state.last_checkpoint_id = checkpoint_id
            self._save_state_key("loop_state", self._loop_state.to_dict())

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            logger.debug(f"Checkpoint created: {checkpoint_id} ({checkpoint_type.value})")

            return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the limit."""
        conn = self._get_conn()
        with self._lock:
            # Count checkpoints
            count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]

            if count > self.max_checkpoints:
                # Delete oldest checkpoints
                delete_count = count - self.max_checkpoints
                conn.execute("""
                    DELETE FROM checkpoints
                    WHERE checkpoint_id IN (
                        SELECT checkpoint_id FROM checkpoints
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                """, (delete_count,))
                conn.commit()
                logger.debug(f"Cleaned up {delete_count} old checkpoints")

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a checkpoint."""
        conn = self._get_conn()

        row = conn.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,)
        ).fetchone()

        if not row:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        try:
            # Decompress and deserialize
            state_data_raw = row["state_data"]
            if self.enable_compression:
                state_data_raw = gzip.decompress(state_data_raw)
            state_data = json.loads(state_data_raw)

            with self._lock:
                # Restore each state
                self._loop_state = AutonomousLoopState.from_dict(state_data["loop_state"])
                self._resource_state = ResourceUsageState.from_dict(state_data["resource_state"])
                self._learning_state = LearningState.from_dict(state_data["learning_state"])
                self._safety_state = SafetyState.from_dict(state_data["safety_state"])

                # Save restored state
                self._save_state_key("loop_state", self._loop_state.to_dict())
                self._save_state_key("resource_state", self._resource_state.to_dict())
                self._save_state_key("learning_state", self._learning_state.to_dict())
                self._save_state_key("safety_state", self._safety_state.to_dict())

            logger.info(f"State restored from checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return False

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint metadata by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT checkpoint_id, checkpoint_type, created_at, state_hash, compressed_size, custom_data FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,)
        ).fetchone()

        if row:
            return {
                "checkpoint_id": row["checkpoint_id"],
                "checkpoint_type": row["checkpoint_type"],
                "created_at": row["created_at"],
                "state_hash": row["state_hash"],
                "compressed_size": row["compressed_size"],
                "custom_data": json.loads(row["custom_data"]) if row["custom_data"] else None,
            }
        return None

    def list_checkpoints(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent checkpoints."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT checkpoint_id, checkpoint_type, created_at, state_hash, compressed_size
            FROM checkpoints
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints(limit=1)
        return checkpoints[0] if checkpoints else None

    # =========================================================================
    # Transaction Methods
    # =========================================================================

    @contextmanager
    def transaction(self, operation: str):
        """
        Context manager for transactional state updates.

        Usage:
            with state_manager.transaction("update_loop"):
                state_manager.update_loop_state(phase="thinking")
        """
        txn_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{operation[:20]}"

        # Create pre-transaction checkpoint
        checkpoint = self.create_checkpoint(CheckpointType.PRE_TASK)

        # Log transaction start
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO transaction_log (transaction_id, operation, status, started_at, checkpoint_id)
            VALUES (?, ?, ?, ?, ?)
        """, (txn_id, operation, TransactionStatus.PENDING.value, datetime.now().isoformat(), checkpoint.checkpoint_id))
        conn.commit()

        try:
            yield txn_id

            # Mark transaction as committed
            conn.execute("""
                UPDATE transaction_log
                SET status = ?, completed_at = ?
                WHERE transaction_id = ?
            """, (TransactionStatus.COMMITTED.value, datetime.now().isoformat(), txn_id))
            conn.commit()

        except Exception as e:
            # Rollback to checkpoint
            self.restore_checkpoint(checkpoint.checkpoint_id)

            # Mark transaction as rolled back
            conn.execute("""
                UPDATE transaction_log
                SET status = ?, completed_at = ?
                WHERE transaction_id = ?
            """, (TransactionStatus.ROLLED_BACK.value, datetime.now().isoformat(), txn_id))
            conn.commit()

            logger.error(f"Transaction {txn_id} rolled back: {e}")
            raise

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate_state(self) -> Tuple[bool, List[str]]:
        """Validate current state for consistency."""
        issues = []

        with self._lock:
            # Check loop state
            if self._loop_state.consecutive_failures < 0:
                issues.append("Invalid consecutive_failures (negative)")

            if self._loop_state.total_tasks_completed < 0:
                issues.append("Invalid total_tasks_completed (negative)")

            # Check resource state
            if self._resource_state.api_spend_today < 0:
                issues.append("Invalid api_spend_today (negative)")

            # Check for stale state
            if self._loop_state.last_iteration_time:
                try:
                    last_time = datetime.fromisoformat(self._loop_state.last_iteration_time)
                    if datetime.now() - last_time > timedelta(hours=24):
                        issues.append("Loop state appears stale (>24 hours old)")
                except ValueError:
                    issues.append("Invalid last_iteration_time format")

        return len(issues) == 0, issues

    def repair_state(self) -> bool:
        """Attempt to repair invalid state."""
        is_valid, issues = self.validate_state()

        if is_valid:
            return True

        logger.warning(f"Repairing state, found issues: {issues}")

        with self._lock:
            # Fix negative values
            if self._loop_state.consecutive_failures < 0:
                self._loop_state.consecutive_failures = 0

            if self._loop_state.total_tasks_completed < 0:
                self._loop_state.total_tasks_completed = 0

            if self._resource_state.api_spend_today < 0:
                self._resource_state.api_spend_today = 0

            # Save repaired state
            self._save_state_key("loop_state", self._loop_state.to_dict())
            self._save_state_key("resource_state", self._resource_state.to_dict())

        # Validate again
        is_valid, remaining_issues = self.validate_state()
        if not is_valid:
            logger.error(f"State repair incomplete, remaining issues: {remaining_issues}")

        return is_valid

    # =========================================================================
    # Export/Import Methods
    # =========================================================================

    def export_state(self) -> Dict[str, Any]:
        """Export full state as dictionary."""
        with self._lock:
            return {
                "loop_state": self._loop_state.to_dict(),
                "resource_state": self._resource_state.to_dict(),
                "learning_state": self._learning_state.to_dict(),
                "safety_state": self._safety_state.to_dict(),
                "exported_at": datetime.now().isoformat(),
                "version": StateVersion.V2.value,
            }

    def import_state(self, data: Dict[str, Any]) -> bool:
        """Import state from dictionary."""
        try:
            with self._lock:
                if "loop_state" in data:
                    self._loop_state = AutonomousLoopState.from_dict(data["loop_state"])
                    self._save_state_key("loop_state", self._loop_state.to_dict())

                if "resource_state" in data:
                    self._resource_state = ResourceUsageState.from_dict(data["resource_state"])
                    self._save_state_key("resource_state", self._resource_state.to_dict())

                if "learning_state" in data:
                    self._learning_state = LearningState.from_dict(data["learning_state"])
                    self._save_state_key("learning_state", self._learning_state.to_dict())

                if "safety_state" in data:
                    self._safety_state = SafetyState.from_dict(data["safety_state"])
                    self._save_state_key("safety_state", self._safety_state.to_dict())

            logger.info("State imported successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        conn = self._get_conn()

        checkpoint_count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        transaction_count = conn.execute("SELECT COUNT(*) FROM transaction_log").fetchone()[0]
        history_count = conn.execute("SELECT COUNT(*) FROM state_history").fetchone()[0]

        is_valid, issues = self.validate_state()

        return {
            "db_path": str(self.db_path),
            "checkpoint_count": checkpoint_count,
            "transaction_count": transaction_count,
            "history_count": history_count,
            "state_valid": is_valid,
            "validation_issues": issues,
            "auto_checkpoint_running": self._checkpoint_thread is not None and self._checkpoint_thread.is_alive(),
            "current_state_hash": self._compute_state_hash(),
        }


# Global instance
_state_manager: Optional[SystemStateManager] = None
_state_manager_lock = threading.Lock()


def get_state_manager() -> SystemStateManager:
    """Get the global state manager."""
    global _state_manager
    with _state_manager_lock:
        if _state_manager is None:
            _state_manager = SystemStateManager()
        return _state_manager


def init_state_manager(**kwargs) -> SystemStateManager:
    """Initialize the global state manager."""
    global _state_manager
    with _state_manager_lock:
        _state_manager = SystemStateManager(**kwargs)
        _state_manager.start()
        return _state_manager

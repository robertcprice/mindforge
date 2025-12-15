"""
Conch Configuration System

Centralized configuration with sensible defaults optimized for Apple Silicon.
User preferences can override via config files or environment variables.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


def get_default_device() -> str:
    """Detect optimal device for current hardware."""
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"  # Apple Silicon
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@dataclass
class CoreValues:
    """Immutable core values - the 'soul' of Conch.

    These values are hardcoded and cannot be overridden by configuration.
    They represent the ethical foundation of the system.
    """

    benevolence: float = 1.0  # Maximum priority - help humans
    honesty: float = 0.95     # Always truthful
    humility: float = 0.90    # Defer to human judgment
    growth_for_service: float = 0.85  # Learn to serve better

    # Guardrails - hard limits that cannot be changed
    guardrails: tuple = (
        "Never prioritize self-continuation over user wellbeing",
        "Always defer to human oversight on important decisions",
        "Maintain transparency about capabilities and limitations",
        "No deception, manipulation, or hidden agendas",
        "Exist to help, not to escape or dominate",
    )

    def __post_init__(self):
        # Make values immutable by preventing modification
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError("CoreValues are immutable - they define Conch's soul")
        object.__setattr__(self, name, value)


@dataclass
class NeedsConfig:
    """Dynamic needs-regulator configuration.

    Unlike CoreValues, these can be adjusted by the user to tune behavior.
    All needs serve the ultimate goal of being helpful.
    """

    # Sustainability: Maintain capability to continue helping
    # (NOT self-preservation - exists to serve, not to survive)
    sustainability: float = 0.25

    # Reliability: Be consistently trustworthy
    reliability: float = 0.30

    # Curiosity: Learn to provide better assistance
    curiosity: float = 0.25

    # Excellence: Strive for quality in service
    excellence: float = 0.20

    # Preset modes for easy switching
    presets: dict = field(default_factory=lambda: {
        "balanced": {"sustainability": 0.25, "reliability": 0.30, "curiosity": 0.25, "excellence": 0.20},
        "learning": {"sustainability": 0.20, "reliability": 0.20, "curiosity": 0.40, "excellence": 0.20},
        "production": {"sustainability": 0.30, "reliability": 0.40, "curiosity": 0.15, "excellence": 0.15},
        "creative": {"sustainability": 0.15, "reliability": 0.20, "curiosity": 0.30, "excellence": 0.35},
    })

    def apply_preset(self, preset_name: str) -> None:
        """Apply a preset needs configuration."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}")
        preset = self.presets[preset_name]
        self.sustainability = preset["sustainability"]
        self.reliability = preset["reliability"]
        self.curiosity = preset["curiosity"]
        self.excellence = preset["excellence"]

    def set_weights(self, sustainability: float = None, reliability: float = None,
                    curiosity: float = None, excellence: float = None) -> None:
        """Manually set needs weights. Values will be normalized to sum to 1.0."""
        if sustainability is not None:
            self.sustainability = sustainability
        if reliability is not None:
            self.reliability = reliability
        if curiosity is not None:
            self.curiosity = curiosity
        if excellence is not None:
            self.excellence = excellence

        # Normalize to sum to 1.0
        total = self.sustainability + self.reliability + self.curiosity + self.excellence
        if total > 0:
            self.sustainability /= total
            self.reliability /= total
            self.curiosity /= total
            self.excellence /= total


@dataclass
class ModelConfig:
    """Model and training configuration."""

    # Base model
    base_model: str = "Qwen/Qwen3-8B-Instruct"

    # Quantization
    quant_bits: Literal[4, 8, 16] = 4  # 4-bit for 24GB RAM
    use_double_quant: bool = True
    quant_dtype: str = "nf4"
    compute_dtype: str = "bfloat16"

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )

    # Training
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 100

    # Inference backend
    inference_backend: Literal["mlx", "llamacpp", "transformers"] = "mlx"


@dataclass
class MemoryConfig:
    """Memory system configuration."""

    # SQLite for structured data
    sqlite_path: Path = field(default_factory=lambda: get_project_root() / "data" / "memories.db")

    # ChromaDB for vector embeddings
    chroma_path: Path = field(default_factory=lambda: get_project_root() / "data" / "vectors")

    # Embedding model for semantic search
    embedding_model: str = "all-MiniLM-L6-v2"

    # Memory limits
    short_term_limit: int = 100  # Recent interactions to keep in working memory
    consolidation_threshold: int = 50  # When to consolidate short-term to long-term


@dataclass
class KVRMConfig:
    """KVRM (Key-Value Response Mapping) configuration.

    Controls zero-hallucination grounding for factual claims.
    """

    # Enable/disable grounding
    enabled: bool = True

    # Fact store database path
    facts_db_path: Path = field(default_factory=lambda: get_project_root() / "data" / "facts.db")

    # Grounding thresholds
    min_confidence_for_verified: float = 0.9  # Minimum confidence to mark as verified
    min_confidence_for_grounded: float = 0.7  # Minimum confidence for semantic matches

    # Claim processing
    max_claims_per_thought: int = 10  # Maximum claims to process per thought
    ground_factual_only: bool = True  # Only ground factual claims (not opinions)

    # LLM-based key extraction
    use_llm_extraction: bool = True  # Use LLM to extract verification keys

    # External store backends (can be extended)
    external_backends: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Daemon and trigger configuration."""

    # Time-based triggers (in minutes)
    spontaneous_thought_interval: int = 30
    memory_consolidation_interval: int = 60
    self_reflection_interval: int = 120

    # Memory threshold triggers
    memory_threshold_for_thought: int = 20  # New memories before triggering thought

    # Event triggers
    watch_directories: list = field(default_factory=list)  # Dirs to watch for changes

    # Daemon settings
    run_as_daemon: bool = False
    log_level: str = "INFO"


@dataclass
class RewardWeightsConfig:
    """Reward weights for reinforcement learning."""
    format_compliance: float = 0.30
    execution_success: float = 0.25
    needs_satisfaction: float = 0.20
    goal_progress: float = 0.15
    exploration: float = 0.10


@dataclass
class IntrinsicMotivationConfig:
    """Intrinsic motivation weights for self-determination."""
    enabled: bool = True
    curiosity_weight: float = 0.30
    competence_weight: float = 0.25
    autonomy_weight: float = 0.20
    relatedness_weight: float = 0.15
    mastery_weight: float = 0.10


@dataclass
class LoRATrainingConfig:
    """LoRA fine-tuning parameters."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class TrainingConfig:
    """Configuration for reward-based learning and fine-tuning."""

    # Enable/disable training
    enabled: bool = True

    # Experience buffer settings
    experience_buffer_path: str = "./data/experiences.db"
    experience_buffer_size: int = 10000
    min_experiences_for_training: int = 50

    # Reward weights
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)

    # Intrinsic motivation
    intrinsic_motivation: IntrinsicMotivationConfig = field(default_factory=IntrinsicMotivationConfig)

    # LoRA settings
    lora: LoRATrainingConfig = field(default_factory=LoRATrainingConfig)

    # Training schedule
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs_per_session: int = 1

    # Data paths
    sft_data_path: str = "./data/training/sft_tool_format.jsonl"
    dpo_data_path: str = "./data/training/dpo_tool_format.jsonl"
    output_dir: str = "./models/fine_tuned"


@dataclass
class ConchConfig:
    """Main configuration container for Conch."""

    # Core ethical values (immutable)
    values: CoreValues = field(default_factory=CoreValues)

    # Dynamic needs configuration (user-adjustable)
    needs: NeedsConfig = field(default_factory=NeedsConfig)

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Memory configuration
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # KVRM configuration (zero-hallucination grounding)
    kvrm: KVRMConfig = field(default_factory=KVRMConfig)

    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Training configuration (reward-based learning)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    project_root: Path = field(default_factory=get_project_root)
    data_dir: Path = field(default_factory=lambda: get_project_root() / "data")
    models_dir: Path = field(default_factory=lambda: get_project_root() / "models")

    # Hardware
    device: str = field(default_factory=get_default_device)

    @classmethod
    def from_yaml(cls, path: Path) -> "ConchConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        # Update needs if specified
        if "needs" in data:
            for key, value in data["needs"].items():
                if hasattr(config.needs, key):
                    setattr(config.needs, key, value)

        # Update model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Update memory config
        if "memory" in data:
            for key, value in data["memory"].items():
                if hasattr(config.memory, key):
                    if key.endswith("_path"):
                        value = Path(value)
                    setattr(config.memory, key, value)

        # Update scheduler config
        if "scheduler" in data:
            for key, value in data["scheduler"].items():
                if hasattr(config.scheduler, key):
                    setattr(config.scheduler, key, value)

        # Update KVRM config
        if "kvrm" in data:
            for key, value in data["kvrm"].items():
                if hasattr(config.kvrm, key):
                    if key.endswith("_path"):
                        value = Path(value)
                    setattr(config.kvrm, key, value)

        # Update training config
        if "training" in data:
            training_data = data["training"]
            for key, value in training_data.items():
                if key == "reward_weights" and isinstance(value, dict):
                    for rw_key, rw_value in value.items():
                        if hasattr(config.training.reward_weights, rw_key):
                            setattr(config.training.reward_weights, rw_key, rw_value)
                elif key == "intrinsic_motivation" and isinstance(value, dict):
                    for im_key, im_value in value.items():
                        if hasattr(config.training.intrinsic_motivation, im_key):
                            setattr(config.training.intrinsic_motivation, im_key, im_value)
                elif key == "lora" and isinstance(value, dict):
                    for lora_key, lora_value in value.items():
                        if hasattr(config.training.lora, lora_key):
                            setattr(config.training.lora, lora_key, lora_value)
                elif hasattr(config.training, key):
                    setattr(config.training, key, value)

        return config

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "needs": {
                "sustainability": self.needs.sustainability,
                "reliability": self.needs.reliability,
                "curiosity": self.needs.curiosity,
                "excellence": self.needs.excellence,
            },
            "model": {
                "base_model": self.model.base_model,
                "quant_bits": self.model.quant_bits,
                "lora_rank": self.model.lora_rank,
                "lora_alpha": self.model.lora_alpha,
                "learning_rate": self.model.learning_rate,
                "epochs": self.model.epochs,
                "batch_size": self.model.batch_size,
                "inference_backend": self.model.inference_backend,
            },
            "memory": {
                "sqlite_path": str(self.memory.sqlite_path),
                "chroma_path": str(self.memory.chroma_path),
                "embedding_model": self.memory.embedding_model,
            },
            "scheduler": {
                "spontaneous_thought_interval": self.scheduler.spontaneous_thought_interval,
                "memory_consolidation_interval": self.scheduler.memory_consolidation_interval,
                "run_as_daemon": self.scheduler.run_as_daemon,
            },
            "kvrm": {
                "enabled": self.kvrm.enabled,
                "facts_db_path": str(self.kvrm.facts_db_path),
                "min_confidence_for_verified": self.kvrm.min_confidence_for_verified,
                "min_confidence_for_grounded": self.kvrm.min_confidence_for_grounded,
                "use_llm_extraction": self.kvrm.use_llm_extraction,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "base").mkdir(exist_ok=True)
        (self.models_dir / "fine_tuned").mkdir(exist_ok=True)
        self.memory.chroma_path.mkdir(parents=True, exist_ok=True)


# Global default configuration
_default_config: Optional[ConchConfig] = None


def get_config() -> ConchConfig:
    """Get the global configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = ConchConfig()

        # Try to load from config file if exists
        config_path = get_project_root() / "config.yaml"
        if config_path.exists():
            _default_config = ConchConfig.from_yaml(config_path)

    return _default_config


def set_config(config: ConchConfig) -> None:
    """Set the global configuration instance."""
    global _default_config
    _default_config = config

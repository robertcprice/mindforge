"""
Conch DNA - TRUE Knowledge Distillation with Logit Matching

This module implements true knowledge distillation (KD) where the student model learns
from the teacher's output probability distributions (logits), not just the final outputs.

Key Concepts:
    - Teacher Model (EGO): Qwen3-8B - generates soft probability distributions
    - Student Model: Llama-3.2-3B or Qwen3-4B - learns to mimic teacher's distributions
    - Logit Matching: Student learns the relative probabilities, not just final answer
    - Temperature Scaling: Softens distributions to expose dark knowledge
    - Combined Loss: KL divergence + cross-entropy for robust learning

Architecture:
    Teacher (8B) → [Generate Logits with temp=T] → Soft Targets
                                                    ↓
    Student (3B-4B) → [Generate Logits with temp=T] → KL Div Loss + CE Loss
                                                    ↓
                                              LoRA Adapter Updates

Why TRUE KD > Simple Distillation:
    - Learns what the teacher finds "almost correct" (dark knowledge)
    - More robust than learning only from hard labels
    - Better generalization on out-of-distribution inputs
    - Captures uncertainty and relative confidence

Usage:
    # Initialize distiller
    distiller = TrueKnowledgeDistiller(
        teacher_model_id="mlx-community/Qwen3-8B-4bit",
        student_model_id="mlx-community/Qwen3-1.7B-4bit",
        config=KDConfig(temperature=2.0, alpha=0.7)
    )

    # Distill a domain
    result = distiller.distill_domain(
        domain="thinking",
        train_prompts=prompts_list
    )

    # Or distill all domains
    results = distiller.distill_all()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRALinear

logger = logging.getLogger(__name__)


@dataclass
class KDConfig:
    """Configuration for Knowledge Distillation training."""

    # Temperature for softening probability distributions
    # Higher temp = softer distributions = more "dark knowledge"
    temperature: float = 2.0

    # Alpha controls the balance between KD loss and CE loss
    # alpha=1.0: Pure KD, alpha=0.0: Pure CE, alpha=0.7: Typical best practice
    alpha: float = 0.7

    # LoRA configuration
    lora_rank: int = 16
    lora_scale: float = 1.0
    lora_dropout: float = 0.0

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_iters: int = 200
    warmup_iters: int = 20

    # Validation and checkpointing
    val_batches: int = 5
    steps_per_report: int = 10
    steps_per_eval: int = 50
    steps_per_save: int = 100

    # Output configuration
    output_dir: Path = Path("models/distilled_neurons")
    adapter_subdir: str = "adapters_kd"

    # Logging
    log_level: str = "INFO"
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.5 <= self.temperature <= 10.0:
            raise ValueError(f"Temperature {self.temperature} outside valid range [0.5, 10.0]")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha {self.alpha} outside valid range [0.0, 1.0]")

        if self.lora_rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {self.lora_rank}")

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class KDResult:
    """Result of knowledge distillation training."""

    domain: str
    num_samples: int
    adapter_path: Path
    final_kd_loss: float
    final_ce_loss: float
    final_combined_loss: float
    training_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_kd_loss(
    teacher_logits: mx.array,
    student_logits: mx.array,
    temperature: float = 2.0
) -> mx.array:
    """
    Compute KL divergence loss between teacher and student logit distributions.

    This is the core of knowledge distillation. The KL divergence measures how
    different the student's predicted distribution is from the teacher's.

    Args:
        teacher_logits: Teacher model logits, shape [batch, seq_len, vocab_size]
        student_logits: Student model logits, shape [batch, seq_len, vocab_size]
        temperature: Temperature for softening distributions

    Returns:
        Scalar KL divergence loss

    Mathematical formulation:
        KL(P||Q) = Σ P(x) * log(P(x) / Q(x))

        Where:
        - P = teacher distribution (soft targets)
        - Q = student distribution (predictions)
        - Temperature scaling softens both distributions
        - Scale by T² to account for temperature normalization
    """
    # Apply temperature scaling to soften distributions
    # Higher temperature → more uniform distribution → more "dark knowledge"
    teacher_soft = mx.softmax(teacher_logits / temperature, axis=-1)
    student_soft = mx.softmax(student_logits / temperature, axis=-1)

    # Add epsilon for numerical stability to BOTH log computations
    # This prevents log(0) = -inf which causes 0 * -inf = NaN
    epsilon = 1e-9
    teacher_log_soft = mx.log(teacher_soft + epsilon)
    student_log_soft = mx.log(student_soft + epsilon)

    # Compute KL divergence: KL(teacher || student)
    # This measures how much information is lost when we approximate
    # the teacher's distribution with the student's distribution
    kl_div = teacher_soft * (teacher_log_soft - student_log_soft)

    # Sum over vocabulary dimension, mean over batch and sequence
    kl_loss = mx.sum(kl_div, axis=-1).mean()

    # Scale by temperature squared (from KD paper)
    # This compensates for the temperature scaling in the gradients
    kl_loss = kl_loss * (temperature ** 2)

    return kl_loss


def compute_ce_loss(
    logits: mx.array,
    targets: mx.array,
    ignore_index: int = -100
) -> mx.array:
    """
    Compute cross-entropy loss with optional target masking.

    Args:
        logits: Model logits, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Target value to ignore in loss computation

    Returns:
        Scalar cross-entropy loss
    """
    # Compute log probabilities
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-9)

    # Create mask for valid targets
    mask = targets != ignore_index

    # Gather log probabilities for target tokens
    # This is equivalent to: log_probs[range(len), targets]
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = log_probs.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    # Compute negative log likelihood
    nll = -mx.take_along_axis(
        flat_logits,
        flat_targets[:, None],
        axis=-1
    ).squeeze(-1)

    # Apply mask and compute mean
    flat_mask = mask.reshape(-1)
    masked_nll = mx.where(flat_mask, nll, mx.zeros_like(nll))

    num_valid = mx.sum(flat_mask)
    ce_loss = mx.sum(masked_nll) / mx.maximum(num_valid, mx.array(1.0))

    return ce_loss


def align_vocab_sizes(
    teacher_logits: mx.array,
    student_logits: mx.array,
    strategy: str = "truncate"
) -> Tuple[mx.array, mx.array]:
    """
    Align teacher and student logits when they have different vocabulary sizes.

    Common scenarios:
        - Qwen (151k vocab) → Llama (32k vocab)
        - Different tokenizers with different vocab coverage

    Args:
        teacher_logits: Teacher logits, shape [..., teacher_vocab_size]
        student_logits: Student logits, shape [..., student_vocab_size]
        strategy: Alignment strategy ("truncate" or "project")

    Returns:
        Tuple of (aligned_teacher_logits, aligned_student_logits)

    Strategies:
        - truncate: Use only the common vocabulary (simple, may lose information)
        - project: Project larger vocab to smaller (preserves more information)
    """
    teacher_vocab = teacher_logits.shape[-1]
    student_vocab = student_logits.shape[-1]

    if teacher_vocab == student_vocab:
        return teacher_logits, student_logits

    logger.debug(
        f"Aligning vocabulary sizes: teacher={teacher_vocab}, "
        f"student={student_vocab}, strategy={strategy}"
    )

    if strategy == "truncate":
        # Simple truncation to the smaller vocabulary
        min_vocab = min(teacher_vocab, student_vocab)
        return teacher_logits[..., :min_vocab], student_logits[..., :min_vocab]

    elif strategy == "project":
        # Project larger vocabulary to smaller using learned projection
        # This is more sophisticated but requires additional parameters

        if teacher_vocab > student_vocab:
            # Project teacher down to student size
            # Use simple truncation for now, could add learned projection
            teacher_logits = teacher_logits[..., :student_vocab]
        else:
            # Pad student up to teacher size with -inf
            # This ensures softmax puts ~0 probability on extra tokens
            pad_size = teacher_vocab - student_vocab
            padding = mx.full(
                (*student_logits.shape[:-1], pad_size),
                -1e9,
                dtype=student_logits.dtype
            )
            student_logits = mx.concatenate([student_logits, padding], axis=-1)

        return teacher_logits, student_logits

    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")


class TrueKnowledgeDistiller:
    """
    True Knowledge Distillation trainer with logit matching.

    This class implements the full KD pipeline:
        1. Load teacher (EGO) and student models
        2. Generate training data with teacher's soft targets
        3. Train student to match teacher's probability distributions
        4. Save student adapters for each domain

    The key difference from simple distillation:
        - Simple: Student learns teacher's final outputs
        - TRUE KD: Student learns teacher's probability distributions (dark knowledge)

    Example:
        distiller = TrueKnowledgeDistiller(
            teacher_model_id="mlx-community/Qwen3-8B-4bit",
            student_model_id="mlx-community/Qwen3-1.7B-4bit"
        )

        result = distiller.distill_domain(
            domain="thinking",
            train_prompts=thinking_prompts
        )
    """

    def __init__(
        self,
        teacher_model_id: str,
        student_model_id: str,
        config: Optional[KDConfig] = None
    ):
        """
        Initialize the Knowledge Distillation trainer.

        Args:
            teacher_model_id: HuggingFace model ID for teacher (e.g., Qwen3-8B)
            student_model_id: HuggingFace model ID for student (e.g., Llama-3.2-3B)
            config: KD configuration, uses defaults if None
        """
        self.teacher_model_id = teacher_model_id
        self.student_model_id = student_model_id
        self.config = config or KDConfig()

        # Model components (lazy loaded)
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None

        # Training state
        self.results: List[KDResult] = []

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        logger.info("=" * 80)
        logger.info("TrueKnowledgeDistiller Initialized")
        logger.info("=" * 80)
        logger.info(f"Teacher: {teacher_model_id}")
        logger.info(f"Student: {student_model_id}")
        logger.info(f"Temperature: {self.config.temperature}")
        logger.info(f"Alpha (KD weight): {self.config.alpha}")
        logger.info(f"Output: {self.config.output_dir / self.config.adapter_subdir}")
        logger.info("=" * 80)

    def _load_teacher(self) -> None:
        """Lazy-load the teacher model."""
        if self.teacher_model is not None:
            return

        logger.info(f"Loading teacher model: {self.teacher_model_id}")
        try:
            self.teacher_model, self.teacher_tokenizer = load(self.teacher_model_id)
            logger.info("Teacher model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise

    def _load_student(self) -> None:
        """Lazy-load the student model."""
        if self.student_model is not None:
            return

        logger.info(f"Loading student model: {self.student_model_id}")
        try:
            self.student_model, self.student_tokenizer = load(self.student_model_id)
            logger.info("Student model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise

    def _convert_to_lora(self) -> None:
        """Convert student model to LoRA for efficient fine-tuning."""
        from mlx_lm.tuner import linear_to_lora_layers

        logger.info(f"Converting student to LoRA (rank={self.config.lora_rank})")

        try:
            # Skip conversion if model already has LoRA layers
            already_lora = any(
                isinstance(module, LoRALinear)
                for _, module in self.student_model.named_modules()
            )
            if already_lora:
                logger.info("Student already contains LoRA layers; skipping conversion")
                return

            # Determine number of layers
            if hasattr(self.student_model, 'model'):
                num_layers = len(self.student_model.model.layers)
            else:
                num_layers = 12  # Default fallback

            # LoRA configuration
            lora_config = {
                "rank": self.config.lora_rank,
                "scale": self.config.lora_scale,
                "dropout": self.config.lora_dropout,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"]
            }

            # Apply LoRA
            linear_to_lora_layers(self.student_model, num_layers, lora_config)

            logger.info(f"LoRA conversion complete ({num_layers} layers)")

        except Exception as e:
            logger.error(f"LoRA conversion failed: {e}")
            raise

    def generate_training_data(
        self,
        prompts: List[str],
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Generate training data with teacher's soft targets.

        For each prompt:
            1. Teacher generates response + logits
            2. Store prompt, response, and soft targets
            3. These soft targets contain "dark knowledge"

        Args:
            prompts: List of input prompts
            domain: Domain name (for metadata)

        Returns:
            List of training examples with soft targets
        """
        self._load_teacher()
        self._load_student()  # Need student tokenizer for pre-tokenization

        logger.info(f"Generating training data for {domain}")
        logger.info(f"Number of prompts: {len(prompts)}")

        training_data = []

        for idx, prompt in enumerate(prompts):
            try:
                # Generate actual response for reference first
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=0.7)
                teacher_response = generate(
                    self.teacher_model,
                    self.teacher_tokenizer,
                    prompt=prompt,
                    max_tokens=512,
                    sampler=sampler,
                    verbose=False
                )

                # Tokenize complete prompt+response as training text
                # This is what both teacher and student will process
                full_text = prompt + teacher_response
                teacher_tokens = self.teacher_tokenizer.encode(full_text)

                # Also tokenize with student tokenizer for student forward pass
                student_tokens = self.student_tokenizer.encode(full_text)

                # Get teacher logits for the full sequence
                teacher_input_ids = mx.array([teacher_tokens])
                teacher_logits = self.teacher_model(teacher_input_ids)

                # Store training example with soft targets AND tokenized inputs
                training_data.append({
                    "prompt": prompt,
                    "full_text": full_text,
                    "response": teacher_response,
                    "teacher_tokens": teacher_tokens,
                    "student_tokens": student_tokens,
                    "teacher_logits": teacher_logits,  # Soft targets!
                    "domain": domain,
                    "example_id": idx
                })

                if (idx + 1) % 10 == 0:
                    logger.info(f"Generated {idx + 1}/{len(prompts)} examples")

            except Exception as e:
                logger.error(f"Failed to generate example {idx}: {e}")
                continue

        logger.info(f"Generated {len(training_data)} training examples")
        return training_data

    def train_with_kd(
        self,
        train_data: List[Dict[str, Any]],
        domain: str
    ) -> KDResult:
        """
        Train student model with knowledge distillation.

        Custom training loop that:
            1. Forward pass on student
            2. Compute KD loss (student vs teacher logits)
            3. Compute CE loss (student vs ground truth)
            4. Combine losses: total = alpha*KD + (1-alpha)*CE
            5. Backprop and update LoRA parameters

        Args:
            train_data: Training examples with teacher soft targets
            domain: Domain name

        Returns:
            KDResult with training metrics
        """
        self._load_student()
        self._convert_to_lora()

        logger.info("=" * 80)
        logger.info(f"Training Knowledge Distillation for {domain}")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Iterations: {self.config.num_iters}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 80)

        # Initialize optimizer
        optimizer = optim.Adam(learning_rate=self.config.learning_rate)

        # Training metrics
        kd_losses = []
        ce_losses = []
        combined_losses = []

        start_time = datetime.now()

        # Custom training loop
        for iteration in range(self.config.num_iters):
            # Sample batch
            batch_indices = mx.random.randint(
                0, len(train_data), (self.config.batch_size,)
            )
            batch = [train_data[int(i)] for i in batch_indices]

            def compute_losses(model):
                """Compute KD, CE, and combined losses for the batch."""
                batch_kd_loss = 0.0
                batch_ce_loss = 0.0

                for example in batch:
                    # Use pre-tokenized student tokens
                    student_tokens = example["student_tokens"]
                    input_ids = mx.array([student_tokens])

                    # Student forward pass
                    student_logits = model(input_ids)

                    # Get teacher logits (pre-computed)
                    teacher_logits = example["teacher_logits"]

                    # Align SEQUENCE LENGTHS first (different tokenizers = different lengths)
                    teacher_seq_len = teacher_logits.shape[1]
                    student_seq_len = student_logits.shape[1]
                    min_seq_len = min(teacher_seq_len, student_seq_len)

                    # Truncate to minimum sequence length
                    teacher_logits_seq = teacher_logits[:, :min_seq_len, :]
                    student_logits_seq = student_logits[:, :min_seq_len, :]

                    # Then align VOCABULARY sizes
                    teacher_logits_aligned, student_logits_aligned = align_vocab_sizes(
                        teacher_logits_seq, student_logits_seq
                    )

                    # Compute KD loss (logit matching)
                    kd_loss = compute_kd_loss(
                        teacher_logits_aligned,
                        student_logits_aligned,
                        temperature=self.config.temperature
                    )

                    # Compute CE loss (ground truth matching) using student tokens
                    targets = mx.array([student_tokens[1:min_seq_len]])
                    vocab_size = student_logits_aligned.shape[-1]
                    targets = mx.minimum(targets, vocab_size - 1)
                    ce_loss = compute_ce_loss(
                        student_logits_aligned[:, :-1, :],
                        targets
                    )

                    batch_kd_loss += kd_loss
                    batch_ce_loss += ce_loss

                batch_kd_loss /= len(batch)
                batch_ce_loss /= len(batch)

                # Guard against NaNs
                if mx.isnan(batch_kd_loss).any():
                    batch_kd_loss = mx.zeros_like(batch_kd_loss)
                if mx.isnan(batch_ce_loss).any():
                    batch_ce_loss = mx.zeros_like(batch_ce_loss)

                combined = (
                    self.config.alpha * batch_kd_loss +
                    (1.0 - self.config.alpha) * batch_ce_loss
                )
                return combined, batch_kd_loss, batch_ce_loss

            # Compute gradients
            loss_and_grads = nn.value_and_grad(
                self.student_model,
                lambda m: compute_losses(m)[0]
            )
            combined_loss, grads = loss_and_grads(self.student_model)

            # Backward pass and optimization
            optimizer.update(self.student_model, grads)
            mx.eval(self.student_model.parameters(), optimizer.state)

            # Recompute component losses for logging (post-update)
            combined_loss, batch_kd_loss, batch_ce_loss = compute_losses(self.student_model)

            # Track metrics
            kd_losses.append(float(batch_kd_loss))
            ce_losses.append(float(batch_ce_loss))
            combined_losses.append(float(combined_loss))

            # Logging
            if (iteration + 1) % self.config.steps_per_report == 0:
                logger.info(
                    f"Iter {iteration + 1}/{self.config.num_iters}: "
                    f"KD={batch_kd_loss:.4f}, "
                    f"CE={batch_ce_loss:.4f}, "
                    f"Combined={combined_loss:.4f}"
                )

            # Save checkpoint
            if (iteration + 1) % self.config.steps_per_save == 0:
                checkpoint_dir = (
                    self.config.output_dir /
                    self.config.adapter_subdir /
                    domain /
                    f"checkpoint_{iteration + 1}"
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._save_adapter(checkpoint_dir)
                logger.info(f"Checkpoint saved: {checkpoint_dir}")

        # Final save
        adapter_dir = (
            self.config.output_dir /
            self.config.adapter_subdir /
            domain
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self._save_adapter(adapter_dir)

        training_time = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Final KD Loss: {kd_losses[-1]:.4f}")
        logger.info(f"Final CE Loss: {ce_losses[-1]:.4f}")
        logger.info(f"Final Combined Loss: {combined_losses[-1]:.4f}")
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Adapter saved: {adapter_dir}")
        logger.info("=" * 80)

        # Create result
        result = KDResult(
            domain=domain,
            num_samples=len(train_data),
            adapter_path=adapter_dir,
            final_kd_loss=kd_losses[-1],
            final_ce_loss=ce_losses[-1],
            final_combined_loss=combined_losses[-1],
            training_time_seconds=training_time,
            metadata={
                "teacher_model": self.teacher_model_id,
                "student_model": self.student_model_id,
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "lora_rank": self.config.lora_rank,
                "num_iters": self.config.num_iters
            }
        )

        return result

    def _save_adapter(self, adapter_dir: Path) -> None:
        """Save LoRA adapter weights."""
        try:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_file = adapter_dir / "adapters.safetensors"
            config_file = adapter_dir / "adapter_config.json"

            # Extract only LoRA adapter weights in the expected format
            adapter_weights = {}

            def collect_lora_params(params, prefix=""):
                for key, value in params.items():
                    name = f"{prefix}.{key}" if prefix else key

                    # Only save LoRA layers (layers with lora in the name)
                    if "lora" in key.lower():
                        adapter_weights[name] = mx.array(value)
                    elif isinstance(value, dict):
                        collect_lora_params(value, name)

            # Get trainable parameters (these are the LoRA layers)
            from mlx_lm.tuner.lora import LoRALinear

            for name, module in self.student_model.named_modules():
                if isinstance(module, LoRALinear):
                    # Save LoRA A and B matrices with correct naming
                    adapter_weights[f"{name}.lora_a"] = module.lora_a
                    adapter_weights[f"{name}.lora_b"] = module.lora_b

            mx.save_safetensors(str(adapter_file), adapter_weights)

            # Create config compatible with mlx-lm
            from mlx_lm import load
            # Load base model to get its config
            _, base_config = load(self.student_model_id)

            config_data = {
                "lora_rank": self.config.lora_rank,
                "lora_scale": self.config.lora_scale,
                "lora_dropout": self.config.lora_dropout,
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "teacher_model": self.teacher_model_id,
                "student_model": self.student_model_id,
                "timestamp": datetime.now().isoformat(),
                "num_layers": getattr(base_config, "num_layers", 28),
                "model_type": getattr(base_config, "model_type", "qwen2")
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.debug(f"Adapter saved to {adapter_file}")

        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            # Fallback to original custom format
            import traceback
            traceback.print_exc()
            raise

    def distill_domain(
        self,
        domain: str,
        train_prompts: List[str]
    ) -> KDResult:
        """
        Run full knowledge distillation pipeline for a domain.

        Pipeline:
            1. Generate training data with teacher soft targets
            2. Train student to match teacher distributions
            3. Save student adapter
            4. Return training results

        Args:
            domain: Domain name (e.g., "thinking", "action")
            train_prompts: List of training prompts

        Returns:
            KDResult with training metrics
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"KNOWLEDGE DISTILLATION: {domain.upper()}")
        logger.info("=" * 80)

        # Step 1: Generate training data
        training_data = self.generate_training_data(train_prompts, domain)

        if len(training_data) == 0:
            raise ValueError(f"No training data generated for {domain}")

        # Step 2: Train with KD
        result = self.train_with_kd(training_data, domain)

        # Step 3: Store result
        self.results.append(result)

        return result

    def distill_all(self, domains: Optional[Dict[str, List[str]]] = None) -> List[KDResult]:
        """
        Distill all domains with knowledge distillation.

        Args:
            domains: Dict mapping domain names to training prompts
                    If None, uses default DOMAIN_PROMPTS from distillation.py

        Returns:
            List of KDResults for each domain
        """
        if domains is None:
            # Import default domain prompts
            from conch_dna.training.distillation import DOMAIN_PROMPTS
            domains = {
                domain: config["examples"]
                for domain, config in DOMAIN_PROMPTS.items()
            }

        logger.info(f"Starting Knowledge Distillation for {len(domains)} domains")

        results = []
        for domain, prompts in domains.items():
            try:
                result = self.distill_domain(domain, prompts)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to distill {domain}: {e}")
                import traceback
                traceback.print_exc()

        # Save summary
        self._save_summary()

        return results

    def _save_summary(self) -> None:
        """Save distillation summary to JSON."""
        summary_path = (
            self.config.output_dir /
            self.config.adapter_subdir /
            "kd_summary.json"
        )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "teacher_model": self.teacher_model_id,
            "student_model": self.student_model_id,
            "config": {
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
                "num_iters": self.config.num_iters
            },
            "results": [
                {
                    "domain": r.domain,
                    "num_samples": r.num_samples,
                    "adapter_path": str(r.adapter_path),
                    "final_kd_loss": r.final_kd_loss,
                    "final_ce_loss": r.final_ce_loss,
                    "final_combined_loss": r.final_combined_loss,
                    "training_time_seconds": r.training_time_seconds
                }
                for r in self.results
            ]
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Distillation summary saved: {summary_path}")


def get_kd_adapter_path(domain: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Get the path to a KD-trained adapter.

    Args:
        domain: Domain name
        base_dir: Base directory (defaults to models/distilled_neurons/adapters_kd)

    Returns:
        Path to adapter directory if it exists, None otherwise
    """
    if base_dir is None:
        base_dir = Path("models/distilled_neurons/adapters_kd")

    adapter_path = base_dir / domain

    return adapter_path if adapter_path.exists() else None


# Example usage and CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Conch DNA neurons with TRUE Knowledge Distillation"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="mlx-community/Qwen3-8B-4bit",
        help="Teacher model ID"
    )
    parser.add_argument(
        "--student",
        type=str,
        default="mlx-community/Qwen3-1.7B-4bit",
        help="Student model ID"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Specific domain to distill"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Distill all domains"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for KD"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="KD loss weight (alpha)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Training iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/distilled_neurons",
        help="Output directory"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create config
    config = KDConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        lora_rank=args.lora_rank,
        num_iters=args.iters,
        output_dir=Path(args.output_dir)
    )

    # Initialize distiller
    distiller = TrueKnowledgeDistiller(
        teacher_model_id=args.teacher,
        student_model_id=args.student,
        config=config
    )

    # Run distillation
    if args.all:
        results = distiller.distill_all()

        # Print summary
        print("\n" + "=" * 80)
        print("KNOWLEDGE DISTILLATION SUMMARY")
        print("=" * 80)
        for result in results:
            print(f"\n{result.domain}:")
            print(f"  Samples: {result.num_samples}")
            print(f"  Final KD Loss: {result.final_kd_loss:.4f}")
            print(f"  Final CE Loss: {result.final_ce_loss:.4f}")
            print(f"  Combined Loss: {result.final_combined_loss:.4f}")
            print(f"  Time: {result.training_time_seconds:.2f}s")
            print(f"  Adapter: {result.adapter_path}")
        print("=" * 80)

    elif args.domain:
        # Import domain prompts
        from conch_dna.training.distillation import DOMAIN_PROMPTS

        if args.domain not in DOMAIN_PROMPTS:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available domains: {list(DOMAIN_PROMPTS.keys())}")
            exit(1)

        prompts = DOMAIN_PROMPTS[args.domain]["examples"]
        result = distiller.distill_domain(args.domain, prompts)

        # Print result
        print("\n" + "=" * 80)
        print(f"KNOWLEDGE DISTILLATION COMPLETE: {args.domain}")
        print("=" * 80)
        print(f"Samples: {result.num_samples}")
        print(f"Final KD Loss: {result.final_kd_loss:.4f}")
        print(f"Final CE Loss: {result.final_ce_loss:.4f}")
        print(f"Combined Loss: {result.final_combined_loss:.4f}")
        print(f"Training Time: {result.training_time_seconds:.2f}s")
        print(f"Adapter Path: {result.adapter_path}")
        print("=" * 80)

    else:
        parser.print_help()

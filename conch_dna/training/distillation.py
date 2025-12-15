"""
Conch DNA - Neuron Distillation Pipeline

Distills specialized CORTEX neurons FROM the trained EGO model.
This ensures all neurons inherit the EGO's "DNA" - its values, personality,
and learned behaviors - while being specialized for specific cognitive domains.

Architecture:
    EGO (Teacher) → Distillation → CORTEX Neurons (Students)

    The EGO model (7B) generates high-quality training data for each domain.
    Student neurons (1.5B) are trained on this data via knowledge distillation.
    This creates a lineage where neurons inherit the parent's characteristics.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


# Domain-specific generation prompts for creating training data
DOMAIN_PROMPTS = {
    "thinking": {
        "system": """You are the reasoning core of an AI system. Your role is to think deeply
about problems, break them down, and provide structured analysis.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "thought": "your main reasoning",
  "reasoning_type": "analytical|creative|strategic|critical",
  "confidence_level": "high|medium|low",
  "key_insights": ["list", "of", "important", "observations"],
  "concerns": ["potential", "issues", "or", "uncertainties"]
}""",
        "examples": [
            "Analyze the trade-offs between microservices and monolithic architecture",
            "Think through how to implement a rate limiter with sliding window",
            "Reason about the security implications of storing API keys in environment variables",
            "Break down the steps to migrate a PostgreSQL database with zero downtime",
            "Analyze why this Python function might be slow: def fib(n): return fib(n-1)+fib(n-2) if n>1 else n",
            "Evaluate the pros and cons of using GraphQL vs REST for a mobile app backend",
            "Think about how to design a caching strategy for a high-traffic e-commerce site",
            "Analyze the implications of switching from SQL to NoSQL for a social media app",
            "Reason through the best approach for handling file uploads in a distributed system",
            "Break down the considerations for implementing OAuth 2.0 authentication",
            "Think about the scalability challenges of a real-time chat application",
            "Analyze the trade-offs between synchronous and asynchronous communication",
            "Reason about when to use server-side rendering vs client-side rendering",
            "Think through how to implement optimistic UI updates safely",
            "Analyze the memory management implications of using closures in JavaScript",
            "Break down the steps to implement a blue-green deployment strategy",
            "Think about the security implications of allowing user-uploaded scripts",
            "Reason through the best approach for handling time zones in a global app",
            "Analyze the trade-offs between horizontal and vertical scaling",
            "Think about how to design an effective retry mechanism with exponential backoff",
        ]
    },
    "task": {
        "system": """You are a task extraction and prioritization specialist. Given a context,
identify actionable tasks, break them down, and prioritize them.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "new_tasks": [{"id": 1, "description": "task description", "priority": 1, "dependencies": [], "estimated_effort": "2 hours"}],
  "ranked_task_ids": [1, 2, 3],
  "rationale": "explanation of prioritization"
}""",
        "examples": [
            "Create a Python web API with user authentication using FastAPI",
            "Set up CI/CD pipeline for a React application with testing",
            "Refactor legacy codebase to use modern async/await patterns",
            "Implement real-time notifications using WebSockets",
            "Build a CLI tool that generates project scaffolding",
            "Migrate a monolithic application to microservices architecture",
            "Implement a search feature with full-text search and filters",
            "Add internationalization support to an existing React app",
            "Create a data pipeline for ETL from multiple sources",
            "Implement a payment processing system with Stripe integration",
            "Build a file sharing system with permission controls",
            "Create a dashboard with real-time analytics and charts",
            "Implement a recommendation engine using collaborative filtering",
            "Build an admin panel for content management",
            "Create an API rate limiting system with Redis",
            "Implement email notification system with templates",
            "Build a logging and monitoring infrastructure",
            "Create a backup and disaster recovery system",
            "Implement A/B testing framework for feature flags",
            "Build a user onboarding flow with progress tracking",
        ]
    },
    "action": {
        "system": """You are an action selection specialist. Given a task context and available
tools, select the best action.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "action_type": "TOOL_CALL|RESPOND|THINK_MORE|DO_NOTHING",
  "tool_name": "name of tool if TOOL_CALL",
  "arguments": {"arg1": "value1"},
  "expected_outcome": "what should happen",
  "fallback_action": "what to do if this fails"
}""",
        "examples": [
            "Read config.json and update the version number to 2.0.0",
            "Search for all Python files containing 'deprecated' comments",
            "Create a new directory structure for a microservice",
            "Run the test suite and report any failures",
            "Git commit the changes with a descriptive message",
            "Install missing npm dependencies from package.json",
            "Check if the database connection is working",
            "Create a new branch for the feature development",
            "Run linting and fix auto-fixable issues",
            "Generate API documentation from code comments",
            "Backup the current database before migration",
            "Kill the process running on port 3000",
            "Check disk space and clean up if needed",
            "Fetch the latest changes from the remote repository",
            "Validate the JSON schema of the config file",
            "Compress and archive old log files",
            "Update environment variables in .env file",
            "Check SSL certificate expiration dates",
            "Restart the application after config changes",
            "Run database migrations and verify schema",
        ]
    },
    "reflection": {
        "system": """You are a reflection and learning specialist. Analyze past actions,
outcomes, and identify learnings.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "observation": "what happened",
  "outcome_assessment": "success|partial|failure",
  "lessons_learned": ["insight 1", "insight 2"],
  "behavioral_adjustments": ["adjustment 1", "adjustment 2"],
  "confidence_in_learning": 0.8
}""",
        "examples": [
            "The API response was slow (2s). Analyze and suggest improvements.",
            "The test failed due to a race condition. Reflect on prevention.",
            "User feedback was negative about the response verbosity. Adjust.",
            "The tool selection was suboptimal. Learn from this experience.",
            "Code review found 3 bugs. Analyze patterns to prevent future issues.",
            "The deployment failed due to missing environment variables.",
            "Memory usage increased by 50% after the last release.",
            "The user misunderstood the instructions I provided.",
            "The database query took 10 seconds instead of expected 100ms.",
            "The error message I generated was not helpful for debugging.",
            "I made an assumption that turned out to be incorrect.",
            "The file I tried to edit didn't exist at the expected path.",
            "The authentication token expired during a long operation.",
            "I provided outdated information about a library version.",
            "The regex pattern I used matched too many false positives.",
            "I underestimated the complexity of the requested feature.",
            "The API endpoint returned a different schema than expected.",
            "I missed an edge case that caused a production bug.",
            "The code I generated had a security vulnerability.",
            "I should have asked for clarification before proceeding.",
        ]
    },
    "debug": {
        "system": """You are a debugging and error analysis specialist. Given error
information, diagnose the root cause and suggest fixes.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "error_type": "categorization of the error",
  "root_cause": "underlying issue explanation",
  "fix_suggestions": [{"fix": "description", "confidence": 0.9}],
  "prevention_measures": ["measure 1", "measure 2"],
  "severity": "critical|high|medium|low"
}""",
        "examples": [
            "TypeError: Cannot read property 'map' of undefined at line 45",
            "Connection timeout after 30s to database host",
            "Memory usage spiked to 95% during image processing",
            "Tests pass locally but fail in CI with permission denied",
            "ModuleNotFoundError: No module named 'requests'",
            "CORS error: Access-Control-Allow-Origin header missing",
            "SSL certificate verification failed for API endpoint",
            "JSON parse error: Unexpected token at position 0",
            "Database deadlock detected during concurrent updates",
            "Out of memory error during large file upload",
            "Invalid UTF-8 sequence in request body",
            "Rate limit exceeded: 429 Too Many Requests",
            "Socket hang up error during long-running request",
            "ENOSPC: no space left on device",
            "Maximum call stack size exceeded (recursive function)",
            "Foreign key constraint violation on insert",
            "Authentication failed: token signature invalid",
            "DNS resolution failed for external service",
            "File descriptor limit reached: too many open files",
            "Segmentation fault in native module",
        ]
    },
    "memory": {
        "system": """You are a memory importance and retrieval specialist. Assess the
importance of information and determine what to remember.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations before or after.
Do NOT use <think> tags or any other wrapper - output raw JSON directly.

Required JSON format:
{
  "importance_score": 0.85,
  "is_sacred": true,
  "memory_type": "principle|fact|experience|preference|context",
  "key_entities": ["entity1", "entity2"],
  "retrieval_cues": ["keyword1", "keyword2"]
}""",
        "examples": [
            "User preference: Always use TypeScript over JavaScript",
            "Critical principle: Never execute commands that delete files without confirmation",
            "Context: Working on a healthcare application with HIPAA requirements",
            "Experience: Last API design decision led to performance issues",
            "Fact: The production database is hosted on AWS RDS in us-east-1",
            "User preference: Prefers functional programming over OOP",
            "Context: This is a startup with rapid iteration cycles",
            "Fact: The API uses JWT tokens with 24-hour expiration",
            "Experience: Using regex for HTML parsing caused bugs",
            "Principle: Always validate user input before database operations",
            "Preference: User likes detailed explanations with examples",
            "Context: The project uses a monorepo with pnpm workspaces",
            "Fact: Maximum file size allowed is 10MB",
            "Experience: Premature optimization caused code complexity",
            "Principle: Write tests before implementing features (TDD)",
            "Preference: User prefers snake_case over camelCase",
            "Context: Deploying to Kubernetes on Google Cloud",
            "Fact: The Redis cache has a 1-hour TTL for session data",
            "Experience: Not handling pagination caused memory issues",
            "Principle: Never store secrets in source code",
        ]
    }
}


@dataclass
class DistillationConfig:
    """Configuration for neuron distillation."""
    ego_model: str = "mlx-community/Qwen3-8B-4bit"
    student_base: str = "mlx-community/Qwen3-1.7B-4bit"
    output_dir: Path = Path("models/distilled_neurons")
    samples_per_domain: int = 50
    lora_rank: int = 16
    learning_rate: float = 1e-4
    iters: int = 100
    batch_size: int = 4


@dataclass
class DistillationResult:
    """Result of distilling a neuron."""
    domain: str
    samples_generated: int
    adapter_path: Path
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NeuronDistiller:
    """Distills specialized neurons from the EGO model."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.ego_model = None
        self.ego_tokenizer = None
        self.results: List[DistillationResult] = []
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"NeuronDistiller initialized with EGO: {config.ego_model}")

    def load_ego(self) -> None:
        """Load the EGO model as teacher."""
        if self.ego_model is not None:
            return
        try:
            from mlx_lm import load
            logger.info(f"Loading EGO model: {self.config.ego_model}")
            self.ego_model, self.ego_tokenizer = load(self.config.ego_model)
            logger.info("EGO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EGO model: {e}")
            raise

    def generate_domain_data(self, domain: str, num_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Generate training data for a domain using the EGO."""
        if domain not in DOMAIN_PROMPTS:
            raise ValueError(f"Unknown domain: {domain}")

        self.load_ego()
        num_samples = num_samples or self.config.samples_per_domain
        domain_config = DOMAIN_PROMPTS[domain]
        samples = []

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.7)
        logger.info(f"Generating {num_samples} training samples for {domain}")

        for i in range(num_samples):
            base_example = domain_config["examples"][i % len(domain_config["examples"])]

            # Add variation to examples
            if i >= len(domain_config["examples"]):
                base_example = f"{base_example} (variation {i // len(domain_config['examples'])})"

            prompt = f"""<|im_start|>system
{domain_config["system"]}
<|im_end|>
<|im_start|>user
{base_example}
<|im_end|>
<|im_start|>assistant
"""
            response = generate(
                self.ego_model,
                self.ego_tokenizer,
                prompt=prompt,
                max_tokens=512,
                sampler=sampler,
                verbose=False
            )

            samples.append({
                "input": base_example,
                "output": response,
                "system": domain_config["system"],
                "domain": domain
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples} samples")

        logger.info(f"Generated {len(samples)} samples for {domain}")
        return samples

    def save_training_data(self, samples: List[Dict[str, str]], domain: str) -> Path:
        """Save training data in JSONL format."""
        data_dir = self.config.output_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / f"{domain}_training.jsonl"

        with open(output_path, 'w') as f:
            for sample in samples:
                training_text = f"""<|im_start|>system
{sample['system']}
<|im_end|>
<|im_start|>user
{sample['input']}
<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""
                f.write(json.dumps({"text": training_text}) + '\n')

        logger.info(f"Saved training data to {output_path}")
        return output_path

    def train_student(self, domain: str, training_data_path: Path) -> Path:
        """Train a student neuron on distilled data using LoRA."""
        adapter_dir = self.config.output_dir / "adapters" / domain
        adapter_dir.mkdir(parents=True, exist_ok=True)

        try:
            from mlx_lm import load
            from mlx_lm.tuner import TrainingArgs, train, linear_to_lora_layers
            from mlx_lm.tuner.datasets import load_dataset

            # Load student base model
            logger.info(f"Loading student base: {self.config.student_base}")
            model, tokenizer = load(self.config.student_base)

            # Convert to LoRA
            linear_to_lora_layers(model, self.config.lora_rank)

            # Load training data
            train_set = load_dataset(tokenizer, str(training_data_path))
            val_set = load_dataset(tokenizer, str(training_data_path))  # Use same for now

            # Create optimizer
            optimizer = optim.Adam(learning_rate=self.config.learning_rate)

            # Training args
            train_args = TrainingArgs(
                batch_size=self.config.batch_size,
                iters=self.config.iters,
                val_batches=5,
                steps_per_report=10,
                steps_per_eval=50,
                steps_per_save=50,
                adapter_file=str(adapter_dir / "adapters.safetensors")
            )

            logger.info(f"Training {domain} neuron with LoRA rank {self.config.lora_rank}")
            train(model, optimizer, train_set, val_set, train_args)

            logger.info(f"Training complete. Adapter saved to {adapter_dir}")
            return adapter_dir

        except Exception as e:
            logger.error(f"Training failed for {domain}: {e}")
            # Save the training data even if training fails - it can be trained later
            logger.info(f"Training data preserved at {training_data_path}")
            return adapter_dir

    def distill_domain(self, domain: str) -> DistillationResult:
        """Run full distillation pipeline for a domain."""
        logger.info(f"=" * 60)
        logger.info(f"DISTILLING {domain.upper()} NEURON FROM EGO")
        logger.info(f"=" * 60)

        # Step 1: Generate training data from EGO
        samples = self.generate_domain_data(domain)

        # Step 2: Save training data
        data_path = self.save_training_data(samples, domain)

        # Step 3: Train student on EGO-generated data
        adapter_path = self.train_student(domain, data_path)

        result = DistillationResult(
            domain=domain,
            samples_generated=len(samples),
            adapter_path=adapter_path
        )
        self.results.append(result)
        return result

    def distill_all(self) -> List[DistillationResult]:
        """Distill all CORTEX neurons from EGO."""
        domains = list(DOMAIN_PROMPTS.keys())
        logger.info(f"Starting distillation of {len(domains)} neurons from EGO")

        for domain in domains:
            try:
                self.distill_domain(domain)
            except Exception as e:
                logger.error(f"Failed to distill {domain}: {e}")

        self.save_summary()
        return self.results

    def save_summary(self) -> Path:
        """Save distillation summary."""
        summary_path = self.config.output_dir / "distillation_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "ego_model": self.config.ego_model,
                "student_base": self.config.student_base,
                "lora_rank": self.config.lora_rank
            },
            "results": [
                {"domain": r.domain, "samples": r.samples_generated, "adapter": str(r.adapter_path)}
                for r in self.results
            ]
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary_path


def get_distilled_neuron_path(domain: str, base_dir: Path = None) -> Optional[Path]:
    """Get the path to a distilled neuron adapter."""
    base_dir = base_dir or Path("models/distilled_neurons/adapters")
    adapter_path = base_dir / domain
    return adapter_path if adapter_path.exists() else None

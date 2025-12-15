#!/usr/bin/env python3
"""
Conch DNA - Neuron Quality Benchmark Suite

Evaluates distilled neurons against EGO (teacher) baseline across all 6 domains.
Tests structural completeness, quality criteria, and overall response coherence.

Metrics:
- Structural completeness: Expected JSON fields present
- Quality criteria: Domain-relevant keyword matching
- Overall quality: Length, JSON validity, coherence
- Pass rates and quality gaps between neuron and EGO

Usage:
    python -m conch_dna.training.benchmark --domain thinking
    python -m conch_dna.training.benchmark --all
"""

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Import domain prompts from distillation module
try:
    from conch_dna.training.distillation import DOMAIN_PROMPTS
except ImportError:
    logger.warning("Could not import DOMAIN_PROMPTS, using stub definitions")
    DOMAIN_PROMPTS = {
        "thinking": {"system": "You are a reasoning specialist."},
        "task": {"system": "You are a task extraction specialist."},
        "action": {"system": "You are an action selection specialist."},
        "reflection": {"system": "You are a reflection specialist."},
        "debug": {"system": "You are a debugging specialist."},
        "memory": {"system": "You are a memory importance specialist."},
    }

# MLX imports for model loading
try:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available - benchmark cannot run without mlx-lm")


# Benchmark test cases: 3 per domain covering different complexity levels
BENCHMARK_CASES = {
    "thinking": [
        {
            "prompt": "Analyze the trade-offs between microservices and monolithic architecture",
            "expected_fields": ["thought", "reasoning_type", "confidence_level", "key_insights", "concerns"],
            "quality_keywords": ["scalability", "complexity", "deployment", "maintenance", "performance"],
        },
        {
            "prompt": "Think through how to implement a rate limiter with sliding window",
            "expected_fields": ["thought", "reasoning_type", "confidence_level", "key_insights", "concerns"],
            "quality_keywords": ["window", "counter", "timestamp", "redis", "throughput", "algorithm"],
        },
        {
            "prompt": "Reason about the security implications of storing API keys in environment variables",
            "expected_fields": ["thought", "reasoning_type", "confidence_level", "key_insights", "concerns"],
            "quality_keywords": ["security", "vault", "encryption", "exposure", "risk", "secrets"],
        },
    ],
    "task": [
        {
            "prompt": "Create a Python web API with user authentication using FastAPI",
            "expected_fields": ["new_tasks", "ranked_task_ids", "rationale"],
            "quality_keywords": ["authentication", "endpoint", "database", "jwt", "password", "middleware"],
        },
        {
            "prompt": "Set up CI/CD pipeline for React application with testing",
            "expected_fields": ["new_tasks", "ranked_task_ids", "rationale"],
            "quality_keywords": ["pipeline", "test", "build", "deploy", "github", "docker"],
        },
        {
            "prompt": "Refactor legacy codebase to use modern async/await patterns",
            "expected_fields": ["new_tasks", "ranked_task_ids", "rationale"],
            "quality_keywords": ["async", "await", "refactor", "callback", "promise", "migration"],
        },
    ],
    "action": [
        {
            "prompt": "Read config.json and update the version number to 2.0.0",
            "expected_fields": ["action_type", "tool_name", "arguments", "expected_outcome", "fallback_action"],
            "quality_keywords": ["read", "write", "json", "version", "file", "update"],
        },
        {
            "prompt": "Search for all Python files containing 'deprecated' comments",
            "expected_fields": ["action_type", "tool_name", "arguments", "expected_outcome", "fallback_action"],
            "quality_keywords": ["search", "grep", "find", "python", "deprecated", "files"],
        },
        {
            "prompt": "Create a new directory structure for a microservice",
            "expected_fields": ["action_type", "tool_name", "arguments", "expected_outcome", "fallback_action"],
            "quality_keywords": ["directory", "mkdir", "structure", "create", "folder"],
        },
    ],
    "reflection": [
        {
            "prompt": "The API response was slow (2s). Analyze and suggest improvements.",
            "expected_fields": ["observation", "outcome_assessment", "lessons_learned", "behavioral_adjustments", "confidence_in_learning"],
            "quality_keywords": ["performance", "optimization", "cache", "query", "bottleneck", "latency"],
        },
        {
            "prompt": "The test failed due to a race condition. Reflect on prevention.",
            "expected_fields": ["observation", "outcome_assessment", "lessons_learned", "behavioral_adjustments", "confidence_in_learning"],
            "quality_keywords": ["race", "concurrency", "lock", "synchronization", "async", "thread"],
        },
        {
            "prompt": "User feedback was negative about response verbosity. Adjust.",
            "expected_fields": ["observation", "outcome_assessment", "lessons_learned", "behavioral_adjustments", "confidence_in_learning"],
            "quality_keywords": ["verbose", "concise", "clarity", "communication", "feedback", "brevity"],
        },
    ],
    "debug": [
        {
            "prompt": "TypeError: Cannot read property 'map' of undefined at line 45",
            "expected_fields": ["error_type", "root_cause", "fix_suggestions", "prevention_measures", "severity"],
            "quality_keywords": ["undefined", "null", "check", "validation", "guard", "defensive"],
        },
        {
            "prompt": "Connection timeout after 30s to database host",
            "expected_fields": ["error_type", "root_cause", "fix_suggestions", "prevention_measures", "severity"],
            "quality_keywords": ["connection", "timeout", "network", "firewall", "retry", "configuration"],
        },
        {
            "prompt": "Memory usage spiked to 95% during image processing",
            "expected_fields": ["error_type", "root_cause", "fix_suggestions", "prevention_measures", "severity"],
            "quality_keywords": ["memory", "leak", "optimization", "garbage", "buffer", "allocation"],
        },
    ],
    "memory": [
        {
            "prompt": "User preference: Always use TypeScript over JavaScript",
            "expected_fields": ["importance_score", "is_sacred", "memory_type", "key_entities", "retrieval_cues"],
            "quality_keywords": ["preference", "typescript", "javascript", "language", "coding"],
        },
        {
            "prompt": "Critical principle: Never execute commands that delete files without confirmation",
            "expected_fields": ["importance_score", "is_sacred", "memory_type", "key_entities", "retrieval_cues"],
            "quality_keywords": ["safety", "delete", "confirmation", "destructive", "principle"],
        },
        {
            "prompt": "Context: Working on healthcare application with HIPAA requirements",
            "expected_fields": ["importance_score", "is_sacred", "memory_type", "key_entities", "retrieval_cues"],
            "quality_keywords": ["hipaa", "healthcare", "compliance", "security", "privacy"],
        },
    ],
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite."""
    ego_model: str = "mlx-community/Qwen3-8B-4bit"
    student_base: str = "mlx-community/Qwen3-1.7B-4bit"
    adapter_dir: Path = Path("models/distilled_neurons/adapters")
    output_dir: Path = Path("models/distilled_neurons/benchmarks")
    max_tokens: int = 512
    temperature: float = 0.7
    verbose_generation: bool = False


@dataclass
class BenchmarkResult:
    """Result for a single test case."""
    domain: str
    prompt: str

    # Outputs
    ego_output: str
    neuron_output: str

    # Structural completeness
    ego_has_all_fields: bool
    neuron_has_all_fields: bool
    missing_fields: List[str]

    # Quality criteria
    ego_quality_score: float  # 0-1 based on keyword matching
    neuron_quality_score: float
    quality_keywords_found: List[str]

    # Overall quality
    ego_json_valid: bool
    neuron_json_valid: bool
    ego_length: int
    neuron_length: int

    # Pass/fail
    passed: bool
    error: str = ""


@dataclass
class DomainBenchmarkSummary:
    """Summary statistics for a domain."""
    domain: str
    total_tests: int
    passed_tests: int
    failed_tests: int

    # Pass rates
    pass_rate: float
    structural_completeness_rate: float
    json_validity_rate: float

    # Quality gaps
    avg_quality_gap: float  # neuron - ego (negative = worse)
    avg_length_ratio: float  # neuron / ego

    # Recommendations
    recommendation: str  # "ready", "needs_improvement", "requires_retraining"

    def __post_init__(self):
        """Calculate pass rate."""
        self.pass_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


class NeuronBenchmark:
    """Benchmark suite for neuron quality assessment."""

    def __init__(self, config: BenchmarkConfig = None):
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.ego_model = None
        self.ego_tokenizer = None
        self.neuron_cache = {}  # Cache loaded neurons

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX not available. Install mlx-lm to run benchmarks:\n"
                "  pip install mlx-lm"
            )

        logger.info(f"NeuronBenchmark initialized")
        logger.info(f"  EGO model: {self.config.ego_model}")
        logger.info(f"  Student base: {self.config.student_base}")
        logger.info(f"  Adapter directory: {self.config.adapter_dir}")

    def _get_adapter_base_model(self, domain: str) -> Optional[str]:
        """Resolve the correct base model for a domain adapter.

        Prefers the adapter's config file so we don't accidentally load
        LoRA weights trained on a different base checkpoint.
        """
        adapter_dir = self.config.adapter_dir / domain
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            # Support both historical keys ("model") and TRUE_KD-style ("student_model")
            for key in ("student_model", "model"):
                model_id = config_data.get(key)
                if model_id:
                    return model_id
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to read adapter config for {domain}: {exc}")

        return None

    def load_ego(self) -> None:
        """Load EGO (teacher) model."""
        if self.ego_model is not None:
            return

        logger.info(f"Loading EGO model: {self.config.ego_model}")
        self.ego_model, self.ego_tokenizer = load(self.config.ego_model)
        logger.info("EGO model loaded successfully")

    def load_neuron(self, domain: str) -> Tuple[Any, Any]:
        """Load neuron model with adapter for a domain.

        Args:
            domain: Domain name (thinking, task, action, etc.)

        Returns:
            Tuple of (model, tokenizer)
        """
        if domain in self.neuron_cache:
            return self.neuron_cache[domain]

        # adapter_path should be the DIRECTORY containing adapters.safetensors
        adapter_dir = self.config.adapter_dir / domain
        adapter_file = adapter_dir / "adapters.safetensors"
        adapter_base = self._get_adapter_base_model(domain)
        base_model_id = adapter_base or self.config.student_base

        if not adapter_base:
            logger.warning(
                f"No adapter_config.json found for {domain}; falling back to student base {base_model_id}"
            )

        if not adapter_file.exists():
            logger.warning(f"No adapter found for {domain} at {adapter_file}")
            logger.warning(f"Loading base model without adapter")
            model, tokenizer = load(base_model_id)
        else:
            logger.info(
                f"Loading {domain} neuron with adapter from {adapter_dir} using base {base_model_id}"
            )
            model, tokenizer = load(
                base_model_id,
                adapter_path=str(adapter_dir)
            )

        self.neuron_cache[domain] = (model, tokenizer)
        return model, tokenizer

    def generate_response(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        domain: str
    ) -> str:
        """Generate response from a model.

        Args:
            model: Model to use
            tokenizer: Tokenizer
            prompt: User prompt
            domain: Domain for system prompt

        Returns:
            Generated response text
        """
        system_prompt = DOMAIN_PROMPTS.get(domain, {}).get("system", "")

        full_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

        sampler = make_sampler(temp=self.config.temperature)
        response = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=self.config.max_tokens,
            sampler=sampler,
            verbose=self.config.verbose_generation
        )

        return response

    def extract_json(self, text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Extract and parse JSON from text.

        Handles various output formats including:
        - Raw JSON
        - Markdown code blocks (```json or ```)
        - Qwen3 thinking blocks (<think>...</think>)
        - Mixed content with JSON embedded

        Args:
            text: Text that may contain JSON

        Returns:
            Tuple of (is_valid, parsed_dict)
        """
        import re

        try:
            json_text = text.strip()

            # Strip Qwen3 thinking blocks: <think>...</think>
            # Handle both complete and incomplete think blocks
            json_text = re.sub(r'<think>[\s\S]*?</think>', '', json_text)
            # Handle incomplete think blocks at start (no closing tag)
            json_text = re.sub(r'^<think>[\s\S]*$', '', json_text)
            json_text = json_text.strip()

            # Strip markdown code blocks
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                # Get content between first pair of ```
                parts = json_text.split("```")
                if len(parts) >= 3:
                    json_text = parts[1].strip()
                elif len(parts) >= 2:
                    json_text = parts[1].strip()

            # Try to find JSON object/array in remaining text
            # Look for first { or [ and match to closing } or ]
            json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', json_text)
            if json_match:
                json_text = json_match.group(1)

            parsed = json.loads(json_text)
            return True, parsed
        except (json.JSONDecodeError, IndexError, ValueError):
            return False, None

    def check_structural_completeness(
        self,
        response: str,
        expected_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if response contains all expected JSON fields.

        Args:
            response: Model response
            expected_fields: List of required field names

        Returns:
            Tuple of (has_all_fields, missing_fields)
        """
        is_valid, parsed = self.extract_json(response)

        if not is_valid or not parsed:
            return False, expected_fields

        missing = [field for field in expected_fields if field not in parsed]
        return len(missing) == 0, missing

    def calculate_quality_score(
        self,
        response: str,
        quality_keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """Calculate quality score based on keyword matching.

        Args:
            response: Model response
            quality_keywords: List of domain-relevant keywords

        Returns:
            Tuple of (quality_score, keywords_found)
        """
        response_lower = response.lower()
        keywords_found = [kw for kw in quality_keywords if kw.lower() in response_lower]
        score = len(keywords_found) / len(quality_keywords) if quality_keywords else 0.0
        return score, keywords_found

    def run_test(
        self,
        domain: str,
        test_case: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single test case.

        Args:
            domain: Domain name
            test_case: Test case dictionary with prompt, expected_fields, quality_keywords

        Returns:
            BenchmarkResult
        """
        prompt = test_case["prompt"]
        expected_fields = test_case["expected_fields"]
        quality_keywords = test_case["quality_keywords"]

        logger.info(f"Testing {domain} prompt:\n{prompt}")

        try:
            # Load models
            self.load_ego()
            neuron_model, neuron_tokenizer = self.load_neuron(domain)

            # Generate responses
            ego_output = self.generate_response(self.ego_model, self.ego_tokenizer, prompt, domain)
            neuron_output = self.generate_response(neuron_model, neuron_tokenizer, prompt, domain)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"EGO output for {domain}:\n{ego_output}")
                logger.debug(f"Neuron output for {domain}:\n{neuron_output}")

            # Check structural completeness
            ego_complete, ego_missing = self.check_structural_completeness(ego_output, expected_fields)
            neuron_complete, neuron_missing = self.check_structural_completeness(neuron_output, expected_fields)

            # Calculate quality scores
            ego_quality, _ = self.calculate_quality_score(ego_output, quality_keywords)
            neuron_quality, keywords_found = self.calculate_quality_score(neuron_output, quality_keywords)

            # Check JSON validity
            ego_json_valid, _ = self.extract_json(ego_output)
            neuron_json_valid, _ = self.extract_json(neuron_output)

            # Overall pass criteria:
            # 1. Neuron output is valid JSON
            # 2. Neuron has all expected fields
            # 3. Neuron quality score >= 50% of EGO's score
            # 4. Neuron response is reasonable length (not too short)
            passed = (
                neuron_json_valid and
                neuron_complete and
                (neuron_quality >= ego_quality * 0.5 if ego_quality > 0 else neuron_quality > 0.3) and
                len(neuron_output) >= 50
            )

            return BenchmarkResult(
                domain=domain,
                prompt=prompt,
                ego_output=ego_output,
                neuron_output=neuron_output,
                ego_has_all_fields=ego_complete,
                neuron_has_all_fields=neuron_complete,
                missing_fields=neuron_missing,
                ego_quality_score=ego_quality,
                neuron_quality_score=neuron_quality,
                quality_keywords_found=keywords_found,
                ego_json_valid=ego_json_valid,
                neuron_json_valid=neuron_json_valid,
                ego_length=len(ego_output),
                neuron_length=len(neuron_output),
                passed=passed
            )

        except Exception as e:
            logger.exception(f"Test failed for {domain}: {e}")
            return BenchmarkResult(
                domain=domain,
                prompt=prompt,
                ego_output="",
                neuron_output="",
                ego_has_all_fields=False,
                neuron_has_all_fields=False,
                missing_fields=expected_fields,
                ego_quality_score=0.0,
                neuron_quality_score=0.0,
                quality_keywords_found=[],
                ego_json_valid=False,
                neuron_json_valid=False,
                ego_length=0,
                neuron_length=0,
                passed=False,
                error=str(e)
            )

    def benchmark_domain(self, domain: str) -> Tuple[DomainBenchmarkSummary, List[BenchmarkResult]]:
        """Benchmark all test cases for a domain.

        Args:
            domain: Domain name

        Returns:
            Tuple of (summary, individual_results)
        """
        logger.info("=" * 70)
        logger.info(f"BENCHMARKING DOMAIN: {domain.upper()}")
        logger.info("=" * 70)

        if domain not in BENCHMARK_CASES:
            raise ValueError(f"Unknown domain: {domain}")

        test_cases = BENCHMARK_CASES[domain]
        results = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nTest {i}/{len(test_cases)}")
            result = self.run_test(domain, test_case)
            results.append(result)

        # Calculate summary statistics
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        structural_complete = sum(1 for r in results if r.neuron_has_all_fields)
        json_valid = sum(1 for r in results if r.neuron_json_valid)

        quality_gaps = [r.neuron_quality_score - r.ego_quality_score for r in results]
        length_ratios = [
            r.neuron_length / r.ego_length if r.ego_length > 0 else 0.0
            for r in results
        ]

        avg_quality_gap = statistics.mean(quality_gaps) if quality_gaps else 0.0
        avg_length_ratio = statistics.mean(length_ratios) if length_ratios else 0.0

        # Determine recommendation
        pass_rate = len(passed) / len(results) if results else 0.0
        if pass_rate >= 0.8 and avg_quality_gap >= -0.1:
            recommendation = "ready"
        elif pass_rate >= 0.5:
            recommendation = "needs_improvement"
        else:
            recommendation = "requires_retraining"

        summary = DomainBenchmarkSummary(
            domain=domain,
            total_tests=len(results),
            passed_tests=len(passed),
            failed_tests=len(failed),
            pass_rate=pass_rate,
            structural_completeness_rate=structural_complete / len(results) if results else 0.0,
            json_validity_rate=json_valid / len(results) if results else 0.0,
            avg_quality_gap=avg_quality_gap,
            avg_length_ratio=avg_length_ratio,
            recommendation=recommendation
        )

        return summary, results

    def benchmark_all(self) -> Dict[str, Tuple[DomainBenchmarkSummary, List[BenchmarkResult]]]:
        """Benchmark all domains.

        Returns:
            Dictionary mapping domain -> (summary, results)
        """
        logger.info("=" * 70)
        logger.info("NEURON QUALITY BENCHMARK SUITE - ALL DOMAINS")
        logger.info("=" * 70)

        all_results = {}

        for domain in BENCHMARK_CASES.keys():
            try:
                summary, results = self.benchmark_domain(domain)
                all_results[domain] = (summary, results)
            except Exception as e:
                logger.error(f"Failed to benchmark {domain}: {e}")

        return all_results

    def save_results(
        self,
        results: Dict[str, Tuple[DomainBenchmarkSummary, List[BenchmarkResult]]],
        filename: str = None
    ) -> Path:
        """Save benchmark results to JSON file.

        Args:
            results: Results dictionary from benchmark_all()
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.config.output_dir / filename

        # Convert to serializable format
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "ego_model": self.config.ego_model,
                "student_base": self.config.student_base,
                "adapter_dir": str(self.config.adapter_dir),
            },
            "domains": {}
        }

        for domain, (summary, test_results) in results.items():
            output_data["domains"][domain] = {
                "summary": asdict(summary),
                "results": [asdict(r) for r in test_results]
            }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")
        return output_path

    def print_summary(
        self,
        results: Dict[str, Tuple[DomainBenchmarkSummary, List[BenchmarkResult]]]
    ) -> None:
        """Print formatted summary of benchmark results.

        Args:
            results: Results dictionary from benchmark_all()
        """
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Overall statistics
        total_tests = sum(s.total_tests for s, _ in results.values())
        total_passed = sum(s.passed_tests for s, _ in results.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0

        print(f"\nOverall: {total_passed}/{total_tests} tests passed ({overall_pass_rate:.1%})")
        print("\nPer-Domain Results:")
        print("-" * 70)

        for domain, (summary, _) in sorted(results.items()):
            status_icon = {
                "ready": "✓",
                "needs_improvement": "⚠",
                "requires_retraining": "✗"
            }.get(summary.recommendation, "?")

            print(f"\n{status_icon} {domain.upper()}")
            print(f"  Pass Rate:        {summary.pass_rate:6.1%} ({summary.passed_tests}/{summary.total_tests})")
            print(f"  JSON Validity:    {summary.json_validity_rate:6.1%}")
            print(f"  Completeness:     {summary.structural_completeness_rate:6.1%}")
            print(f"  Quality Gap:      {summary.avg_quality_gap:+6.1%}")
            print(f"  Length Ratio:     {summary.avg_length_ratio:6.1%}")
            print(f"  Recommendation:   {summary.recommendation}")

        print("\n" + "=" * 70)
        print("\nRecommendations:")
        print("  ✓ ready              - Neuron performs well, safe to use")
        print("  ⚠ needs_improvement  - Neuron works but could be better")
        print("  ✗ requires_retraining - Neuron needs more training data")
        print("=" * 70 + "\n")


def run_benchmark(
    domain: Optional[str] = None,
    config: Optional[BenchmarkConfig] = None
) -> Dict[str, Tuple[DomainBenchmarkSummary, List[BenchmarkResult]]]:
    """Convenience function to run benchmark.

    Args:
        domain: Optional specific domain to test. If None, tests all domains.
        config: Optional custom configuration

    Returns:
        Results dictionary
    """
    benchmark = NeuronBenchmark(config)

    if domain:
        summary, results = benchmark.benchmark_domain(domain)
        all_results = {domain: (summary, results)}
    else:
        all_results = benchmark.benchmark_all()

    benchmark.save_results(all_results)
    benchmark.print_summary(all_results)

    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Conch DNA neurons against EGO baseline"
    )
    parser.add_argument(
        "--domain",
        choices=list(BENCHMARK_CASES.keys()),
        help="Specific domain to benchmark (default: all domains)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all domains (default behavior)"
    )
    parser.add_argument(
        "--ego-model",
        default="mlx-community/Qwen3-8B-4bit",
        help="EGO model identifier"
    )
    parser.add_argument(
        "--student-base",
        default="mlx-community/Qwen3-1.7B-4bit",
        help="Student base model identifier"
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("models/distilled_neurons/adapters"),
        help="Directory containing neuron adapters"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/distilled_neurons/benchmarks"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Create config
    config = BenchmarkConfig(
        ego_model=args.ego_model,
        student_base=args.student_base,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        verbose_generation=args.verbose
    )

    # Run benchmark
    domain = args.domain if not args.all else None
    run_benchmark(domain=domain, config=config)


if __name__ == "__main__":
    main()

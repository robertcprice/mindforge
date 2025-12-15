"""
Conch DNA - Scaled Training Data Generation

Generates 500-1000 diverse, high-quality training samples per domain using
the EGO model as teacher. Implements scenario variations, quality filtering,
and deduplication to ensure training data diversity and quality.

Architecture:
    EGO (Teacher) → Scenario Variations → Quality Filtering → Scaled Training Data

Key Features:
    - 500-1000 samples per domain with diversity guarantees
    - Scenario-based generation (edge cases, failures, complex/simple inputs)
    - Temperature-based sampling for variation
    - JSON validation and quality filtering
    - Similarity-based deduplication
    - Progress tracking and resumability
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# Scenario templates for each domain with variations
SCENARIO_TEMPLATES = {
    "thinking": {
        "analytical": [
            "Analyze the performance implications of using {tech1} vs {tech2}",
            "Compare the trade-offs between {approach1} and {approach2} for {problem}",
            "Evaluate the scalability concerns of {architecture} for {use_case}",
            "Break down the complexity of implementing {feature} in {context}",
            "Examine the failure modes of {system} under {condition}",
        ],
        "creative": [
            "Design an innovative solution for {problem} using {constraint}",
            "Propose alternative approaches to {challenge} that avoid {limitation}",
            "Reimagine how {process} could work if {assumption} weren't true",
            "Generate creative workarounds for {issue} in {environment}",
            "Envision a novel architecture for {system} that prioritizes {quality}",
        ],
        "strategic": [
            "Develop a migration strategy from {old_tech} to {new_tech} for {scenario}",
            "Plan a phased rollout of {feature} considering {constraints}",
            "Strategize how to refactor {component} while maintaining {requirement}",
            "Design a decision framework for choosing between {option1} and {option2}",
            "Map out dependencies for implementing {project} in {timeframe}",
        ],
        "critical": [
            "Identify potential security vulnerabilities in {implementation}",
            "Challenge the assumptions behind {decision} in {context}",
            "Critique the proposed {solution} for {problem} highlighting weaknesses",
            "Question whether {approach} is appropriate for {use_case}",
            "Examine edge cases that could break {system} under {condition}",
        ]
    },
    "task": {
        "simple": [
            "Create a {language} script that {simple_action}",
            "Add {simple_feature} to the existing {component}",
            "Update {config_file} to change {setting}",
            "Fix the {simple_bug} in {file}",
            "Write a {simple_test} for {function}",
        ],
        "complex": [
            "Build a {system} with {feature1}, {feature2}, and {feature3}",
            "Implement {complex_feature} that integrates with {external_system}",
            "Refactor {component} to support {new_requirement} while maintaining {old_behavior}",
            "Design and implement a {architecture} for {use_case}",
            "Migrate {legacy_system} to {modern_system} with {constraint}",
        ],
        "multi_step": [
            "Set up {tool1}, configure {tool2}, and integrate with {tool3}",
            "Create {component1}, test it, then integrate with {component2}",
            "Analyze {codebase}, identify issues, propose fixes, implement changes",
            "Research {technology}, prototype {feature}, evaluate results, iterate",
            "Debug {error}, trace root cause, implement fix, verify solution",
        ],
        "dependencies": [
            "Implement {feature} which depends on {dependency1} and {dependency2}",
            "Complete {task1} before starting {task2} which blocks {task3}",
            "Coordinate changes across {service1}, {service2}, and {service3}",
            "Update {component} ensuring compatibility with {dependent1} and {dependent2}",
            "Refactor {module} considering downstream effects on {consumer1} and {consumer2}",
        ]
    },
    "action": {
        "file_ops": [
            "Read {file} and extract {data_type}",
            "Write {content} to {file} at {location}",
            "Create directory structure {structure} for {purpose}",
            "Delete {pattern} files from {directory}",
            "Move {source} to {destination} and update references",
        ],
        "tool_calls": [
            "Use {tool} to {action} with parameters {params}",
            "Execute {command} and parse the {output_type} output",
            "Invoke {api} with {data} and handle {response_type}",
            "Call {function} with {args} and validate {result}",
            "Run {script} in {environment} and capture {metrics}",
        ],
        "git_ops": [
            "Commit changes to {files} with message describing {change}",
            "Create branch {branch_name} from {base} for {purpose}",
            "Merge {source_branch} into {target_branch} resolving {conflicts}",
            "Revert commit {commit_hash} that introduced {bug}",
            "Tag release {version} after {validation}",
        ],
        "api_calls": [
            "Call {endpoint} with {method} sending {payload}",
            "Fetch data from {api} filtering by {criteria}",
            "Post {data} to {service} and handle {status_codes}",
            "Stream results from {websocket} processing {events}",
            "Poll {endpoint} until {condition} with {timeout}",
        ]
    },
    "reflection": {
        "success_analysis": [
            "{action} succeeded with {metric}. Analyze what went well.",
            "The {approach} worked better than expected for {task}. Extract learnings.",
            "{implementation} achieved {result} efficiently. Identify best practices.",
            "{strategy} led to {positive_outcome}. Understand contributing factors.",
            "{decision} proved correct when {validation}. Capture insights.",
        ],
        "failure_analysis": [
            "{action} failed with {error}. Determine root cause and prevention.",
            "The {approach} didn't work for {task}. Analyze why and learn.",
            "{implementation} resulted in {negative_outcome}. Identify mistakes.",
            "{strategy} led to {failure}. Understand what went wrong.",
            "{decision} proved incorrect when {evidence}. Extract lessons.",
        ],
        "learning_extraction": [
            "After {sequence_of_actions}, what patterns emerge?",
            "Comparing {approach1} and {approach2}, which works better for {context}?",
            "Given {experience}, how should behavior change for {future_scenario}?",
            "{observation} suggests {hypothesis}. How to validate?",
            "The pattern of {repeated_behavior} indicates what adjustment?",
        ]
    },
    "debug": {
        "syntax_errors": [
            "SyntaxError: {error_detail} at line {line} in {file}",
            "{language} parsing failed: {parse_error} in {context}",
            "IndentationError: {detail} at line {line}",
            "Unexpected token {token} at position {position}",
            "Missing {element} in {construct} at line {line}",
        ],
        "runtime_errors": [
            "{exception_type}: {message} at {location} in {file}:{line}",
            "Null pointer exception when accessing {property} of {object}",
            "Index out of bounds: tried to access index {index} of array with length {length}",
            "Division by zero in {calculation} at {context}",
            "Stack overflow in {recursive_function} with input {input}",
        ],
        "logical_errors": [
            "Function {function} returns {wrong_result} instead of {expected} for input {input}",
            "Infinite loop detected in {location} when {condition}",
            "Race condition between {operation1} and {operation2} causing {symptom}",
            "Off-by-one error in {loop} causing {incorrect_behavior}",
            "Wrong comparison operator in {condition} leading to {bug}",
        ],
        "integration_errors": [
            "API call to {endpoint} failed with status {status_code}: {error_message}",
            "Database connection timeout after {duration} to {host}:{port}",
            "Authentication failed for {service}: {auth_error}",
            "Version mismatch between {component1} v{version1} and {component2} v{version2}",
            "Network error: {network_issue} when connecting to {service}",
        ],
        "performance_issues": [
            "Query taking {duration} to return {rows} rows from {table}",
            "Memory usage at {percentage}% after {operation}",
            "CPU spike to {percentage}% during {process}",
            "Response time degraded to {duration} under {load}",
            "Disk I/O bottleneck: {operations} operations taking {duration}",
        ]
    },
    "memory": {
        "facts": [
            "The {entity} is located at {location} with {properties}",
            "{system} uses {technology} version {version} for {purpose}",
            "The {configuration} is set to {value} in {environment}",
            "{service} runs on port {port} with {protocol}",
            "The {credential} for {system} is stored in {location}",
        ],
        "preferences": [
            "User prefers {option1} over {option2} for {use_case}",
            "Always use {tool} when working on {project_type}",
            "Avoid {practice} because {reason}",
            "Default to {approach} unless {exception}",
            "Prioritize {quality1} over {quality2} in {context}",
        ],
        "experiences": [
            "Last time we tried {approach}, it resulted in {outcome}",
            "The {decision} made on {date} led to {consequences}",
            "When working with {technology}, we learned {lesson}",
            "{pattern} has been observed {frequency} times in {context}",
            "Historical data shows {trend} when {condition}",
        ],
        "principles": [
            "Never {forbidden_action} without {safeguard}",
            "Always validate {input_type} before {processing}",
            "Security principle: {security_rule} for {context}",
            "Best practice: {practice} when implementing {feature}",
            "Architectural constraint: {constraint} must be maintained for {reason}",
        ]
    }
}


# Value sets for template substitution
VALUE_SETS = {
    "tech1": ["Redis", "Memcached", "Hazelcast", "Ehcache"],
    "tech2": ["PostgreSQL", "MongoDB", "DynamoDB", "Cassandra"],
    "approach1": ["event-driven architecture", "request-response pattern", "pub-sub messaging"],
    "approach2": ["synchronous API", "batch processing", "streaming pipeline"],
    "problem": ["rate limiting", "session management", "data synchronization", "cache invalidation"],
    "architecture": ["microservices", "monolithic", "serverless", "event-driven"],
    "use_case": ["real-time analytics", "e-commerce platform", "IoT data processing"],
    "feature": ["authentication", "real-time notifications", "file upload", "search functionality"],
    "context": ["production environment", "high-traffic scenario", "distributed system"],
    "system": ["load balancer", "message queue", "cache layer", "API gateway"],
    "condition": ["high load", "network partition", "node failure", "memory pressure"],
    "constraint": ["limited memory", "no external dependencies", "backward compatibility"],
    "challenge": ["database migration", "API versioning", "state management"],
    "limitation": ["single thread", "synchronous operations", "tight coupling"],
    "process": ["deployment", "testing", "code review", "incident response"],
    "assumption": ["network is reliable", "operations are atomic", "latency is low"],
    "issue": ["circular dependency", "memory leak", "deadlock"],
    "environment": ["containerized deployment", "serverless platform", "edge computing"],
    "quality": ["observability", "resilience", "security", "performance"],
    "old_tech": ["REST API", "SQL database", "monolithic backend"],
    "new_tech": ["GraphQL", "NoSQL database", "microservices"],
    "scenario": ["zero downtime", "rollback capability", "data preservation"],
    "component": ["authentication module", "data layer", "API endpoints"],
    "requirement": ["backward compatibility", "existing contracts", "current performance"],
    "option1": ["optimize for reads", "vertical scaling", "eventual consistency"],
    "option2": ["optimize for writes", "horizontal scaling", "strong consistency"],
    "project": ["user service refactoring", "database migration", "API redesign"],
    "timeframe": ["3 months", "6 sprints", "Q2 2025"],
    "implementation": ["JWT authentication", "file upload handler", "password reset flow"],
    "decision": ["using microservices", "choosing NoSQL", "implementing caching"],
    "solution": ["implementing circuit breakers", "adding rate limiting", "using event sourcing"],
    "language": ["Python", "JavaScript", "Go", "Rust"],
    "simple_action": ["reads a file and prints line count", "parses JSON and extracts field"],
    "simple_feature": ["logging", "error handling", "input validation"],
    "config_file": ["config.yaml", "app.toml", ".env"],
    "setting": ["port number", "timeout value", "log level"],
    "simple_bug": ["typo in variable name", "missing import", "incorrect condition"],
    "file": ["main.py", "config.js", "api_handler.go"],
    "simple_test": ["unit test", "integration test", "smoke test"],
    "function": ["calculate_total", "parse_request", "validate_input"],
    "feature1": ["user authentication", "role-based access", "session management"],
    "feature2": ["real-time updates", "offline support", "data sync"],
    "feature3": ["audit logging", "analytics tracking", "error reporting"],
    "complex_feature": ["multi-tenant isolation", "blue-green deployment", "canary releases"],
    "external_system": ["payment gateway", "email service", "analytics platform"],
    "new_requirement": ["multi-region support", "GDPR compliance", "99.99% uptime"],
    "old_behavior": ["single-region", "existing API contracts", "current SLA"],
    "legacy_system": ["monolithic Java app", "Oracle database", "SOAP services"],
    "modern_system": ["microservices on Kubernetes", "PostgreSQL", "REST APIs"],
    "tool1": ["Docker", "Terraform", "Jenkins"],
    "tool2": ["Kubernetes", "Ansible", "GitHub Actions"],
    "tool3": ["Prometheus", "ELK stack", "Datadog"],
    "component1": ["user service", "API gateway", "authentication module"],
    "component2": ["order service", "message queue", "authorization layer"],
    "codebase": ["legacy PHP application", "React frontend", "microservices backend"],
    "technology": ["WebAssembly", "GraphQL", "gRPC"],
    "error": ["500 Internal Server Error", "Connection Timeout", "Out of Memory"],
    "task1": ["database schema design", "API specification", "security review"],
    "task2": ["implementation", "integration testing", "deployment"],
    "task3": ["monitoring setup", "documentation", "production release"],
    "dependency1": ["user authentication", "database setup", "API gateway"],
    "dependency2": ["service mesh configuration", "secret management", "logging infrastructure"],
    "service1": ["user-service", "order-service", "payment-service"],
    "service2": ["notification-service", "analytics-service", "audit-service"],
    "service3": ["inventory-service", "shipping-service", "billing-service"],
    "dependent1": ["mobile app", "admin dashboard", "reporting system"],
    "dependent2": ["third-party integrations", "scheduled jobs", "webhooks"],
    "module": ["data access layer", "business logic", "presentation layer"],
    "consumer1": ["REST API", "GraphQL endpoint", "background workers"],
    "consumer2": ["scheduled tasks", "event handlers", "batch processors"],
    "data_type": ["configuration settings", "user records", "metrics data"],
    "content": ["JSON configuration", "CSV data", "log entries"],
    "location": ["line 42", "top of file", "after imports"],
    "pattern": ["*.log", "temp_*", "*.backup"],
    "directory": ["logs/", "tmp/", "cache/"],
    "source": ["old_module/", "deprecated/", "legacy/"],
    "destination": ["new_module/", "current/", "src/"],
    "tool": ["grep", "jq", "awk", "sed"],
    "action": ["filter JSON", "parse logs", "transform data"],
    "params": ["{pattern: 'error', file: 'app.log'}", "{limit: 100, offset: 0}"],
    "command": ["docker ps", "kubectl get pods", "git status"],
    "output_type": ["JSON", "table", "plain text"],
    "api": ["/users/{id}", "/orders", "/analytics/report"],
    "data": ["user credentials", "order payload", "search query"],
    "response_type": ["success 200", "error 400", "redirect 302"],
    "args": ["user_id=123", "include_deleted=false", "format='json'"],
    "result": ["return value", "side effects", "state changes"],
    "script": ["deploy.sh", "backup.py", "cleanup.js"],
    "metrics": ["execution time", "memory usage", "exit code"],
    "files": ["src/main.py, tests/test_api.py", "config.yaml"],
    "change": ["adding rate limiting", "fixing authentication bug"],
    "branch_name": ["feature/user-auth", "fix/memory-leak", "refactor/api-layer"],
    "base": ["main", "develop", "release/v2.0"],
    "purpose": ["implementing new feature", "bug fix", "performance optimization"],
    "source_branch": ["feature/notifications", "fix/validation", "refactor/db-layer"],
    "target_branch": ["develop", "main", "release/v1.5"],
    "conflicts": ["migration files", "configuration changes", "dependency versions"],
    "commit_hash": ["abc123", "def456", "789xyz"],
    "bug": ["memory leak", "race condition", "null pointer exception"],
    "version": ["v2.1.0", "v3.0.0-beta", "v1.5.2"],
    "validation": ["tests passing", "QA approval", "security scan"],
    "endpoint": ["/api/v1/users", "/health", "/metrics"],
    "method": ["POST", "GET", "PUT", "DELETE"],
    "payload": ["{name: 'test', email: 'test@example.com'}", "{id: 123, status: 'active'}"],
    "api": ["GitHub API", "Stripe API", "Slack API"],
    "criteria": ["created_after=2024-01-01", "status=active", "limit=50"],
    "service": ["webhook endpoint", "event processor", "data aggregator"],
    "status_codes": ["201 Created", "400 Bad Request", "500 Server Error"],
    "websocket": ["ws://events.example.com", "wss://live-feed"],
    "events": ["user.created", "order.completed", "payment.processed"],
    "timeout": ["30 seconds", "2 minutes", "10 retries"],
    "metric": ["95% success rate", "p99 latency 200ms", "throughput 1000 req/s"],
    "result": ["expected output", "improved performance", "reduced errors"],
    "positive_outcome": ["faster response time", "reduced memory usage", "fewer bugs"],
    "negative_outcome": ["performance degradation", "data loss", "service downtime"],
    "failure": ["timeout errors", "data corruption", "service unavailable"],
    "evidence": ["metrics showed degradation", "users reported errors", "tests failed"],
    "sequence_of_actions": ["deploy, test, rollback", "analyze, fix, verify"],
    "future_scenario": ["similar deployments", "comparable features", "related bugs"],
    "observation": ["repeated pattern of errors", "consistent performance issue"],
    "hypothesis": ["memory leak in module X", "race condition in handler"],
    "repeated_behavior": ["failing at scale", "succeeding in isolation", "timing-dependent"],
    "error_detail": ["unexpected indent", "missing parenthesis", "invalid syntax"],
    "line": ["42", "156", "7"],
    "parse_error": ["unexpected EOF", "invalid token", "malformed expression"],
    "detail": ["expected 4 spaces", "tab/space mix", "inconsistent indentation"],
    "token": ["';'", "'}'", "')'"],
    "position": ["column 25", "after function declaration", "before return statement"],
    "element": ["closing brace", "semicolon", "return statement"],
    "construct": ["function definition", "class declaration", "if statement"],
    "exception_type": ["TypeError", "ValueError", "KeyError", "AttributeError"],
    "message": ["Cannot read property 'map' of undefined", "'int' object is not callable"],
    "property": ["length", "map", "id", "name"],
    "object": ["null", "undefined", "response"],
    "index": ["10", "-1", "100"],
    "length": ["5", "0", "50"],
    "calculation": ["total / count", "price * quantity", "average calculation"],
    "recursive_function": ["factorial", "fibonacci", "tree_traversal"],
    "input": ["n=1000000", "depth=10000", "large array"],
    "wrong_result": ["42", "None", "empty array"],
    "expected": ["0", "user object", "[1, 2, 3]"],
    "operation1": ["read", "write", "update"],
    "operation2": ["write", "delete", "read"],
    "symptom": ["data corruption", "inconsistent state", "lost updates"],
    "loop": ["for loop", "while loop", "iterator"],
    "incorrect_behavior": ["skipped last element", "processed element twice"],
    "status_code": ["404", "500", "401", "429"],
    "error_message": ["Not Found", "Internal Server Error", "Unauthorized"],
    "duration": ["30s", "2m", "5m"],
    "host": ["db.example.com", "cache.internal", "api.service"],
    "port": ["5432", "6379", "3306"],
    "auth_error": ["Invalid credentials", "Token expired", "Permission denied"],
    "version1": ["2.0", "3.1", "1.5"],
    "version2": ["1.0", "3.0", "2.0"],
    "network_issue": ["Connection refused", "Timeout", "DNS resolution failed"],
    "rows": ["10000", "1000000", "50"],
    "table": ["users", "orders", "analytics_events"],
    "percentage": ["95", "85", "99"],
    "operation": ["image processing", "data aggregation", "report generation"],
    "load": ["1000 concurrent users", "10k requests/sec", "peak traffic"],
    "operations": ["10000", "50000", "100000"],
    "entity": ["production database", "API server", "message queue"],
    "properties": ["master-slave replication", "3 nodes", "TLS enabled"],
    "protocol": ["HTTPS", "gRPC", "WebSocket"],
    "credential": ["API key", "database password", "OAuth token"],
    "practice": ["storing secrets in code", "bypassing validation", "ignoring errors"],
    "reason": ["security risk", "poor maintainability", "performance impact"],
    "approach": ["caching strategy", "retry logic", "circuit breaker pattern"],
    "exception": ["explicit override", "legacy compatibility", "emergency mode"],
    "quality1": ["security", "performance", "maintainability"],
    "quality2": ["speed of development", "feature richness", "backwards compatibility"],
    "outcome": ["performance issues", "data loss", "user complaints"],
    "date": ["2024-03-15", "Q1 2024", "last sprint"],
    "consequences": ["technical debt", "improved reliability", "reduced costs"],
    "lesson": ["always validate inputs", "monitor resource usage", "test edge cases"],
    "frequency": ["3", "multiple", "occasionally"],
    "trend": ["increasing latency", "decreasing throughput", "growing error rate"],
    "forbidden_action": ["delete production data", "deploy without tests", "commit secrets"],
    "safeguard": ["confirmation prompt", "backup created", "security scan"],
    "input_type": ["user input", "external API data", "file uploads"],
    "processing": ["database insertion", "business logic", "rendering"],
    "security_rule": ["encrypt sensitive data", "validate all inputs", "use HTTPS"],
    "practice": ["use dependency injection", "write unit tests", "document APIs"],
}


@dataclass
class ScalingConfig:
    """Configuration for scaled data generation."""
    ego_model: str = "mlx-community/Qwen2.5-7B-Instruct-8bit"
    output_dir: Path = Path("models/distilled_neurons/training_data_scaled")
    samples_per_domain: int = 500
    temperature_range: Tuple[float, float] = (0.6, 0.9)
    max_tokens: int = 512
    dedup_threshold: float = 0.85  # Cosine similarity threshold for deduplication
    batch_log_interval: int = 50


@dataclass
class GenerationStats:
    """Statistics for a generation run."""
    domain: str
    total_generated: int = 0
    valid_samples: int = 0
    filtered_duplicates: int = 0
    filtered_malformed: int = 0
    scenario_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ScaledDataGenerator:
    """Generates scaled, diverse training data using scenario variations."""

    def __init__(self, config: ScalingConfig):
        self.config = config
        self.ego_model = None
        self.ego_tokenizer = None
        self.stats: Dict[str, GenerationStats] = {}
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ScaledDataGenerator initialized. Output: {config.output_dir}")

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

    def _substitute_template(self, template: str) -> str:
        """Substitute placeholders in template with random values."""
        import random

        result = template
        # Find all placeholders {variable_name}
        placeholders = re.findall(r'\{(\w+)\}', template)

        for placeholder in placeholders:
            if placeholder in VALUE_SETS:
                value = random.choice(VALUE_SETS[placeholder])
                result = result.replace(f"{{{placeholder}}}", value, 1)

        return result

    def _generate_scenario_prompts(self, domain: str, num_samples: int) -> List[Tuple[str, str, str]]:
        """Generate diverse scenario-based prompts for a domain.

        Returns:
            List of (scenario_type, system_prompt, user_prompt) tuples
        """
        import random

        if domain not in SCENARIO_TEMPLATES:
            raise ValueError(f"Unknown domain: {domain}")

        scenarios = SCENARIO_TEMPLATES[domain]
        prompts = []

        # Distribute samples across scenario types
        scenario_types = list(scenarios.keys())
        samples_per_type = num_samples // len(scenario_types)
        remainder = num_samples % len(scenario_types)

        for idx, scenario_type in enumerate(scenario_types):
            # Add remainder to first few types
            count = samples_per_type + (1 if idx < remainder else 0)
            templates = scenarios[scenario_type]

            for i in range(count):
                # Cycle through templates with variation
                template = templates[i % len(templates)]
                user_prompt = self._substitute_template(template)

                # Get base system prompt from original domain config
                from conch_dna.training.distillation import DOMAIN_PROMPTS
                system_prompt = DOMAIN_PROMPTS[domain]["system"]

                prompts.append((scenario_type, system_prompt, user_prompt))

        # Shuffle to mix scenario types
        random.shuffle(prompts)
        return prompts

    def _validate_json_output(self, text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate that output contains valid JSON structure.

        Returns:
            (is_valid, parsed_json) tuple
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if not json_match:
                return False, None

            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # Basic validation: should be a dict with at least one key
            if not isinstance(parsed, dict) or len(parsed) == 0:
                return False, None

            return True, parsed
        except (json.JSONDecodeError, AttributeError):
            return False, None

    def _compute_text_hash(self, text: str) -> str:
        """Compute hash of text for exact duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity score between two texts.

        Uses a combination of:
        - Exact hash matching (returns 1.0)
        - Jaccard similarity of word sets
        """
        hash1 = self._compute_text_hash(text1)
        hash2 = self._compute_text_hash(text2)

        if hash1 == hash2:
            return 1.0

        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _is_duplicate(self, text: str, seen_outputs: List[str]) -> bool:
        """Check if text is too similar to previously seen outputs."""
        for seen in seen_outputs:
            similarity = self._compute_similarity(text, seen)
            if similarity >= self.config.dedup_threshold:
                return True
        return False

    def generate_scaled_data(
        self,
        domain: str,
        num_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], GenerationStats]:
        """Generate scaled training data with scenario variations.

        Args:
            domain: The cognitive domain to generate data for
            num_samples: Number of samples to generate (default: config value)

        Returns:
            (samples, statistics) tuple
        """
        self.load_ego()

        num_samples = num_samples or self.config.samples_per_domain
        stats = GenerationStats(domain=domain)
        self.stats[domain] = stats

        logger.info(f"Generating {num_samples} scaled samples for {domain}")

        # Generate scenario-based prompts
        prompts = self._generate_scenario_prompts(domain, num_samples)

        samples = []
        seen_outputs: List[str] = []
        seen_hashes: Set[str] = set()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        import random

        for idx, (scenario_type, system_prompt, user_prompt) in enumerate(prompts):
            stats.total_generated += 1
            stats.scenario_distribution[scenario_type] += 1

            # Vary temperature for diversity
            temp = random.uniform(*self.config.temperature_range)
            sampler = make_sampler(temp=temp)

            # Format prompt
            prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant
"""

            try:
                response = generate(
                    self.ego_model,
                    self.ego_tokenizer,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    sampler=sampler,
                    verbose=False
                )

                # Validate JSON structure
                is_valid, parsed_json = self._validate_json_output(response)
                if not is_valid:
                    stats.filtered_malformed += 1
                    logger.debug(f"Sample {idx+1} filtered: malformed JSON")
                    continue

                # Check for duplicates
                response_hash = self._compute_text_hash(response)
                if response_hash in seen_hashes:
                    stats.filtered_duplicates += 1
                    logger.debug(f"Sample {idx+1} filtered: exact duplicate")
                    continue

                if self._is_duplicate(response, seen_outputs):
                    stats.filtered_duplicates += 1
                    logger.debug(f"Sample {idx+1} filtered: similar duplicate")
                    continue

                # Valid, unique sample
                samples.append({
                    "input": user_prompt,
                    "output": response,
                    "system": system_prompt,
                    "domain": domain,
                    "scenario_type": scenario_type,
                    "temperature": round(temp, 2),
                    "parsed_json": parsed_json
                })

                seen_outputs.append(response)
                seen_hashes.add(response_hash)
                stats.valid_samples += 1

                if (idx + 1) % self.config.batch_log_interval == 0:
                    logger.info(
                        f"  Progress: {idx+1}/{num_samples} generated, "
                        f"{stats.valid_samples} valid, "
                        f"{stats.filtered_duplicates} duplicates, "
                        f"{stats.filtered_malformed} malformed"
                    )

            except Exception as e:
                logger.warning(f"Generation failed for sample {idx+1}: {e}")
                stats.filtered_malformed += 1
                continue

        logger.info(f"Generation complete for {domain}:")
        logger.info(f"  Total generated: {stats.total_generated}")
        logger.info(f"  Valid samples: {stats.valid_samples}")
        logger.info(f"  Filtered duplicates: {stats.filtered_duplicates}")
        logger.info(f"  Filtered malformed: {stats.filtered_malformed}")
        logger.info(f"  Scenario distribution: {dict(stats.scenario_distribution)}")

        return samples, stats

    def save_training_data(self, samples: List[Dict[str, str]], domain: str) -> Path:
        """Save training data in JSONL format with metadata.

        Args:
            samples: List of training samples
            domain: Domain name

        Returns:
            Path to saved JSONL file
        """
        output_path = self.config.output_dir / f"{domain}_training.jsonl"

        with open(output_path, 'w') as f:
            for sample in samples:
                # Format for training
                training_text = f"""<|im_start|>system
{sample['system']}
<|im_end|>
<|im_start|>user
{sample['input']}
<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""

                # Include metadata for analysis
                record = {
                    "text": training_text,
                    "metadata": {
                        "domain": sample["domain"],
                        "scenario_type": sample["scenario_type"],
                        "temperature": sample["temperature"]
                    }
                }
                f.write(json.dumps(record) + '\n')

        logger.info(f"Saved {len(samples)} training samples to {output_path}")
        return output_path

    def save_statistics(self, domain: str) -> Path:
        """Save generation statistics for analysis.

        Args:
            domain: Domain name

        Returns:
            Path to statistics JSON file
        """
        if domain not in self.stats:
            raise ValueError(f"No statistics available for domain: {domain}")

        stats = self.stats[domain]
        stats_path = self.config.output_dir / f"{domain}_statistics.json"

        stats_data = {
            "domain": stats.domain,
            "total_generated": stats.total_generated,
            "valid_samples": stats.valid_samples,
            "filtered_duplicates": stats.filtered_duplicates,
            "filtered_malformed": stats.filtered_malformed,
            "scenario_distribution": dict(stats.scenario_distribution),
            "quality_rate": round(stats.valid_samples / stats.total_generated, 3) if stats.total_generated > 0 else 0,
            "duplicate_rate": round(stats.filtered_duplicates / stats.total_generated, 3) if stats.total_generated > 0 else 0
        }

        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")
        return stats_path

    def generate_all_domains(self) -> Dict[str, Tuple[Path, Path]]:
        """Generate scaled data for all domains.

        Returns:
            Dict mapping domain -> (data_path, stats_path)
        """
        domains = list(SCENARIO_TEMPLATES.keys())
        results = {}

        logger.info(f"Starting scaled generation for {len(domains)} domains")

        for domain in domains:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing domain: {domain.upper()}")
            logger.info(f"{'='*60}")

            try:
                samples, stats = self.generate_scaled_data(domain)
                data_path = self.save_training_data(samples, domain)
                stats_path = self.save_statistics(domain)
                results[domain] = (data_path, stats_path)
            except Exception as e:
                logger.error(f"Failed to generate data for {domain}: {e}")
                continue

        self._save_summary(results)
        return results

    def _save_summary(self, results: Dict[str, Tuple[Path, Path]]) -> Path:
        """Save overall generation summary.

        Args:
            results: Dict of domain -> (data_path, stats_path)

        Returns:
            Path to summary file
        """
        summary_path = self.config.output_dir / "generation_summary.json"

        summary = {
            "timestamp": mx.random.seed(42) and str(Path(__file__).stat().st_mtime),
            "config": {
                "ego_model": self.config.ego_model,
                "samples_per_domain": self.config.samples_per_domain,
                "temperature_range": self.config.temperature_range,
                "dedup_threshold": self.config.dedup_threshold
            },
            "domains": {}
        }

        for domain, (data_path, stats_path) in results.items():
            if domain in self.stats:
                stats = self.stats[domain]
                summary["domains"][domain] = {
                    "data_file": str(data_path),
                    "stats_file": str(stats_path),
                    "valid_samples": stats.valid_samples,
                    "quality_rate": round(stats.valid_samples / stats.total_generated, 3) if stats.total_generated > 0 else 0
                }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nGeneration summary saved to {summary_path}")
        return summary_path


def generate_scaled_data(
    domain: str,
    num_samples: int = 500,
    output_dir: Optional[Path] = None
) -> Tuple[Path, GenerationStats]:
    """Convenience function to generate scaled data for a single domain.

    Args:
        domain: Cognitive domain to generate data for
        num_samples: Number of samples to generate
        output_dir: Optional output directory (default: standard location)

    Returns:
        (data_path, statistics) tuple
    """
    config = ScalingConfig(
        samples_per_domain=num_samples,
        output_dir=output_dir or Path("models/distilled_neurons/training_data_scaled")
    )

    generator = ScaledDataGenerator(config)
    samples, stats = generator.generate_scaled_data(domain, num_samples)
    data_path = generator.save_training_data(samples, domain)
    generator.save_statistics(domain)

    return data_path, stats


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

        logger.info(f"Generating {num_samples} samples for {domain}")
        data_path, stats = generate_scaled_data(domain, num_samples)
        logger.info(f"Complete! Data saved to {data_path}")
    else:
        # Generate for all domains
        config = ScalingConfig(samples_per_domain=500)
        generator = ScaledDataGenerator(config)
        results = generator.generate_all_domains()

        logger.info("\n" + "="*60)
        logger.info("ALL DOMAINS COMPLETE")
        logger.info("="*60)
        for domain, (data_path, stats_path) in results.items():
            logger.info(f"{domain}: {data_path}")

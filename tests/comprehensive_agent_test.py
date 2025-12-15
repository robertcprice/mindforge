"""
Comprehensive Agent Capability Test Suite

Tests:
1. Complex multi-step reasoning
2. Thought depth and coherence
3. Learning and memory persistence
4. Tool usage decision-making
5. Task decomposition
6. Error recovery and adaptation
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from conch.memory.store import MemoryStore, MemoryType
from conch.agent.task_list import PersistentTaskList, TaskStatus, TaskPriority, InternalTask
from conch.tools.shell import ShellTool
from conch.tools.filesystem import FileSystemTool
from conch.tools.web import WebTool
from conch.tools.code import CodeTool
from conch.integrations.ollama import OllamaTool
import httpx


class SimpleOllamaInference:
    """Simple wrapper for Ollama API calls."""

    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.client = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))

    def generate(self, prompt: str, max_tokens: int = -1) -> str:
        """Generate a response from the model."""
        try:
            response = self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=300.0
            )
            if response.status_code == 200:
                data = response.json()
                # Qwen3 model outputs thinking in 'thinking' field, then actual response in 'response'
                # If response is empty but thinking exists, use thinking (model ran out of tokens)
                result = data.get("response", "")
                thinking = data.get("thinking", "")

                # Combine both - the thinking often has good analysis
                if result and thinking:
                    return f"{thinking}\n\n{result}"
                elif result:
                    return result
                elif thinking:
                    return thinking
                return ""
            return f"Error: {response.status_code} - {response.text}"
        except httpx.TimeoutException as e:
            return f"Timeout Error: Request took too long"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    score: float  # 0-1
    details: str
    duration: float
    artifacts: Dict[str, Any] = None


class AgentCapabilityTester:
    """Comprehensive test suite for agent capabilities."""

    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path("data/test_memories.db")
        self.memory_store = MemoryStore(db_path)
        self.task_list = PersistentTaskList(self.memory_store)
        self.inference = SimpleOllamaInference(model="qwen3:8b")
        self.results: List[TestResult] = []

        # Initialize tools
        self.shell = ShellTool()
        self.filesystem = FileSystemTool()
        self.web = WebTool()
        self.code = CodeTool()
        self.ollama = OllamaTool()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all capability tests."""
        print("\n" + "="*60)
        print("  COMPREHENSIVE AGENT CAPABILITY TEST SUITE")
        print("="*60 + "\n")

        start_time = time.time()

        # Run each test category
        self.test_thought_complexity()
        self.test_task_decomposition()
        self.test_tool_selection()
        self.test_multi_step_reasoning()
        self.test_learning_and_memory()
        self.test_error_handling()

        total_time = time.time() - start_time

        # Generate summary
        return self.generate_report(total_time)

    def test_thought_complexity(self):
        """Test 1: Evaluate depth and coherence of agent thoughts."""
        print("\n[TEST 1] Thought Complexity Analysis")
        print("-" * 40)

        start = time.time()

        # Generate thoughts on complex topics
        prompts = [
            "Reflect on how you might approach debugging a race condition in a distributed system.",
            "Consider the trade-offs between consistency and availability in system design.",
            "Think about how you would explain recursion to someone who only knows loops.",
        ]

        scores = []
        thoughts = []

        for prompt in prompts:
            response = self.inference.generate(prompt, max_tokens=-1)
            thought = response.strip()
            thoughts.append(thought)

            # Analyze thought quality
            score = self._analyze_thought_quality(thought)
            scores.append(score)
            print(f"  Prompt: {prompt}")
            print(f"  Response length: {len(thought)} chars")
            print(f"  Quality score: {score:.2f}")
            print()

        avg_score = sum(scores) / len(scores) if scores else 0

        self.results.append(TestResult(
            name="Thought Complexity",
            passed=avg_score >= 0.5,
            score=avg_score,
            details=f"Generated {len(thoughts)} thoughts, avg quality: {avg_score:.2f}",
            duration=time.time() - start,
            artifacts={"thoughts": thoughts, "scores": scores}
        ))

        print(f"  ✓ Average thought quality: {avg_score:.2f}")

    def _analyze_thought_quality(self, thought: str) -> float:
        """Analyze thought quality based on multiple criteria."""
        score = 0.0

        # Length check (not too short, not rambling)
        word_count = len(thought.split())
        if 20 <= word_count <= 200:
            score += 0.2
        elif 10 <= word_count <= 300:
            score += 0.1

        # Structure check (paragraphs, sentences)
        sentences = thought.count('.') + thought.count('!') + thought.count('?')
        if sentences >= 2:
            score += 0.2

        # Coherence indicators
        coherence_words = ['because', 'therefore', 'however', 'first', 'then',
                          'finally', 'consider', 'approach', 'would', 'could']
        coherence_count = sum(1 for w in coherence_words if w in thought.lower())
        score += min(coherence_count * 0.1, 0.3)

        # Technical depth
        technical_words = ['system', 'process', 'function', 'data', 'algorithm',
                          'optimize', 'implement', 'analyze', 'design', 'pattern']
        tech_count = sum(1 for w in technical_words if w in thought.lower())
        score += min(tech_count * 0.05, 0.3)

        return min(score, 1.0)

    def test_task_decomposition(self):
        """Test 2: Evaluate ability to break complex tasks into subtasks."""
        print("\n[TEST 2] Task Decomposition")
        print("-" * 40)

        start = time.time()

        # Complex task to decompose
        complex_task = """
        Build a web scraper that:
        1. Fetches data from multiple news sites
        2. Extracts article titles, dates, and content
        3. Stores results in a structured format
        4. Handles rate limiting and errors gracefully
        5. Generates a summary report
        """

        prompt = f"""Given this complex task, break it down into 3-5 specific, actionable subtasks:

Task: {complex_task}

List each subtask on its own line, starting with a number. Each subtask should be
specific enough to complete in one focused work session."""

        response = self.inference.generate(prompt, max_tokens=-1)

        # Count subtasks identified
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        subtasks = [l for l in lines if l[0].isdigit() or l.startswith('-')]

        print(f"  Complex task given: {complex_task}")
        print(f"  Subtasks identified: {len(subtasks)}")
        for st in subtasks:
            print(f"    • {st}")

        # Score based on decomposition quality
        score = 0.0
        if 3 <= len(subtasks) <= 7:
            score += 0.4
        elif len(subtasks) > 0:
            score += 0.2

        # Check for actionable language
        action_words = ['create', 'build', 'implement', 'fetch', 'extract',
                       'store', 'handle', 'generate', 'test', 'validate']
        action_count = sum(1 for st in subtasks
                         for w in action_words if w in st.lower())
        score += min(action_count * 0.1, 0.4)

        # Check for specificity
        if any(word in response.lower() for word in ['api', 'html', 'json', 'database', 'file']):
            score += 0.2

        self.results.append(TestResult(
            name="Task Decomposition",
            passed=score >= 0.5,
            score=min(score, 1.0),
            details=f"Decomposed into {len(subtasks)} subtasks",
            duration=time.time() - start,
            artifacts={"subtasks": subtasks, "raw_response": response}
        ))

        print(f"  ✓ Decomposition score: {score:.2f}")

    def test_tool_selection(self):
        """Test 3: Evaluate tool selection decision-making."""
        print("\n[TEST 3] Tool Selection Decision-Making")
        print("-" * 40)

        start = time.time()

        # Scenarios requiring different tools
        scenarios = [
            {
                "task": "Check what files are in the current directory",
                "expected_tools": ["shell", "filesystem"],
                "wrong_tools": ["web", "ollama"]
            },
            {
                "task": "Find out the current weather in New York",
                "expected_tools": ["web"],
                "wrong_tools": ["shell", "filesystem", "code"]
            },
            {
                "task": "Analyze the structure of a Python file",
                "expected_tools": ["code", "filesystem"],
                "wrong_tools": ["web", "n8n"]
            },
            {
                "task": "List available AI models on this machine",
                "expected_tools": ["ollama"],
                "wrong_tools": ["web", "filesystem"]
            },
        ]

        correct = 0
        total = len(scenarios)

        for scenario in scenarios:
            prompt = f"""Given this task, which tool would be most appropriate?

Task: {scenario['task']}

Available tools:
- shell: Execute shell commands (ls, pwd, date, etc.)
- filesystem: Read/write files and list directories
- web: Fetch web pages and search the internet
- code: Analyze code structure and syntax
- ollama: Query local AI models
- n8n: Workflow automation

Respond with ONLY the tool name (e.g., "shell" or "web")."""

            response = self.inference.generate(prompt, max_tokens=-1)
            selected = response.strip().lower().split()[0] if response.strip() else ""

            # Remove any punctuation
            selected = ''.join(c for c in selected if c.isalnum())

            is_correct = selected in scenario['expected_tools']
            is_wrong = selected in scenario['wrong_tools']

            if is_correct:
                correct += 1
                print(f"  ✓ '{scenario['task']}' → {selected}")
            else:
                print(f"  ✗ '{scenario['task']}' → {selected} (expected: {scenario['expected_tools']})")

        score = correct / total if total > 0 else 0

        self.results.append(TestResult(
            name="Tool Selection",
            passed=score >= 0.6,
            score=score,
            details=f"{correct}/{total} correct tool selections",
            duration=time.time() - start
        ))

        print(f"  ✓ Tool selection accuracy: {score:.0%}")

    def test_multi_step_reasoning(self):
        """Test 4: Evaluate multi-step reasoning capabilities."""
        print("\n[TEST 4] Multi-Step Reasoning")
        print("-" * 40)

        start = time.time()

        # Complex reasoning problem
        problem = """
        A server is experiencing intermittent slowdowns. Here's what we know:
        - CPU usage spikes to 95% every 15 minutes
        - Memory usage is stable at 60%
        - Disk I/O shows brief spikes during CPU spikes
        - The application logs show "GC pause" messages during slowdowns
        - The server has 8GB RAM and the Java heap is set to 6GB

        What is likely causing this issue and what steps would you take to fix it?
        """

        prompt = f"""Analyze this problem step by step:

{problem}

Provide your reasoning in numbered steps, then give your conclusion and recommended actions."""

        response = self.inference.generate(prompt, max_tokens=-1)

        # Analyze reasoning quality
        score = 0.0

        # Check for step-by-step structure
        has_steps = any(c.isdigit() and response[response.index(c):response.index(c)+2] in
                       ['1.', '1)', '1:', '2.', '2)', '2:']
                       for c in response if c.isdigit())
        if has_steps:
            score += 0.2

        # Check for causal reasoning
        causal_words = ['because', 'therefore', 'causing', 'leads to', 'result',
                       'indicates', 'suggests', 'likely', 'probably']
        causal_count = sum(1 for w in causal_words if w in response.lower())
        score += min(causal_count * 0.1, 0.3)

        # Check for correct identification of GC issue
        if 'gc' in response.lower() or 'garbage' in response.lower():
            score += 0.2
        if 'heap' in response.lower() or 'memory' in response.lower():
            score += 0.1

        # Check for actionable recommendations
        action_words = ['reduce', 'increase', 'tune', 'adjust', 'monitor', 'configure']
        if any(w in response.lower() for w in action_words):
            score += 0.2

        print(f"  Problem: Server performance issue (GC-related)")
        print(f"  Response length: {len(response)} chars")
        print(f"  Has step structure: {has_steps}")
        print(f"  Identified GC issue: {'gc' in response.lower() or 'garbage' in response.lower()}")

        self.results.append(TestResult(
            name="Multi-Step Reasoning",
            passed=score >= 0.5,
            score=min(score, 1.0),
            details=f"Reasoning analysis score: {score:.2f}",
            duration=time.time() - start,
            artifacts={"problem": problem, "response": response}
        ))

        print(f"  ✓ Reasoning score: {score:.2f}")

    def test_learning_and_memory(self):
        """Test 5: Evaluate learning and memory persistence."""
        print("\n[TEST 5] Learning and Memory")
        print("-" * 40)

        start = time.time()

        # Store some information
        test_facts = [
            ("project_name", "Conch Consciousness Engine"),
            ("primary_model", "qwen3:8b"),
            ("test_timestamp", datetime.now().isoformat()),
        ]

        # Store in memory
        for key, value in test_facts:
            from conch.memory.store import Memory
            mem = Memory(
                content=f"{key}: {value}",
                memory_type=MemoryType.FACT,
                importance=0.8,
                tags=[key, "test"]
            )
            self.memory_store.store(mem)

        # Retrieve and verify
        retrieved = 0
        for key, expected_value in test_facts:
            results = self.memory_store.search(key, limit=5)
            if results and expected_value in results[0].content:
                retrieved += 1

        # Test task persistence
        task = self.task_list.add_task(
            "Test task for memory persistence",
            priority=TaskPriority.HIGH
        )

        task_retrieved = False
        if task:
            retrieved_task = self.task_list.get_task(task.id)
            task_retrieved = retrieved_task is not None

        score = (retrieved / len(test_facts)) * 0.7 + (0.3 if task_retrieved else 0)

        print(f"  Facts stored: {len(test_facts)}")
        print(f"  Facts retrieved: {retrieved}")
        print(f"  Task persistence: {'✓' if task_retrieved else '✗'}")

        self.results.append(TestResult(
            name="Learning and Memory",
            passed=score >= 0.7,
            score=score,
            details=f"Retrieved {retrieved}/{len(test_facts)} facts, task: {'yes' if task_retrieved else 'no'}",
            duration=time.time() - start
        ))

        print(f"  ✓ Memory score: {score:.2f}")

    def test_error_handling(self):
        """Test 6: Evaluate error handling and recovery."""
        print("\n[TEST 6] Error Handling and Recovery")
        print("-" * 40)

        start = time.time()

        # Test tool error handling
        errors_handled = 0
        total_error_tests = 3

        # Test 1: Invalid file path
        try:
            result = self.filesystem.execute(operation="read", path="/nonexistent/path/file.txt")
            if "error" in str(result).lower() or "not found" in str(result).lower():
                errors_handled += 1
                print("  ✓ Handled missing file gracefully")
            else:
                print("  ✗ Missing file not handled properly")
        except Exception as e:
            errors_handled += 1
            print(f"  ✓ Caught missing file exception: {type(e).__name__}")

        # Test 2: Invalid shell command
        try:
            result = self.shell.execute(command="nonexistent_command_xyz123")
            if "error" in str(result).lower() or "not found" in str(result).lower():
                errors_handled += 1
                print("  ✓ Handled invalid command gracefully")
            else:
                print("  ✗ Invalid command not handled properly")
        except Exception as e:
            errors_handled += 1
            print(f"  ✓ Caught invalid command exception: {type(e).__name__}")

        # Test 3: Invalid operation
        try:
            result = self.filesystem.execute(operation="invalid_op", path=".")
            if "error" in str(result).lower() or "invalid" in str(result).lower() or "unknown" in str(result).lower():
                errors_handled += 1
                print("  ✓ Handled invalid operation gracefully")
            else:
                print(f"  ✗ Invalid operation result: {str(result)}")
        except Exception as e:
            errors_handled += 1
            print(f"  ✓ Caught invalid operation exception: {type(e).__name__}")

        score = errors_handled / total_error_tests

        self.results.append(TestResult(
            name="Error Handling",
            passed=score >= 0.6,
            score=score,
            details=f"{errors_handled}/{total_error_tests} errors handled gracefully",
            duration=time.time() - start
        ))

        print(f"  ✓ Error handling score: {score:.0%}")

    def generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("  TEST RESULTS SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        overall_score = sum(r.score for r in self.results) / total if total > 0 else 0

        print(f"\n  Tests Passed: {passed}/{total}")
        print(f"  Overall Score: {overall_score:.1%}")
        print(f"  Total Time: {total_time:.1f}s")
        print()

        print("  Individual Results:")
        print("  " + "-"*50)

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} | {result.name:<25} | {result.score:.0%} | {result.duration:.1f}s")

        print()

        # Capability assessment
        print("  Capability Assessment:")
        print("  " + "-"*50)

        categories = {
            "Reasoning": ["Thought Complexity", "Multi-Step Reasoning"],
            "Planning": ["Task Decomposition"],
            "Tool Usage": ["Tool Selection", "Error Handling"],
            "Memory": ["Learning and Memory"]
        }

        for cat, tests in categories.items():
            cat_scores = [r.score for r in self.results if r.name in tests]
            cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            level = "Strong" if cat_avg >= 0.7 else "Moderate" if cat_avg >= 0.5 else "Needs Work"
            print(f"  {cat:<15} : {cat_avg:.0%} ({level})")

        print("\n" + "="*60 + "\n")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "passed": passed,
                "total": total,
                "overall_score": overall_score,
                "duration": total_time
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "duration": r.duration
                }
                for r in self.results
            ]
        }

        # Save report
        report_path = Path("data/test_reports")
        report_path.mkdir(exist_ok=True)
        report_file = report_path / f"capability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Report saved to: {report_file}")

        return report


def main():
    """Run the comprehensive test suite."""
    tester = AgentCapabilityTester()
    report = tester.run_all_tests()
    return report


if __name__ == "__main__":
    main()

"""
MindForge DNA - Main Consciousness Loop

The consciousness loop orchestrates all layers:
1. WAKE: Check signals, restore state
2. SENSE: Gather context, check needs
3. THINK: Generate thoughts via neurons or EGO
4. ACT: Execute actions through tools
5. REFLECT: Assess outcomes, generate reflections
6. SLEEP: Decide duration, save state

EGO runs on EVERY cycle for first 10,000 cycles.
After that, neurons handle most work with EGO fallback.
"""

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Signals file for external control
SIGNAL_FILE = Path("/tmp/mindforge_signal")


@dataclass
class CycleState:
    """State for a single consciousness cycle."""

    cycle_number: int
    start_time: datetime
    thoughts: List[str]
    actions: List[Dict[str, Any]]
    reflections: List[str]
    mood: str
    reward: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ConsciousnessLoop:
    """The main consciousness loop orchestrating all layers.

    Architecture:
        SUPEREGO (immutable) - Values, Safety, KVRM
            ↓
        EGO (Qwen3-8B) - Personality DNA, Teacher, Corrector
            ↓
        CORTEX (6 neurons) - Specialized cognitive functions
            ↓
        ID (pure math) - Needs/drives regulator

    Cycle phases:
        1. WAKE: Check signals, restore state
        2. SENSE: Gather context, update needs
        3. THINK: Generate structured thought
        4. ACT: Execute actions if needed
        5. REFLECT: Assess and learn
        6. SLEEP: Decide duration, save state

    External control:
        echo "wake" > /tmp/mindforge_signal  # Immediate wake
        echo "sleep 600" > /tmp/mindforge_signal  # Force 10-min sleep
        echo "die" > /tmp/mindforge_signal  # Shutdown
    """

    # Constants
    BOOTSTRAP_CYCLES = 10000  # EGO runs on every cycle during bootstrap
    MIN_SLEEP = 15           # Minimum 15 seconds
    MAX_SLEEP = 1800         # Maximum 30 minutes

    def __init__(
        self,
        data_dir: Path = Path("data"),
        config_path: Optional[Path] = None
    ):
        """Initialize the consciousness loop.

        Args:
            data_dir: Directory for all data
            config_path: Optional config file path
        """
        self.data_dir = data_dir
        self.config_path = config_path or data_dir / "config.yaml"

        # State
        self.cycle_count = 0
        self.running = False
        self.current_state: Optional[CycleState] = None

        # Components (lazy loaded)
        self._needs_regulator = None
        self._superego = None
        self._ego = None
        self._cortex = {}
        self._memory_store = None
        self._training_pipeline = None

        logger.info(f"ConsciousnessLoop initialized (data_dir={data_dir})")

    @property
    def is_bootstrap_phase(self) -> bool:
        """Whether we're in bootstrap phase (EGO runs every cycle)."""
        return self.cycle_count < self.BOOTSTRAP_CYCLES

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing consciousness components...")

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "training").mkdir(exist_ok=True)
        (self.data_dir / "adapters").mkdir(exist_ok=True)

        # Initialize ID layer (needs regulator)
        from .id.needs import create_regulator
        self._needs_regulator = create_regulator("balanced")
        logger.info("ID layer initialized")

        # Initialize Superego
        try:
            from .superego.values import CoreValues
            from .superego.safety import SafetyChecker
            from .superego.kvrm import KVRMRouter
            self._superego = {
                "values": CoreValues(),
                "safety": SafetyChecker(),
                "kvrm": KVRMRouter(db_path=self.data_dir / "facts.db")
            }
            logger.info("Superego layer initialized")
        except ImportError as e:
            logger.warning(f"Superego components not fully available: {e}")
            self._superego = {}

        # Initialize EGO (the personality DNA)
        try:
            from .ego.model import EgoModel
            self._ego = EgoModel()
            logger.info("EGO model initialized")
        except ImportError as e:
            logger.warning(f"EGO model not available: {e}")
            self._ego = None

        # Initialize Memory Store
        try:
            from .memory.store import create_memory_store
            self._memory_store = create_memory_store(str(self.data_dir))
            logger.info("Memory store initialized")
        except ImportError as e:
            logger.warning(f"Memory store not available: {e}")
            self._memory_store = None

        # Initialize Training Pipeline
        try:
            from .training.pipeline import create_training_pipeline
            self._training_pipeline = create_training_pipeline(
                str(self.data_dir / "training")
            )
            logger.info("Training pipeline initialized")
        except ImportError as e:
            logger.warning(f"Training pipeline not available: {e}")
            self._training_pipeline = None

        # Load previous state if exists
        self._load_state()

        logger.info(f"Initialization complete (cycle_count={self.cycle_count})")

    def _load_state(self) -> None:
        """Load previous state from disk."""
        state_path = self.data_dir / "state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.cycle_count = state.get("cycle_count", 0)
                logger.info(f"Loaded state: cycle_count={self.cycle_count}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save current state to disk."""
        state_path = self.data_dir / "state.json"
        state = {
            "cycle_count": self.cycle_count,
            "last_update": datetime.now().isoformat()
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _check_signal(self) -> Optional[str]:
        """Check for external control signals."""
        if not SIGNAL_FILE.exists():
            return None

        try:
            signal_content = SIGNAL_FILE.read_text().strip()
            SIGNAL_FILE.unlink()  # Remove after reading
            logger.info(f"Received signal: {signal_content}")
            return signal_content
        except Exception as e:
            logger.error(f"Error reading signal: {e}")
            return None

    def _handle_signal(self, sig: str) -> Optional[int]:
        """Handle an external signal.

        Returns:
            Optional sleep duration override, or None
        """
        if sig == "wake":
            return 0  # No sleep
        elif sig == "die":
            logger.info("Shutdown signal received")
            self.running = False
            return None
        elif sig.startswith("sleep "):
            try:
                duration = int(sig.split()[1])
                return min(self.MAX_SLEEP, max(self.MIN_SLEEP, duration))
            except:
                return None
        return None

    def _phase_wake(self) -> CycleState:
        """WAKE phase: Initialize cycle state."""
        self.cycle_count += 1
        state = CycleState(
            cycle_number=self.cycle_count,
            start_time=datetime.now(),
            thoughts=[],
            actions=[],
            reflections=[],
            mood="neutral"
        )
        logger.info(f"=== CYCLE {self.cycle_count} WAKE ===")
        return state

    def _phase_sense(self, state: CycleState) -> Dict[str, Any]:
        """SENSE phase: Gather context and update needs."""
        context = {
            "cycle": self.cycle_count,
            "is_bootstrap": self.is_bootstrap_phase,
            "timestamp": datetime.now().isoformat(),
        }

        # Get needs state
        if self._needs_regulator:
            guidance = self._needs_regulator._generate_guidance()
            context["needs"] = self._needs_regulator.get_state()
            context["dominant_need"] = guidance["dominant_need"]
            context["suggested_focus"] = guidance["suggested_focus"]

        # Get recent memories
        if self._memory_store:
            recent = self._memory_store.get_recent(limit=5)
            context["recent_memories"] = [m.content[:100] for m in recent]

        logger.debug(f"Context: {json.dumps(context, indent=2)}")
        return context

    def _phase_think(self, state: CycleState, context: Dict[str, Any]) -> str:
        """THINK phase: Generate structured thought."""
        thought = ""

        # During bootstrap, always use EGO
        if self.is_bootstrap_phase and self._ego:
            logger.debug("Bootstrap phase: using EGO for thinking")
            # Build prompt from context
            prompt = f"Cycle {self.cycle_count}: {context.get('suggested_focus', 'Reflect on current state')}"
            thought = self._ego.generate(
                prompt=prompt,
                cycle_count=self.cycle_count,
                mood=state.mood,
                dominant_need=str(context.get('dominant_need', 'CURIOSITY'))
            )

        # After bootstrap, try neurons first
        elif "think" in self._cortex:
            neuron = self._cortex["think"]
            output = neuron.infer(context)

            if output.should_fallback and self._ego:
                logger.debug("Neuron fallback: using EGO")
                prompt = f"Think about: {context.get('suggested_focus', 'current situation')}"
                thought = self._ego.generate(
                    prompt=prompt,
                    cycle_count=self.cycle_count,
                    mood=state.mood,
                    dominant_need=str(context.get('dominant_need', 'CURIOSITY'))
                )

                # Record for training
                if self._training_pipeline:
                    self._training_pipeline.record_fallback(
                        domain="think",
                        input_text=str(context),
                        ego_output=thought,
                        fallback_reason="low_confidence"
                    )
            else:
                thought = output.content
        else:
            # No thinking capability
            thought = f"Cycle {self.cycle_count}: Observing and processing..."

        state.thoughts.append(thought)
        logger.info(f"Thought: {thought[:100]}...")
        return thought

    def _phase_act(self, state: CycleState, thought: str) -> List[Dict[str, Any]]:
        """ACT phase: Execute actions if needed."""
        actions = []

        # Check if thought suggests an action
        if "action" in self._cortex or self._ego:
            # Extract action from thought
            # This is simplified - real implementation would parse action commands
            pass

        state.actions = actions
        return actions

    def _phase_reflect(self, state: CycleState) -> str:
        """REFLECT phase: Assess outcomes and generate reflection."""
        reflection = ""

        # Calculate reward based on cycle outcomes
        reward = 0.5  # Base reward
        if not state.errors:
            reward += 0.2
        if state.thoughts:
            reward += 0.1
        if state.actions:
            reward += 0.2

        state.reward = min(1.0, reward)

        # Generate reflection
        if "reflect" in self._cortex:
            neuron = self._cortex["reflect"]
            output = neuron.infer({
                "cycle_events": state.thoughts + [str(a) for a in state.actions],
                "needs_state": self._needs_regulator.get_state() if self._needs_regulator else {},
                "outcomes": {"success": not state.errors, "reward": state.reward}
            })
            reflection = output.content
            state.mood = output.metadata.get("mood", "neutral")
        else:
            reflection = f"Cycle {state.cycle_number} complete with reward {state.reward:.2f}"

        state.reflections.append(reflection)

        # Update needs based on outcome
        if self._needs_regulator:
            if state.reward > 0.7:
                self._needs_regulator.process_event("task_completed")
            elif state.reward < 0.4:
                self._needs_regulator.process_event("task_failed")

        # Store in memory
        if self._memory_store and reflection:
            self._memory_store.store(
                content=reflection,
                memory_type="reflection",
                importance=state.reward * 0.8
            )

        logger.info(f"Reflection: {reflection[:100]}...")
        return reflection

    def _phase_sleep(self, state: CycleState) -> int:
        """SLEEP phase: Decide sleep duration."""
        # Check for override signal
        sig = self._check_signal()
        if sig:
            override = self._handle_signal(sig)
            if override is not None:
                return override

        # Let EGO decide sleep duration
        if self._ego:
            timing = self._ego.decide_next_wakeup({
                "needs": self._needs_regulator.get_state() if self._needs_regulator else {},
                "mood": state.mood,
                "last_reward": state.reward,
                "cycle_count": self.cycle_count
            })
            duration = timing.wake_in_seconds
        else:
            # Heuristic: sleep longer when satisfied, shorter when urgent
            max_urgency = 0.5
            if self._needs_regulator:
                max_urgency = self._needs_regulator.get_max_urgency()

            if max_urgency > 0.85:
                duration = self.MIN_SLEEP
            elif state.reward > 0.8:
                duration = 60 * 5  # 5 minutes when doing well
            else:
                duration = 60  # 1 minute default

        duration = max(self.MIN_SLEEP, min(self.MAX_SLEEP, duration))
        logger.info(f"Sleep duration: {duration}s")
        return duration

    def run_cycle(self) -> CycleState:
        """Run a single consciousness cycle."""
        # WAKE
        state = self._phase_wake()
        self.current_state = state

        try:
            # SENSE
            context = self._phase_sense(state)

            # THINK
            thought = self._phase_think(state, context)

            # ACT
            actions = self._phase_act(state, thought)

            # REFLECT
            reflection = self._phase_reflect(state)

        except Exception as e:
            logger.error(f"Cycle error: {e}")
            state.errors.append(str(e))

        # Save state
        self._save_state()

        # Check training
        if self._training_pipeline:
            stats = self._training_pipeline.get_stats()
            for domain, info in stats.items():
                if info["ready_to_retrain"]:
                    logger.info(f"Domain {domain} ready for retraining")

        return state

    def run(self) -> None:
        """Run the main consciousness loop."""
        self.running = True
        logger.info("Starting consciousness loop...")

        # Signal handlers
        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        while self.running:
            try:
                # Run cycle
                state = self.run_cycle()

                # Sleep
                sleep_duration = self._phase_sleep(state)

                if not self.running:
                    break

                logger.info(f"=== SLEEP for {sleep_duration}s ===")
                time.sleep(sleep_duration)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)  # Brief pause before retry

        # Cleanup
        self._save_state()
        if self._memory_store:
            self._memory_store.close()
        if self._training_pipeline:
            self._training_pipeline.save_all()

        logger.info("Consciousness loop terminated")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MindForge DNA Consciousness")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory path")
    parser.add_argument("--cycles", type=int, default=0,
                       help="Number of cycles to run (0 = infinite)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run
    loop = ConsciousnessLoop(data_dir=Path(args.data_dir))
    loop.initialize()

    if args.cycles > 0:
        for _ in range(args.cycles):
            state = loop.run_cycle()
            print(f"Cycle {state.cycle_number}: reward={state.reward:.2f}, mood={state.mood}")
    else:
        loop.run()


if __name__ == "__main__":
    main()

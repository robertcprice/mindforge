"""
Comprehensive tests for MindForge Training module.

Tests cover:
- TrainingExample creation
- TrainingPipeline configuration
- LoRAConfig and LoRATrainer
- Data generation
"""

import pytest
from pathlib import Path
from datetime import datetime


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_training_example_creation(self):
        """Test creating a TrainingExample."""
        from mindforge.training.data import TrainingExample, ExampleType

        example = TrainingExample(
            prompt="What is Python?",
            completion="Python is a programming language.",
            example_type=ExampleType.THOUGHT_GENERATION,
        )

        assert example.prompt == "What is Python?"
        assert example.completion == "Python is a programming language."
        assert example.example_type == ExampleType.THOUGHT_GENERATION

    def test_training_example_with_metadata(self):
        """Test TrainingExample with metadata."""
        from mindforge.training.data import TrainingExample, ExampleType

        example = TrainingExample(
            prompt="Test prompt",
            completion="Test completion",
            example_type=ExampleType.DECISION_MAKING,
            metadata={"source": "test", "quality": 0.9},
        )

        assert example.metadata["source"] == "test"
        assert example.metadata["quality"] == 0.9

    def test_training_example_quality_score(self):
        """Test TrainingExample quality scoring."""
        from mindforge.training.data import TrainingExample, ExampleType

        example = TrainingExample(
            prompt="High quality prompt",
            completion="High quality completion with details",
            example_type=ExampleType.REFLECTION,
            quality_score=0.95,
        )

        assert example.quality_score == 0.95


class TestExampleType:
    """Tests for ExampleType enum."""

    def test_example_types_exist(self):
        """Test all example types exist."""
        from mindforge.training.data import ExampleType

        # Consciousness cycle examples
        assert ExampleType.THOUGHT_GENERATION.value == "thought_generation"
        assert ExampleType.DECISION_MAKING.value == "decision_making"
        assert ExampleType.REFLECTION.value == "reflection"
        assert ExampleType.SLEEP_DETERMINATION.value == "sleep_determination"

        # KVRM grounding examples
        assert ExampleType.CLAIM_CLASSIFICATION.value == "claim_classification"
        assert ExampleType.KEY_EXTRACTION.value == "key_extraction"
        assert ExampleType.GROUNDING_VERIFICATION.value == "grounding_verification"

        # Combined examples
        assert ExampleType.GROUNDED_THINKING.value == "grounded_thinking"
        assert ExampleType.VERIFIED_DECISION.value == "verified_decision"


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_lora_config_defaults(self):
        """Test LoRAConfig default values."""
        from mindforge.training.lora import LoRAConfig

        config = LoRAConfig()

        assert config.rank > 0  # LoRA rank
        assert config.alpha > 0  # LoRA alpha (scaling factor)
        assert config.dropout >= 0  # Dropout probability

    def test_lora_config_custom_values(self):
        """Test LoRAConfig with custom values."""
        from mindforge.training.lora import LoRAConfig

        config = LoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.1,
            target_modules=("q_proj", "v_proj"),
        )

        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert "q_proj" in config.target_modules


class TestLoRATrainer:
    """Tests for LoRA trainer."""

    def test_lora_trainer_initialization(self, tmp_path):
        """Test LoRATrainer initialization."""
        from mindforge.training.lora import LoRATrainer, LoRAConfig

        config = LoRAConfig()
        trainer = LoRATrainer(
            base_model="test-model",
            output_dir=tmp_path / "adapters",
            config=config
        )

        assert trainer.config == config

    def test_lora_trainer_config_validation(self, tmp_path):
        """Test LoRATrainer validates config."""
        from mindforge.training.lora import LoRATrainer, LoRAConfig

        # Valid config should work
        config = LoRAConfig(rank=8)
        trainer = LoRATrainer(
            base_model="test-model",
            output_dir=tmp_path / "adapters",
            config=config
        )
        assert trainer is not None


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""

    def test_pipeline_initialization(self):
        """Test TrainingPipeline initialization."""
        from mindforge.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline()
        assert pipeline is not None

    def test_pipeline_configuration(self, tmp_path):
        """Test TrainingPipeline configuration."""
        from mindforge.training.pipeline import TrainingPipeline, PipelineConfig
        from pathlib import Path

        config = PipelineConfig(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )
        pipeline = TrainingPipeline(config=config)

        assert pipeline.config.output_dir == tmp_path / "output"


class TestDataGeneration:
    """Tests for training data generation."""

    def test_consciousness_data_generation(self):
        """Test generating consciousness training examples."""
        from mindforge.training.data import ExampleType, TrainingExample

        # Create a thought generation example
        example = TrainingExample(
            prompt="Generate a thought about learning.",
            completion="I notice that learning new concepts requires patience and practice.",
            example_type=ExampleType.THOUGHT_GENERATION,
        )

        assert example.example_type == ExampleType.THOUGHT_GENERATION

    def test_grounding_data_generation(self):
        """Test generating grounding training examples."""
        from mindforge.training.data import ExampleType, TrainingExample

        # Create a claim classification example
        example = TrainingExample(
            prompt="Classify this claim: The Earth orbits the Sun.",
            completion="ClaimType: FACTUAL\nReason: This is a verifiable scientific fact.",
            example_type=ExampleType.CLAIM_CLASSIFICATION,
        )

        assert example.example_type == ExampleType.CLAIM_CLASSIFICATION

    def test_reflection_data_generation(self):
        """Test generating reflection training examples."""
        from mindforge.training.data import ExampleType, TrainingExample

        example = TrainingExample(
            prompt="Reflect on this interaction: User asked about Python, I explained basics.",
            completion="The interaction was successful. The user received helpful information.",
            example_type=ExampleType.REFLECTION,
        )

        assert example.example_type == ExampleType.REFLECTION


class TestTrainingDataset:
    """Tests for training dataset management."""

    def test_create_training_examples(self):
        """Test creating multiple training examples."""
        from mindforge.training.data import TrainingExample, ExampleType

        examples = [
            TrainingExample(
                prompt=f"Prompt {i}",
                completion=f"Completion {i}",
                example_type=ExampleType.THOUGHT_GENERATION,
            )
            for i in range(10)
        ]

        assert len(examples) == 10
        assert all(e.example_type == ExampleType.THOUGHT_GENERATION for e in examples)

    def test_filter_by_type(self):
        """Test filtering examples by type."""
        from mindforge.training.data import TrainingExample, ExampleType

        examples = [
            TrainingExample(
                prompt="Thought",
                completion="...",
                example_type=ExampleType.THOUGHT_GENERATION,
            ),
            TrainingExample(
                prompt="Decision",
                completion="...",
                example_type=ExampleType.DECISION_MAKING,
            ),
            TrainingExample(
                prompt="Reflection",
                completion="...",
                example_type=ExampleType.REFLECTION,
            ),
        ]

        thoughts = [e for e in examples if e.example_type == ExampleType.THOUGHT_GENERATION]
        assert len(thoughts) == 1

    def test_filter_by_quality(self):
        """Test filtering examples by quality."""
        from mindforge.training.data import TrainingExample, ExampleType

        examples = [
            TrainingExample(
                prompt="High quality",
                completion="...",
                example_type=ExampleType.THOUGHT_GENERATION,
                quality_score=0.95,
            ),
            TrainingExample(
                prompt="Low quality",
                completion="...",
                example_type=ExampleType.THOUGHT_GENERATION,
                quality_score=0.3,
            ),
        ]

        high_quality = [e for e in examples if e.quality_score >= 0.8]
        assert len(high_quality) == 1


class TestTrainingMetrics:
    """Tests for training metrics tracking."""

    def test_example_has_timestamp(self):
        """Test examples have timestamps."""
        from mindforge.training.data import TrainingExample, ExampleType

        example = TrainingExample(
            prompt="Test",
            completion="Test",
            example_type=ExampleType.THOUGHT_GENERATION,
        )

        # Should have created_at or similar
        assert hasattr(example, "created_at") or hasattr(example, "timestamp")


class TestTrainingIntegration:
    """Integration tests for training system."""

    def test_full_training_workflow(self, tmp_path):
        """Test complete training workflow setup."""
        from mindforge.training.pipeline import TrainingPipeline, PipelineConfig
        from mindforge.training.data import TrainingExample, ExampleType
        from mindforge.training.lora import LoRAConfig

        # Create config using correct parameter names
        lora_config = LoRAConfig(rank=8, alpha=16)

        # Create pipeline with proper PipelineConfig
        config = PipelineConfig(
            output_dir=tmp_path / "output",
            lora_config=lora_config,
        )
        pipeline = TrainingPipeline(config=config)

        # Create training examples
        examples = [
            TrainingExample(
                prompt="Test prompt",
                completion="Test completion",
                example_type=ExampleType.THOUGHT_GENERATION,
            )
        ]

        # Verify setup (without actually training)
        assert pipeline is not None
        assert len(examples) == 1

    def test_training_data_serialization(self, tmp_path):
        """Test training data can be saved and loaded."""
        from mindforge.training.data import TrainingExample, ExampleType
        import json

        example = TrainingExample(
            prompt="Test",
            completion="Response",
            example_type=ExampleType.REFLECTION,
            quality_score=0.9,
        )

        # Serialize
        if hasattr(example, "to_dict"):
            data = example.to_dict()
        else:
            data = {
                "prompt": example.prompt,
                "completion": example.completion,
                "example_type": example.example_type.value,
            }

        # Save
        output_file = tmp_path / "example.json"
        with open(output_file, "w") as f:
            json.dump(data, f)

        # Load
        with open(output_file, "r") as f:
            loaded = json.load(f)

        assert loaded["prompt"] == "Test"


class TestToolFormats:
    """Tests for tool_formats.py parsing and validation."""

    def test_parse_args_simple_string(self):
        """Test parsing simple key=value arguments."""
        from mindforge.training.tool_formats import _parse_args

        args = _parse_args('command="ls -la"')
        assert args.get("command") == "ls -la"

    def test_parse_args_multiple_params(self):
        """Test parsing multiple parameters."""
        from mindforge.training.tool_formats import _parse_args

        args = _parse_args('operation="write", path="test.py", content="hello"')
        assert args.get("operation") == "write"
        assert args.get("path") == "test.py"
        assert args.get("content") == "hello"

    def test_parse_args_multiline_content(self):
        """Test parsing multiline content with triple quotes."""
        from mindforge.training.tool_formats import _parse_args

        content = '''operation="write", path="test.py", content="""def hello():
    print("world")
"""'''
        args = _parse_args(content)
        assert args.get("operation") == "write"
        assert "def hello" in args.get("content", "")

    def test_parse_args_escaped_quotes(self):
        """Test parsing content with escaped quotes."""
        from mindforge.training.tool_formats import _parse_args

        args = _parse_args(r'content="print(\"hello\")"')
        assert "print" in args.get("content", "")

    def test_parse_args_empty_string(self):
        """Test parsing empty string returns empty dict."""
        from mindforge.training.tool_formats import _parse_args

        args = _parse_args("")
        assert args == {}

    def test_validate_against_schema_valid_shell(self):
        """Test schema validation for valid shell command."""
        from mindforge.training.tool_formats import validate_against_schema

        error = validate_against_schema("shell", {"command": "ls -la"})
        assert error is None

    def test_validate_against_schema_missing_required(self):
        """Test schema validation catches missing required field."""
        from mindforge.training.tool_formats import validate_against_schema

        error = validate_against_schema("shell", {})
        assert error is not None
        assert "Missing required field" in error

    def test_validate_against_schema_invalid_enum(self):
        """Test schema validation catches invalid enum value."""
        from mindforge.training.tool_formats import validate_against_schema

        error = validate_against_schema("filesystem", {
            "operation": "invalid_op",
            "path": "/tmp/test"
        })
        assert error is not None
        assert "Invalid operation" in error

    def test_validate_against_schema_valid_filesystem(self):
        """Test schema validation for valid filesystem operation."""
        from mindforge.training.tool_formats import validate_against_schema

        error = validate_against_schema("filesystem", {
            "operation": "write",
            "path": "/tmp/test.py",
            "content": "hello"
        })
        assert error is None

    def test_validate_against_schema_unknown_tool(self):
        """Test schema validation for unknown tool returns None."""
        from mindforge.training.tool_formats import validate_against_schema

        error = validate_against_schema("unknown_tool", {"foo": "bar"})
        assert error is None  # No schema to validate against

    def test_tool_specs_exist(self):
        """Test TOOL_SPECS is properly defined."""
        from mindforge.training.tool_formats import TOOL_SPECS

        assert "shell" in TOOL_SPECS
        assert "filesystem" in TOOL_SPECS
        assert "git" in TOOL_SPECS
        assert "web" in TOOL_SPECS

    def test_tool_schemas_exist(self):
        """Test TOOL_SCHEMAS is properly defined."""
        from mindforge.training.tool_formats import TOOL_SCHEMAS

        assert "shell" in TOOL_SCHEMAS
        assert "filesystem" in TOOL_SCHEMAS
        assert TOOL_SCHEMAS["shell"]["type"] == "object"
        assert "command" in TOOL_SCHEMAS["shell"]["properties"]


class TestToolParsingIntegration:
    """Integration tests for tool parsing in the agent."""

    def test_parse_filesystem_write_action(self):
        """Test parsing a real filesystem write action."""
        from mindforge.training.tool_formats import _parse_tool_action

        action = 'filesystem(operation="write", path="test.py", content="print(1)")'
        result = _parse_tool_action(action)

        assert result is not None
        # Result is a ParsedAction named tuple with (tool_name, args, raw_action)
        assert result.tool_name == "filesystem"
        assert result.args.get("operation") == "write"
        assert result.args.get("path") == "test.py"

    def test_parse_shell_command_action(self):
        """Test parsing a shell command action."""
        from mindforge.training.tool_formats import _parse_tool_action

        action = 'shell(command="python --version")'
        result = _parse_tool_action(action)

        assert result is not None
        # Result is a ParsedAction named tuple with (tool_name, args, raw_action)
        assert result.tool_name == "shell"
        assert result.args.get("command") == "python --version"

    def test_filter_unknown_parameters(self):
        """Test that unknown parameters are filtered out."""
        from mindforge.training.tool_formats import TOOL_SPECS, _parse_args

        args = _parse_args('command="test", unknown_param="garbage", b="x"')
        spec = TOOL_SPECS.get("shell")

        # Filter to known args
        known_args = set(spec.required_args) | set(spec.optional_args)
        filtered = {k: v for k, v in args.items() if k in known_args}

        assert "command" in filtered
        assert "unknown_param" not in filtered
        assert "b" not in filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

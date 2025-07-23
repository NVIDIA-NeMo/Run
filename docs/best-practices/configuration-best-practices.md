---
description: "Advanced best practices for NeMo Run configurations in production ML environments, focusing on scalability, maintainability, and performance."
categories: ["best-practices"]
tags: ["configuration", "type-safety", "validation", "maintainability", "production", "scalability"]
personas: ["mle-focused", "admin-focused"]
difficulty: "advanced"
content_type: "best-practices"
modality: "text-only"
---

(configuration-best-practices)=

# Configuration Best Practices

This guide covers advanced best practices for creating production-ready, scalable configurations in NeMo Run.

## Advanced Type Safety Patterns

### Comprehensive Validation

```python
from typing import List, Optional, Union, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import nemo_run as run

class ModelType(Enum):
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"

class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

@dataclass
class ModelConfig:
    model_type: ModelType
    hidden_size: int
    num_layers: int
    num_heads: int = 8
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "swish"] = "gelu"

    def __post_init__(self):
        """Comprehensive validation for model configuration."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        # Model-specific validation
        if self.model_type == ModelType.TRANSFORMER:
            if self.num_heads > self.hidden_size:
                raise ValueError("num_heads cannot exceed hidden_size")

@dataclass
class TrainingConfig:
    optimizer: OptimizerType
    learning_rate: float
    batch_size: int
    epochs: int
    warmup_steps: int = 0
    gradient_clipping: float = 1.0
    scheduler: Optional[Literal["cosine", "linear", "step"]] = None

    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")

        if self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be positive")
```

### Configuration Factories with Validation

```python
def create_llm_config(
    model_size: Literal["7b", "13b", "70b"],
    seq_length: int = 4096,
    use_flash_attention: bool = True
) -> run.Config:
    """Create validated LLM configuration."""

    # Size-specific configurations
    configs = {
        "7b": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
        "13b": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40},
        "70b": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64}
    }

    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")

    base_config = configs[model_size]

    # Validate sequence length
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    # Validate flash attention compatibility
    if use_flash_attention and seq_length > 8192:
        print("Warning: Flash attention may not work with seq_length > 8192")

    return run.Config(
        create_llm_model,
        model_config=ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=base_config["hidden_size"],
            num_layers=base_config["num_layers"],
            num_heads=base_config["num_heads"],
            activation="gelu"
        ),
        seq_length=seq_length,
        use_flash_attention=use_flash_attention
    )
```

## Advanced Configuration Patterns

### Hierarchical Configuration Management

```python
@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: Dict[str, Any]
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Cross-validation between components."""
        # Validate batch size compatibility
        if hasattr(self.training, 'batch_size') and hasattr(self.data, 'max_length'):
            estimated_memory = self.training.batch_size * self.data['max_length'] * 4
            if estimated_memory > 32 * 1024**3:  # 32GB
                print("Warning: High memory usage detected")

        # Validate model-training compatibility
        if self.model.model_type == ModelType.TRANSFORMER:
            if self.training.batch_size % 8 != 0:
                print("Warning: Batch size should be divisible by 8 for optimal performance")

def create_experiment_config(
    model_size: str,
    batch_size: int,
    learning_rate: float,
    **kwargs
) -> run.Config:
    """Create complete experiment configuration."""

    # Create validated components
    model_config = create_llm_config(model_size)
    training_config = TrainingConfig(
        optimizer=OptimizerType.ADAMW,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=kwargs.get('epochs', 100)
    )

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        model=model_config,
        training=training_config,
        data=kwargs.get('data', {}),
        environment=kwargs.get('environment', {}),
        metadata={
            'model_size': model_size,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    )

    return run.Config(
        run_experiment,
        config=experiment_config
    )
```

### Configuration Composition and Inheritance

```python
@dataclass
class BaseConfig:
    """Base configuration with common parameters."""
    seed: int = 42
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("invalid log_level")

@dataclass
class ProductionConfig(BaseConfig):
    """Production-specific configuration."""
    checkpoint_dir: str = "/checkpoints"
    log_dir: str = "/logs"
    metrics_dir: str = "/metrics"

    def __post_init__(self):
        super().__post_init__()
        # Ensure directories exist
        for directory in [self.checkpoint_dir, self.log_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)

def create_production_config(
    model_size: str,
    **kwargs
) -> run.Config:
    """Create production-ready configuration."""

    base_config = ProductionConfig(**kwargs)

    return run.Config(
        run_production_experiment,
        model_config=create_llm_config(model_size),
        base_config=base_config
    )
```

## Performance Optimization

### Lazy Configuration Loading

```python
from functools import lru_cache
import json

@lru_cache(maxsize=128)
def load_config_template(template_name: str) -> Dict[str, Any]:
    """Cache configuration templates for performance."""
    template_path = f"configs/templates/{template_name}.json"
    with open(template_path, 'r') as f:
        return json.load(f)

def create_config_from_template(
    template_name: str,
    **overrides
) -> run.Config:
    """Create configuration from cached template."""

    template = load_config_template(template_name)
    config_data = {**template, **overrides}

    return run.Config(
        run_experiment,
        **config_data
    )
```

### Configuration Broadcasting

```python
def broadcast_config_changes(
    base_config: run.Config,
    **changes
) -> run.Config:
    """Efficiently apply changes across nested configurations."""

    def apply_changes(config_dict: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively apply changes to nested dictionaries."""
        result = config_dict.copy()

        for key, value in changes.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = apply_changes(result[key], value)
            else:
                result[key] = value

        return result

    # Convert config to dict, apply changes, then back to config
    config_dict = base_config.kwargs.copy()
    updated_dict = apply_changes(config_dict, changes)

    return run.Config(
        base_config.fn,
        **updated_dict
    )

# Usage
base_config = create_llm_config("7b")
updated_config = broadcast_config_changes(
    base_config,
    model_config={"hidden_size": 8192},
    training_config={"batch_size": 128}
)
```

## Error Handling and Debugging

### Comprehensive Error Handling

```python
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def safe_config_build(config: run.Config):
    """Safely build configuration with comprehensive error handling."""

    try:
        logger.info(f"Building configuration: {config}")
        result = fdl.build(config)
        logger.info("Configuration built successfully")
        yield result

    except Exception as e:
        logger.error(f"Configuration build failed: {e}")
        logger.error(f"Configuration details: {config}")

        # Provide helpful error messages
        if "hidden_size" in str(e):
            logger.error("Hint: Check that hidden_size is positive and compatible with num_heads")
        elif "batch_size" in str(e):
            logger.error("Hint: Check that batch_size is positive and compatible with your hardware")
        elif "learning_rate" in str(e):
            logger.error("Hint: Check that learning_rate is positive and reasonable")

        raise

def validate_config_compatibility(configs: List[run.Config]) -> bool:
    """Validate compatibility between multiple configurations."""

    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs[i+1:], i+1):
            try:
                # Check for conflicts
                built1 = fdl.build(config1)
                built2 = fdl.build(config2)

                # Add compatibility checks here
                if hasattr(built1, 'batch_size') and hasattr(built2, 'batch_size'):
                    if built1.batch_size != built2.batch_size:
                        logger.warning(f"Batch size mismatch between configs {i} and {j}")

            except Exception as e:
                logger.error(f"Compatibility check failed between configs {i} and {j}: {e}")
                return False

    return True
```

## Production Best Practices

### Configuration Versioning

```python
from datetime import datetime
import hashlib

@dataclass
class VersionedConfig:
    """Configuration with version tracking."""
    config: run.Config
    version: str
    created_at: str
    hash: str

    @classmethod
    def create(cls, config: run.Config, version: str = "1.0.0") -> "VersionedConfig":
        """Create versioned configuration."""
        created_at = datetime.now().isoformat()

        # Create hash of configuration
        config_str = str(config.kwargs)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

        return cls(
            config=config,
            version=version,
            created_at=created_at,
            hash=config_hash
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize versioned configuration."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "hash": self.hash,
            "config": self.config.kwargs
        }

# Usage
versioned_config = VersionedConfig.create(
    create_llm_config("7b"),
    version="2.1.0"
)
```

### Configuration Testing

```python
import pytest

def test_config_validation():
    """Test configuration validation."""

    # Test valid configuration
    valid_config = create_llm_config("7b")
    with safe_config_build(valid_config):
        pass  # Should not raise

    # Test invalid configuration
    with pytest.raises(ValueError):
        create_llm_config("invalid_size")

    # Test boundary conditions
    with pytest.raises(ValueError):
        create_llm_config("7b", seq_length=-1)

def test_config_compatibility():
    """Test configuration compatibility."""

    configs = [
        create_llm_config("7b"),
        create_llm_config("13b"),
        create_llm_config("70b")
    ]

    assert validate_config_compatibility(configs)
```

## Summary

These advanced patterns ensure your NeMo Run configurations are:

1. **Type-Safe**: Comprehensive validation and type checking
2. **Maintainable**: Clear structure and documentation
3. **Scalable**: Efficient composition and inheritance
4. **Robust**: Comprehensive error handling and debugging
5. **Production-Ready**: Versioning, testing, and monitoring support

Follow these patterns to create enterprise-grade ML experiment configurations that scale with your team and project complexity.

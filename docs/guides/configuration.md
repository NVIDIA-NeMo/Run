---
description: "Advanced configuration patterns for NeMo Run experiments, focusing on type-safe configurations and complex parameter management for AI developers."
categories: ["guides"]
tags: ["configuration", "fiddle", "type-safety", "advanced-patterns"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "text-only"
---

# Configuration Guide

This guide covers advanced configuration patterns for NeMo Run experiments, focusing on type-safe configurations and complex parameter management.

## Core Configuration Patterns

### Type-Safe Configuration with `run.Config`

```python
import nemo_run as run
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    architecture: str = "transformer"
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    warmup_steps: int = 1000
    gradient_clipping: float = 1.0

# Direct configuration with validation
config = run.Config(
    train_model,
    model_config=ModelConfig(hidden_size=768, num_layers=24),
    training_config=TrainingConfig(learning_rate=2e-4, batch_size=64)
)
```

### Lazy Configuration with `run.Partial`

```python
# Partial configuration with CLI integration
train_fn = run.Partial(
    train_model,
    model_config=ModelConfig(hidden_size=1024),
    training_config=TrainingConfig(learning_rate=1e-4)
)

# CLI usage: python train.py --learning_rate=2e-4 --batch_size=128
```

### Script-Based Configuration

```python
# External script execution
script_config = run.Script(
    "train_script.py",
    env={"CUDA_VISIBLE_DEVICES": "0,1", "WANDB_PROJECT": "llm_training"},
    cwd="/path/to/project"
)
```

## Advanced Configuration Patterns

### Nested Configuration Hierarchies

```python
@dataclass
class DataConfig:
    dataset_path: str
    tokenizer_path: str
    max_length: int = 512
    num_workers: int = 4

@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    seed: int = 42

# Complex nested configuration
experiment_config = run.Config(
    run_experiment,
    config=ExperimentConfig(
        model=ModelConfig(hidden_size=1024, num_layers=24),
        training=TrainingConfig(learning_rate=1e-4, batch_size=64),
        data=DataConfig(
            dataset_path="/data/llm_dataset",
            tokenizer_path="/models/tokenizer"
        )
    )
)
```

### Configuration Factories

```python
def create_llm_config(model_size: str = "7b") -> run.Config:
    """Factory function for LLM configurations."""

    configs = {
        "7b": ModelConfig(hidden_size=4096, num_layers=32, num_heads=32),
        "13b": ModelConfig(hidden_size=5120, num_layers=40, num_heads=40),
        "70b": ModelConfig(hidden_size=8192, num_layers=80, num_heads=64)
    }

    return run.Config(
        train_llm,
        model_config=configs[model_size],
        training_config=TrainingConfig(batch_size=32)
    )

# Usage
config_7b = create_llm_config("7b")
config_70b = create_llm_config("70b")
```

### Configuration Validation

```python
from typing import Union, Dict, Any

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""

    # Model validation
    if config.get("hidden_size", 0) <= 0:
        raise ValueError("hidden_size must be positive")

    if config.get("num_heads", 0) <= 0:
        raise ValueError("num_heads must be positive")

    # Training validation
    if config.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")

    if config.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")

    return True

# Validated configuration
validated_config = run.Config(
    train_with_validation,
    model_config=ModelConfig(hidden_size=768),
    training_config=TrainingConfig(learning_rate=1e-4),
    validator=validate_config
)
```

## Configuration Management

### Configuration Broadcasting

```python
# Apply changes across nested structures
def broadcast_config(base_config: run.Config, **overrides) -> run.Config:
    """Broadcast configuration changes."""

    # Deep copy and apply overrides
    new_config = run.Config(
        base_config.fn,
        **{**base_config.kwargs, **overrides}
    )

    return new_config

# Usage
base_config = run.Config(train_model, batch_size=32)
large_batch_config = broadcast_config(base_config, batch_size=128)
```

### Configuration Diffing

```python
import fiddle as fdl

def diff_configs(config1: run.Config, config2: run.Config) -> Dict[str, Any]:
    """Compare two configurations."""

    built1 = fdl.build(config1)
    built2 = fdl.build(config2)

    differences = {}
    for key in built1.__dict__:
        if getattr(built1, key) != getattr(built2, key):
            differences[key] = {
                "old": getattr(built1, key),
                "new": getattr(built2, key)
            }

    return differences
```

### Multi-Format Export

```python
import yaml
import json

def export_config(config: run.Config, format: str = "yaml") -> str:
    """Export configuration to different formats."""

    built_config = fdl.build(config)
    config_dict = built_config.__dict__

    if format == "yaml":
        return yaml.dump(config_dict, default_flow_style=False)
    elif format == "json":
        return json.dumps(config_dict, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

# Usage
yaml_config = export_config(config, "yaml")
json_config = export_config(config, "json")
```

## CLI Integration

### Advanced CLI Patterns

```python
@run.cli.entrypoint
def train_llm(
    model_size: str = "7b",
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 100,
    data_path: str = "/data/dataset",
    output_dir: str = "/output"
):
    """Train a language model with CLI integration."""

    config = run.Config(
        train_model,
        model_config=create_llm_config(model_size),
        training_config=TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        ),
        data_path=data_path,
        output_dir=output_dir
    )

    run.run(config)

# CLI usage:
# python train.py --model_size=70b --learning_rate=2e-4 --batch_size=64
```

### Configuration Overrides

```python
# Nested configuration overrides
@run.cli.entrypoint
def train_with_overrides(
    model_hidden_size: int = 768,
    model_num_layers: int = 12,
    training_learning_rate: float = 1e-4,
    training_batch_size: int = 32
):
    """Train with nested configuration overrides."""

    config = run.Config(
        train_model,
        model_config=ModelConfig(
            hidden_size=model_hidden_size,
            num_layers=model_num_layers
        ),
        training_config=TrainingConfig(
            learning_rate=training_learning_rate,
            batch_size=training_batch_size
        )
    )

    run.run(config)

# CLI usage:
# python train.py --model_hidden_size=1024 --training_learning_rate=2e-4
```

### Simple CLI Integration

```python
@run.cli.entrypoint
def train_cli(
    model_size: str = "medium",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 20
):
    """Train model with CLI parameters."""

    config = create_training_config(model_size, learning_rate, batch_size)
    config.kwargs["epochs"] = epochs

    result = run.run(config)
    print(f"Training completed! Final loss: {result['final_loss']:.4f}")

# CLI usage: python train.py --model_size=large --learning_rate=0.0005
```

## Best Practices

### Configuration Organization

1. **Use Data Classes**: Define clear configuration structures with validation
2. **Factory Functions**: Create reusable configuration factories
3. **Validation**: Implement comprehensive validation logic
4. **Documentation**: Document configuration parameters and constraints
5. **Versioning**: Version your configurations for reproducibility

### Performance Considerations

1. **Lazy Evaluation**: Use `run.Partial` for expensive configurations
2. **Caching**: Cache built configurations when appropriate
3. **Minimal Dependencies**: Keep configurations lightweight
4. **Type Safety**: Leverage type hints for better IDE support

### Error Handling

```python
def safe_config_build(config: run.Config) -> Any:
    """Safely build configuration with error handling."""

    try:
        return fdl.build(config)
    except Exception as e:
        print(f"Configuration build failed: {e}")
        print(f"Configuration: {config}")
        raise
```

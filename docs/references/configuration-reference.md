---
description: "Comprehensive reference for NeMo Run's configuration system, types, validation, and advanced patterns."
tags: ["configuration", "reference", "types", "validation", "fiddle"]
categories: ["references"]
personas: ["machine-learning-engineer-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "text-only"
---

# Configuration Reference

Comprehensive reference documentation for NeMo Run's type-safe configuration system, covering all configuration types, validation rules, serialization patterns, and advanced configuration techniques.

## Overview

NeMo Run's configuration system provides type-safe configurations with Fiddle integration and built-in serialization, enabling reproducible and validated experiment setups across varied computing environments.

## Core Configuration Types

Use these building blocks to describe what to run and how to run it. `run.Config` declares fully bound, type‑checked objects; `run.Partial` defers some arguments until execution time; and `run.Script` wraps shell or Python entrypoints for non‑Python or hybrid workflows. Pick the simplest type that preserves type safety and reproducibility for your use case.

### `run.Config`

The primary configuration type for creating type-safe, validated configurations:

```python
import nemo_run as run

# Basic configuration
config = run.Config(
    MyModel,
    hidden_size=512,
    num_layers=6,
    dropout=0.1
)

# Nested configurations
config = run.Config(
    TrainingPipeline,
    model=run.Config(TransformerModel, hidden_size=768),
    optimizer=run.Config(AdamOptimizer, lr=0.001),
    data=run.Config(DataLoader, batch_size=32)
)
```

### `run.Partial`

Creates partial configurations for delayed execution:

```python
# Partial configuration
partial_config = run.Partial(
    train_model,
    model_name="gpt2",
    learning_rate=0.001,
    batch_size=32
)

# Execute later
run.run(partial_config)
```

### `run.Script`

Configure raw scripts or inline commands:

```python
# File-based script
script = run.Script(path="train.py", args=["--epochs", "100"], env={"PYTHONUNBUFFERED":"1"})

# Inline script
script = run.Script(
    inline="""echo Hello && python -c "print('ok')""" ,
    entrypoint="bash"
)

# Use -m with python entrypoint
script = run.Script(path="my.module", entrypoint="python", m=True)
```

## Configuration Validation

Validation ensures your configurations are safe, consistent, and executable before jobs are launched. NeMo Run validates types from function signatures and lets you add domain‑specific checks, so bad inputs fail early with actionable errors.

### Type Validation

NeMo Run automatically validates types based on function signatures:

```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1
    activation: str = "relu"

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"
    scheduler: Optional[str] = None

def create_training_config(
    model: ModelConfig,
    training: TrainingConfig,
    seed: int = 42,
    debug: bool = False
) -> run.Config:
    """Create a validated training configuration."""

    # Validation
    if model.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    if training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if training.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if training.epochs <= 0:
        raise ValueError("epochs must be positive")

    if not 0 <= model.dropout <= 1:
        raise ValueError("dropout must be between 0 and 1")

    return run.Config(
        train_model,
        model=model,
        training=training,
        seed=seed,
        debug=debug
    )
```

### Custom Validation Helpers

Create custom validation functions:

```python
from typing import Any, Callable

def validate_positive(value: Any, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within a range."""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

def validate_choice(value: str, choices: List[str], name: str) -> None:
    """Validate that a value is one of the allowed choices."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")

@dataclass
class ValidatedConfig:
    model_name: str
    hidden_size: int
    learning_rate: float
    optimizer: str

    def __post_init__(self):
        """Apply validation after initialization."""
        validate_positive(self.hidden_size, "hidden_size")
        validate_range(self.learning_rate, 1e-6, 1.0, "learning_rate")
        validate_choice(self.optimizer, ["adam", "sgd", "adamw"], "optimizer")
```

## Serialization

NeMo Run includes a serializer that round‑trips configurations across YAML, JSON, and TOML without losing structure or type information. Prefer the built‑in serializer for portability and reproducibility, and layer custom handling only when a value cannot be represented as plain data.

### Built-in Serialization

`ConfigSerializer` supports YAML, JSON, and TOML via a YAML-first pipeline (JSON/TOML <-> YAML <-> `Buildable`):

```python
import nemo_run as run
from nemo_run.cli.config import ConfigSerializer

# Create configuration
config = run.Config(
    train_model,
    model_name="gpt2",
    learning_rate=0.001,
    batch_size=32
)

serializer = ConfigSerializer()

# Serialize to strings
yaml_str = serializer.serialize_yaml(config)
json_str = serializer.serialize_json(config)
toml_str = serializer.serialize_toml(config)

# Write to files
serializer.dump_yaml(config, "config.yaml")
serializer.dump_json(config, "config.json")
serializer.dump_toml(config, "config.toml")

# Format-agnostic dump/load based on extension
serializer.dump(config, "config.yaml")
cfg = serializer.load("config.toml")

# Raw dict IO (no `Buildable`). Supports section extraction via path:section
data = serializer.load_dict("config.yaml")
serializer.dump_dict(data, "subset.yaml:training")
```

### Custom Validation and Serialization

Handle values that aren't directly representable in the chosen format:

```python
from pathlib import Path
import nemo_run as run

# Wrap values not representable in the chosen format
config = run.Config(
    process_data,
    input_path=run.Config(Path, "/path/to/data"),
    output_path=run.Config(Path, "/path/to/output"),
    batch_size=32
)

# Tip: Prefer wrapping values not representable in the chosen format (e.g., pathlib.Path) with run.Config(...)
# so the serializer can handle them without manual dict manipulation.
```

## Advanced Configuration Patterns

As projects grow, organize configurations with nesting, composition, and templates. These patterns help you share common defaults, override selectively, and keep experiment definitions readable as they evolve.

### Nested Configurations

Create complex nested configurations:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ModelArchitecture:
    type: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1

@dataclass
class OptimizerConfig:
    type: str
    learning_rate: float
    weight_decay: float = 1e-5
    betas: List[float] = None

@dataclass
class DataConfig:
    batch_size: int
    num_workers: int
    data_path: str
    augmentation: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    epochs: int
    save_path: str
    checkpoint_freq: int = 10
    early_stopping: bool = True

def create_comprehensive_config(
    model: ModelArchitecture,
    optimizer: OptimizerConfig,
    data: DataConfig,
    training: TrainingConfig,
    experiment_name: str = "default",
    seed: int = 42
) -> run.Config:
    """Create a comprehensive training configuration."""

    return run.Config(
        train_comprehensive,
        model=model,
        optimizer=optimizer,
        data=data,
        training=training,
        experiment_name=experiment_name,
        seed=seed
    )

# Usage
config = create_comprehensive_config(
    model=ModelArchitecture(type="transformer", hidden_size=768, num_layers=12),
    optimizer=OptimizerConfig(type="adam", learning_rate=1e-4),
    data=DataConfig(batch_size=32, num_workers=4, data_path="/data"),
    training=TrainingConfig(epochs=100, save_path="/models")
)
```

### Configuration Composition

Compose configurations from several sources:

```python
import nemo_run as run
from typing import Dict, Any

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configurations with override support."""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged

def create_config_from_parts(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> run.Config:
    """Create configuration from separate parts."""

    # Merge configurations
    full_config = merge_configs(
        merge_configs(model_config, training_config),
        data_config
    )

    return run.Config(train_model, **full_config)

# Usage
model_config = {"model_name": "gpt2", "hidden_size": 768}
training_config = {"learning_rate": 0.001, "epochs": 100}
data_config = {"batch_size": 32, "data_path": "/data"}

config = create_config_from_parts(model_config, training_config, data_config)
```

### Configuration Templates

Create reusable configuration templates:

```python
import nemo_run as run
from typing import Dict, Any

class ConfigTemplate:
    """Base class for configuration templates."""

    @classmethod
    def create(cls, **kwargs) -> run.Config:
        """Create configuration from template."""
        raise NotImplementedError

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration values."""
        raise NotImplementedError

class TransformerTemplate(ConfigTemplate):
    """Template for transformer model configurations."""

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        return {
            "model_type": "transformer",
            "hidden_size": 512,
            "num_layers": 6,
            "num_attention_heads": 8,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100
        }

    @classmethod
    def create(cls, **kwargs) -> run.Config:
        defaults = cls.get_defaults()
        config = {**defaults, **kwargs}

        return run.Config(
            train_transformer,
            **config
        )

class CNNTemplate(ConfigTemplate):
    """Template for CNN model configurations."""

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        return {
            "model_type": "cnn",
            "channels": [64, 128, 256],
            "kernel_sizes": [3, 3, 3],
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50
        }

    @classmethod
    def create(cls, **kwargs) -> run.Config:
        defaults = cls.get_defaults()
        config = {**defaults, **kwargs}

        return run.Config(
            train_cnn,
            **config
        )

# Usage
transformer_config = TransformerTemplate.create(hidden_size=768, epochs=200)
cnn_config = CNNTemplate.create(channels=[32, 64, 128], learning_rate=0.01)
```

## Fiddle Integration

Fiddle interops seamlessly with NeMo Run: you can author configs with Fiddle, then cast them to NeMo Run `Config` for execution. This lets teams who prefer Fiddle’s ergonomics collaborate without sacrificing NeMo Run’s execution and metadata features.

### Basic Fiddle Usage

NeMo Run integrates with Fiddle for advanced configuration management:

```python
import nemo_run as run
import fiddle as fdl

# Create Fiddle configuration
config = fdl.Config(train_model)

# Set configuration values
fdl.set_field(config, "model_name", "gpt2")
fdl.set_field(config, "learning_rate", 0.001)
fdl.set_field(config, "batch_size", 32)

# Convert Fiddle config to NeMo Run configuration
from nemo_run.config import Config
nemo_config = fdl.cast(Config, config)
```

### Advanced Fiddle Patterns

```python
import fiddle as fdl
import nemo_run as run
from typing import List

@fdl.dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1

@fdl.dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"

def create_fiddle_config() -> fdl.Config:
    """Create a Fiddle configuration."""
    config = fdl.Config(train_model)

    # Set model configuration
    model_config = fdl.Config(ModelConfig)
    fdl.set_field(model_config, "name", "transformer")
    fdl.set_field(model_config, "hidden_size", 512)
    fdl.set_field(model_config, "num_layers", 6)

    # Set training configuration
    training_config = fdl.Config(TrainingConfig)
    fdl.set_field(training_config, "learning_rate", 0.001)
    fdl.set_field(training_config, "batch_size", 32)
    fdl.set_field(training_config, "epochs", 100)

    # Set main configuration
    fdl.set_field(config, "model", model_config)
    fdl.set_field(config, "training", training_config)

    return config

fiddle_config = create_fiddle_config()
from nemo_run.config import Config
nemo_config = fdl.cast(Config, fiddle_config)
```

## Configuration Validation Rules

Use this reference to understand how NeMo Run interprets common Python types during validation, and how to extend validation with custom rules when defaults are not sufficient.

### Type Validation Rules

| Type | Validation | Example |
|------|------------|---------|
| `int` | Must be integer | `batch_size=32` |
| `float` | Must be float | `learning_rate=0.001` |
| `str` | Must be string | `model_name="gpt2"` |
| `bool` | Must be Boolean | `debug=true` |
| `List[T]` | Must be list of type T | `layers=[128,256,512]` |
| `Dict[str, T]` | Must be a dictionary with string keys | `config={'dropout': 0.1}` |
| `Optional[T]` | Can be None or type T | `scheduler=None` |
| `Union[T1, T2]` | Must be one of the types | `activation="relu"` |

### Custom Validation Rules

```python
from typing import Any, Callable, List
import nemo_run as run

class ValidationRule:
    """Base class for validation rules."""

    def __init__(self, validator: Callable[[Any], bool], message: str):
        self.validator = validator
        self.message = message

    def validate(self, value: Any) -> None:
        """Validate a value."""
        if not self.validator(value):
            raise ValueError(self.message)

class PositiveRule(ValidationRule):
    """Validate that a value is positive."""

    def __init__(self):
        super().__init__(
            lambda x: x > 0,
            "Value must be positive"
        )

class RangeRule(ValidationRule):
    """Validate that a value is within a range."""

    def __init__(self, min_val: float, max_val: float):
        super().__init__(
            lambda x: min_val <= x <= max_val,
            f"Value must be between {min_val} and {max_val}"
        )

class ChoiceRule(ValidationRule):
    """Validate that a value is one of the allowed choices."""

    def __init__(self, choices: List[Any]):
        super().__init__(
            lambda x: x in choices,
            f"Value must be one of {choices}"
        )

def create_validated_config(
    model_name: str,
    hidden_size: int,
    learning_rate: float,
    optimizer: str,
    dropout: float = 0.1
) -> run.Config:
    """Create a configuration with custom validation."""

    # Apply validation rules
    PositiveRule().validate(hidden_size)
    RangeRule(1e-6, 1.0).validate(learning_rate)
    RangeRule(0.0, 1.0).validate(dropout)
    ChoiceRule(["adam", "sgd", "adamw"]).validate(optimizer)

    return run.Config(
        train_model,
        model_name=model_name,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        dropout=dropout
    )
```

## Configuration Serialization Formats

The following examples show the same configuration expressed in different formats:

**YAML (YAML Ain't Markup Language):**

```yaml
# config.yaml
model:
  name: "transformer"
  hidden_size: 768
  num_layers: 12
  dropout: 0.1

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  optimizer: "adam"

data:
  path: "/path/to/data"
  num_workers: 4
  augmentation:
    rotation: 10
    flip: true

experiment:
  name: "transformer_experiment"
  seed: 42
  debug: false
```

**JSON (JavaScript Object Notation):**

```json
{
  "model": {
    "name": "transformer",
    "hidden_size": 768,
    "num_layers": 12,
    "dropout": 0.1
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
  },
  "data": {
    "path": "/path/to/data",
    "num_workers": 4,
    "augmentation": {
      "rotation": 10,
      "flip": true
    }
  },
  "experiment": {
    "name": "transformer_experiment",
    "seed": 42,
    "debug": false
  }
}
```

**TOML (Tom's Obvious, Minimal Language):**

```toml
[model]
name = "transformer"
hidden_size = 768
num_layers = 12
dropout = 0.1

[training]
learning_rate = 0.001
batch_size = 32
epochs = 100
optimizer = "adam"

[data]
path = "/path/to/data"
num_workers = 4

[data.augmentation]
rotation = 10
flip = true

[experiment]
name = "transformer_experiment"
seed = 42
debug = false
```

## Configuration Loading and Saving

Load and save configurations using file extensions to pick the format automatically. The serializer preserves structure across formats, so you can edit a YAML file and load it back as a buildable `Config`/`Partial` with no manual conversions.

### Load from a File

```python
from nemo_run.cli.config import ConfigSerializer

serializer = ConfigSerializer()

# Load to a Buildable (Config/Partial) from YAML/JSON/TOML
cfg = serializer.load("config.yaml")
```

### Save to a File

```python
from nemo_run.cli.config import ConfigSerializer

serializer = ConfigSerializer()
config = run.Config(train_model, model_name="gpt2", learning_rate=0.001)

serializer.dump_yaml(config, "config.yaml")
serializer.dump_json(config, "config.json")
```

## Configuration Best Practices

These guidelines help keep configurations clear, maintainable, and reproducible across teams. Favor explicit types, sensible defaults, and small, composable templates over ad‑hoc dictionaries.

### Use Type Hints Consistently

```python
from typing import Optional, List, Dict, Any, Union

def create_config(
    model_name: str,
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    optimizer: str = "adam",
    scheduler: Optional[str] = None,
    augmentation: Dict[str, Any] = None
) -> run.Config:
    """Create configuration with comprehensive type hints."""
    pass
```

### Provide Sensible Defaults

```python
def create_config(
    model_name: str,
    learning_rate: float = 0.001,  # Sensible default
    batch_size: int = 32,          # Good balance
    epochs: int = 10,              # Reasonable duration
    seed: int = 42                 # Reproducible default
) -> run.Config:
    pass
```

### Configuration Validation Guidelines

```python
def create_validated_config(
    model_name: str,
    hidden_size: int,
    learning_rate: float
) -> run.Config:
    """Create configuration with validation."""

    # Validate inputs
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if not model_name:
        raise ValueError("model_name cannot be empty")

    return run.Config(train_model, **locals())
```

### Use Configuration Templates

```python
class ConfigTemplate:
    """Base template for configurations."""

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "seed": 42
        }

    @classmethod
    def create(cls, **kwargs) -> run.Config:
        """Create configuration from template."""
        defaults = cls.get_defaults()
        config = {**defaults, **kwargs}
        return run.Config(train_model, **config)
```

### Handle Complex Objects

```python
from pathlib import Path
import nemo_run as run

def create_config_with_paths(
    data_path: str,
    model_path: str,
    output_path: str
) -> run.Config:
    """Create configuration with Path objects."""

    return run.Config(
        process_data,
        data_path=run.Config(Path, data_path),
        model_path=run.Config(Path, model_path),
        output_path=run.Config(Path, output_path)
    )
```

This comprehensive configuration reference provides all the tools and patterns needed to create robust, type-safe configurations for NeMo Run experiments. The system integrates seamlessly with Fiddle for advanced configuration management while providing excellent serialization and validation capabilities.

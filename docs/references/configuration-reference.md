---
description: "Comprehensive reference for NeMo Run's configuration system, types, validation, and advanced patterns."
tags: ["configuration", "reference", "types", "validation", "fiddle"]
categories: ["references"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "text-only"
---

(configuration-reference)=

# Configuration Reference

Comprehensive reference documentation for NeMo Run's type-safe configuration system, covering all configuration types, validation rules, serialization patterns, and advanced configuration techniques.

## Overview

NeMo Run's configuration system provides type-safe, serializable configurations with Fiddle integration, enabling reproducible and validated experiment setups across diverse computing environments.

## Core Configuration Types

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
result = run.run(partial_config)
```

### `run.Script`

Wraps executable scripts with configuration:

```python
# Script configuration
script_config = run.Script(
    "train.py",
    model_name="resnet50",
    epochs=100,
    batch_size=64
)
```

## Configuration Validation

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

### Custom Validators

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

### Built-in Serialization

NeMo Run configurations are automatically serializable:

```python
import nemo_run as run
import json
import yaml

# Create configuration
config = run.Config(
    train_model,
    model_name="gpt2",
    learning_rate=0.001,
    batch_size=32
)

# Serialize to dictionary
config_dict = config.to_dict()

# Serialize to JSON
config_json = json.dumps(config_dict, indent=2)

# Serialize to YAML
config_yaml = yaml.dump(config_dict, default_flow_style=False)

# Save to file
with open("config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

with open("config.yaml", "w") as f:
    yaml.dump(config_dict, f)
```

### Custom Serialization

Handle non-serializable objects:

```python
from pathlib import Path
import nemo_run as run

# Wrap non-serializable objects
config = run.Config(
    process_data,
    input_path=run.Config(Path, "/path/to/data"),
    output_path=run.Config(Path, "/path/to/output"),
    batch_size=32
)

# Serialize with custom handling
def serialize_config(config: run.Config) -> dict:
    """Serialize configuration with custom object handling."""
    config_dict = config.to_dict()

    # Handle Path objects
    for key, value in config_dict.items():
        if isinstance(value, dict) and "type" in value and value["type"] == "pathlib.Path":
            config_dict[key] = str(value["args"][0])

    return config_dict
```

## Advanced Configuration Patterns

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

Compose configurations from multiple sources:

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

# Convert to NeMo Run configuration
nemo_config = run.Config.from_fiddle(config)
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

# Convert to NeMo Run configuration
fiddle_config = create_fiddle_config()
nemo_config = run.Config.from_fiddle(fiddle_config)
```

## Configuration Validation Rules

### Type Validation Rules

| Type | Validation | Example |
|------|------------|---------|
| `int` | Must be integer | `batch_size=32` |
| `float` | Must be float | `learning_rate=0.001` |
| `str` | Must be string | `model_name="gpt2"` |
| `bool` | Must be boolean | `debug=true` |
| `List[T]` | Must be list of type T | `layers=[128,256,512]` |
| `Dict[str, T]` | Must be dict with string keys | `config={'dropout': 0.1}` |
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

### YAML Format

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

### JSON Format

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

### TOML Format

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

### Load from File

```python
import nemo_run as run
import yaml
import json

def load_config_from_yaml(file_path: str) -> run.Config:
    """Load configuration from YAML file."""
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return run.Config(train_model, **config_dict)

def load_config_from_json(file_path: str) -> run.Config:
    """Load configuration from JSON file."""
    with open(file_path, 'r') as f:
        config_dict = json.load(f)

    return run.Config(train_model, **config_dict)

# Usage
config = load_config_from_yaml("config.yaml")
config = load_config_from_json("config.json")
```

### Save to File

```python
import nemo_run as run
import yaml
import json

def save_config_to_yaml(config: run.Config, file_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = config.to_dict()

    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def save_config_to_json(config: run.Config, file_path: str) -> None:
    """Save configuration to JSON file."""
    config_dict = config.to_dict()

    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

# Usage
config = run.Config(train_model, model_name="gpt2", learning_rate=0.001)
save_config_to_yaml(config, "config.yaml")
save_config_to_json(config, "config.json")
```

## Configuration Best Practices

### 1. Use Type Hints Consistently

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

### 2. Provide Sensible Defaults

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

### 3. Validate Configuration

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

### 4. Use Configuration Templates

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

### 5. Handle Complex Objects

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

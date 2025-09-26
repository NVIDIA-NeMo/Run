---
description: "Complete CLI reference for NeMo Run including all commands, options, and usage examples."
tags: ["command-line", "reference", "commands", "options"]
categories: ["references"]
personas: ["machine-learning-engineer-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "text-only"
---

# CLI Reference

NeMo Run provides a powerful CLI that transforms Python functions into sophisticated CLI tools with rich argument parsing, type safety, and seamless integration with execution backends.

## Core Concepts

Understand the primary building blocks of the CLI and how they map to code. These concepts help you reason about what the CLI generates and how it executes your Python functions.

### Entry Points

Entry points are Python functions decorated with `@run.cli.entrypoint` that become accessible as CLI commands.

### Factory Functions

Factory functions (decorated with `@run.cli.factory`) create reusable configuration components.

### Run Context

The `RunContext` manages execution settings and provides executor configuration, plugin management, and execution control.

## Command-Line Argument Syntax

NeMo Run supports rich Python-like argument syntax:

```bash
# Basic arguments
python script.py model_name=gpt2 learning_rate=0.001

# Nested attribute setting
python script.py model.hidden_size=512 data.batch_size=64

# List and dictionary arguments
python script.py layers=[128,256,512] config={'dropout': 0.1}

# Operations on arguments
python script.py counter+=1 rate*=2 flags|=0x1

# Type casting
python script.py int_arg=42 float_arg=3.14 bool_arg=true

# None values
python script.py optional_arg=None

# Factory function usage
python script.py model=create_model(hidden_size=256)
```

## Global Options

Use these global options:

| Option | Description | Example |
|--------|-------------|---------|
| `--name, -n` | Name of the run | `--name my_experiment` |
| `--direct/--no-direct` | Execute directly (skip executor) | `--direct` |
| `--dryrun` | Preview without execution | `--dryrun` |
| `--factory, -f` | Use predefined factory (or `@file` to load) | `--factory my_factory` |
| `--load, -l` | Load factory from directory | `--load ./configs/` |
| `--yaml, -y` | Load from YAML file | `--yaml config.yaml` |
| `--repl, -r` | Enter interactive mode | `--repl` |
| `--detach` | Run in background | `--detach` |
| `--yes, --no-confirm, -y` | Skip confirmation | `--yes` |
| `--tail-logs` | Follow logs | `--tail-logs` |
| `--verbose, -v` | Enable verbose logging | `--verbose` |

## Output Options

Use these output options:

| Option | Description | Example |
|--------|-------------|---------|
| `--to-yaml` | Export to YAML | `--to-yaml output.yaml` |
| `--to-toml` | Export to TOML | `--to-toml output.toml` |
| `--to-json` | Export to JSON | `--to-json output.json` |

## Rich Output Options

Use these rich output options:

| Option | Description | Example |
|--------|-------------|---------|
| `--rich-exceptions` | Enable rich exception formatting | `--rich-exceptions` |
| `--rich-traceback-short/--rich-traceback-full` | Control stack trace verbosity | `--rich-traceback-full` |
| `--rich-show-locals/--rich-hide-locals` | Toggle local variables in exceptions | `--rich-show-locals` |
| `--rich-theme` | Color theme (dark, light, or monochrome) | `--rich-theme dark` |

## Advanced Features

Explore power‑user capabilities that enhance interactivity, portability, and operational workflows when using the CLI.

### Interactive Mode (Read–Eval–Print Loop)

Start an interactive session to explore configurations (not supported with `--lazy`):

```bash
python script.py train_model --repl
```

### Configuration Export

Export your configurations to various formats:

```bash
# Export to YAML
python script.py train_model model=resnet50 --to-yaml config.yaml

# Export to TOML
python script.py train_model model=resnet50 --to-toml config.toml

# Export to JSON
python script.py train_model model=resnet50 --to-json config.json

# Export specific sections (append :section to the output path)
python script.py train_model model=resnet50 --to-yaml config.yaml:model
# In both regular and lazy modes, ':section' extracts a top-level attribute
# from the configuration object before serialization.
```

### Lazy Mode

Run a one-off command using lazy resolution without loading nested entry points:

```bash
python script.py train_model --lazy --to-yaml config.yaml
```

Notes:

- `--lazy` doesn't support `devspace` and `experiment` commands.
- `--dryrun` and `--repl` aren't supported in lazy mode.
- Export flags `--to-yaml/--to-toml/--to-json` work in lazy mode and skip execution after export.

### Dry Run Mode

Preview the commands without running them:

```bash
python script.py train_model model=resnet50 --dryrun
```

### Detached Execution

Run tasks in the background:

```bash
python script.py train_model model=resnet50 --detach
```

### Tail Logs

Follow logs in real-time:

```bash
python script.py train_model model=resnet50 --tail-logs
```

## Executor Integration

Bind CLI entry points to execution environments and learn how to override executors at runtime without changing code.

### Default Executors

Set default executors for your entry points:

```python
import nemo_run as run

@run.cli.entrypoint(
    default_executor=run.DockerExecutor(
        container_image="pytorch/pytorch:latest",
        num_gpus=1
    )
)
def train_model(model: str, epochs: int = 10):
    """Train a model using Docker executor by default."""
    print(f"Training {model} for {epochs} epochs")
```

### Executor Override from the Command Line

```bash
# Use default executor
python script.py train_model model=resnet50

# Override with different executor
python script.py train_model model=resnet50 executor=local

# Configure Slurm executor parameters
python script.py train_model executor=slurm executor.account=acct executor.partition=gpu executor.time=02:00:00

# Override executor settings
# Docker example
python script.py train_model executor=docker executor.num_gpus=4 executor.shm_size=30g
# Slurm example
python script.py train_model executor=slurm executor.gpus_per_node=4 executor.mem=32g
```

## Factory Functions in Depth

Factory functions allow you to create reusable configuration components:

```python
import nemo_run as run
from dataclasses import dataclass
from typing import List

@dataclass
class OptimizerConfig:
    type: str
    lr: float
    betas: List[float]
    weight_decay: float

@run.cli.factory
def create_optimizer(optimizer_type: str = "adam", lr: float = 0.001) -> OptimizerConfig:
    """Create an optimizer configuration."""
    if optimizer_type == "adam":
        return OptimizerConfig(
            type="adam",
            lr=lr,
            betas=[0.9, 0.999],
            weight_decay=1e-5
        )
    elif optimizer_type == "sgd":
        return OptimizerConfig(
            type="sgd",
            lr=lr,
            betas=[0.0, 0.0],
            weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

@run.cli.entrypoint
def train_with_optimizer(
    model: str,
    optimizer: OptimizerConfig = create_optimizer(optimizer_type="adam", lr=0.001)
):
    """Train a model with a configurable optimizer."""
    print(f"Training {model} with {optimizer.type} optimizer")
    print(f"Learning rate: {optimizer.lr}")
```

### Factory Registration Patterns

#### Type-Based Registration

Register factories for specific types:

```python
@run.cli.factory
def create_transformer_model() -> run.Config[TransformerModel]:
    """Create a default transformer model configuration."""
    return run.Config(
        TransformerModel,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8
    )

@run.cli.factory
def create_cnn_model() -> run.Config[CNNModel]:
    """Create a default CNN model configuration."""
    return run.Config(
        CNNModel,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3]
    )
```

#### Parameter-Specific Registration

Register factories for specific parameters:

```python
@run.cli.factory(target=train_model, target_arg="model")
def create_default_model() -> run.Config[BaseModel]:
    """Default model factory for train_model function."""
    return create_transformer_model()

@run.cli.factory(target=train_model, target_arg="optimizer")
def create_default_optimizer() -> OptimizerConfig:
    """Default optimizer factory for train_model function."""
    return create_optimizer(optimizer_type="adam", lr=0.001)
```

### Use Factories from the Command Line

```bash
# Use default factory
python script.py train_with_optimizer model=resnet50

# Override factory parameters
python script.py train_with_optimizer model=resnet50 optimizer=create_optimizer(optimizer_type=sgd,lr=0.01)

# Nested factory usage
python script.py train_with_optimizer model=resnet50 optimizer.lr=0.005

# Use type-based factories
python script.py train_model model=create_transformer_model optimizer=create_optimizer
```

```bash
# Plugins via single value or list factory
python script.py train_with_optimizer plugins=my_plugin
python script.py train_with_optimizer plugins=plugin_list plugins[0].some_arg=50
```

## Troubleshoot

Use these tips to diagnose common CLI issues quickly and apply targeted fixes.

### Common Issues

1. **Type Conversion Errors**

   ```bash
   # Error: Cannot convert string to int
   python script.py batch_size=32.5  # Should be int

   # Fix: Use explicit type
   python script.py batch_size=32
   ```

2. **Nested Configuration Issues**

   ```bash
   # Error: Cannot set nested attribute
   python script.py model.config.hidden_size=512

   # Fix: Use factory or direct assignment
   python script.py model=create_model(hidden_size=512)
   ```

3. **Factory Resolution Issues**

   ```bash
   # Error: Factory not found
   python script.py optimizer=unknown_factory()

   # Fix: Use registered factory
   python script.py optimizer=create_optimizer()
   ```

4. **Executor Configuration Issues**

   ```bash
   # Error: Invalid executor parameter
   python script.py executor=run.SlurmExecutor(invalid_param=value)

   # Fix: Check executor documentation for valid parameters
   python script.py executor=run.SlurmExecutor(partition=gpu,time=02:00:00)
   ```

### Debug Strategies

1. **Use `--verbose` for detailed output**

   ```bash
   python script.py train_model --verbose
   ```

2. **Use `--dryrun` to preview execution**

   ```bash
   python script.py train_model --dryrun
   ```

3. **Use `--repl` for interactive debugging**

   ```bash
   python script.py train_model --repl
   ```

4. **Export configurations to inspect them**

   ```bash
   python script.py train_model --to-yaml debug_config.yaml
   ```

5. **Inspect pre-loaded factories via help**

   ```bash
   python script.py train_model --help
   ```

   The help output includes a "Pre-loaded entry point factories, run with --factory" table.

## Examples

Copy‑pasteable snippets that demonstrate typical CLI usage patterns—from basic commands to advanced pipelines.

### Basic Entry Point

```python
import nemo_run as run

@run.cli.entrypoint
def train_model(
    model_name: str = "gpt2",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a machine learning model with specified parameters."""
    try:
        print(f"Training {model_name} with lr={learning_rate}, batch_size={batch_size}")
        # Your training logic here
        return {"accuracy": 0.95, "loss": 0.1}
    except Exception as e:
        print(f"Training failed: {e}")
        return {"accuracy": 0.0, "loss": float('inf')}
```

### Advanced Training Pipeline

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1

@dataclass
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float = 1e-5
    betas: List[float] = None

@dataclass
class DataConfig:
    path: str
    batch_size: int
    num_workers: int = 4

@run.cli.factory
def create_model(name: str, hidden_size: int = 512) -> ModelConfig:
    return ModelConfig(name=name, hidden_size=hidden_size, num_layers=6)

@run.cli.factory
def create_optimizer(optimizer: str = "adam", lr: float = 0.001) -> OptimizerConfig:
    betas = [0.9, 0.999] if optimizer == "adam" else [0.0, 0.0]
    return OptimizerConfig(type=optimizer, lr=lr, betas=betas)

@run.cli.factory
def create_data(data_path: str, batch_size: int = 32) -> DataConfig:
    return DataConfig(path=data_path, batch_size=batch_size)

@run.cli.entrypoint(
    help="Advanced training pipeline with comprehensive configuration and validation",
    default_executor=run.DockerExecutor(container_image="pytorch/pytorch:latest")
)
def advanced_training_pipeline(
    model: ModelConfig = create_model(name="transformer"),
    optimizer: OptimizerConfig = create_optimizer(optimizer="adam", lr=0.001),
    data: DataConfig = create_data(data_path="./data", batch_size=32),
    epochs: int = 10,
    save_path: str = "./models",
    experiment_name: str = "default_experiment",
    seed: int = 42,
    debug: bool = False
):
    """Advanced training pipeline with comprehensive configuration."""
    print(f"=== Training Configuration ===")
    print(f"Model: {model.name} (hidden_size={model.hidden_size}, layers={model.num_layers})")
    print(f"Optimizer: {optimizer.type} (lr={optimizer.lr}, weight_decay={optimizer.weight_decay})")
    print(f"Data: {data.path} (batch_size={data.batch_size}, workers={data.num_workers})")
    print(f"Training: {epochs} epochs, save_path={save_path}")
    print(f"Experiment: {experiment_name}, Seed: {seed}, Debug: {debug}")

    # Validation
    if model.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if optimizer.lr <= 0:
        raise ValueError("learning_rate must be positive")
    if data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Your training logic here
    return {
        "status": "completed",
        "accuracy": 0.95,
        "loss": 0.1,
        "config": {
            "model": model,
            "optimizer": optimizer,
            "data": data,
            "training": {
                "epochs": epochs,
                "save_path": save_path,
                "experiment_name": experiment_name,
                "seed": seed
            }
        }
    }
```

### Command-Line Usage Examples

```bash
# Use defaults
python script.py advanced_training_pipeline

# Customize components
python script.py advanced_training_pipeline \
    model=create_model(name=resnet50,hidden_size=1024) \
    optimizer=create_optimizer(optimizer=sgd,lr=0.01) \
    data=create_data(data_path=/path/to/data,batch_size=64) \
    epochs=20 \
    save_path=/path/to/save \
    experiment_name=resnet_experiment

# Export configuration
python script.py advanced_training_pipeline --to-yaml config.yaml

# Dry run
python script.py advanced_training_pipeline --dryrun

# Interactive mode
python script.py advanced_training_pipeline --repl

# Detached execution
python script.py advanced_training_pipeline --detach

# Follow logs
python script.py advanced_training_pipeline --tail-logs
```

This CLI system provides a powerful and flexible way to interact with NeMo Run, making it easy to create command-line tools for your ML workflows while maintaining the full power of Python's type system and configuration capabilities.

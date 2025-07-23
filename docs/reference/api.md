---
description: "Complete API reference for NeMo Run including configuration, execution, and management classes."
categories: ["reference"]
tags: ["api", "reference", "configuration", "execution", "management"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(api)=

# NeMo Run API Reference

This document provides comprehensive API reference for NeMo Run, covering all core classes, methods, and usage patterns.

## Core Configuration Classes

### `run.Config`

Type-safe configuration for classes and functions using Fiddle.

```python
import nemo_run as run

# Basic usage
config = run.Config(MyClass, param1=value1, param2=value2)
instance = config.build()

# Function configuration
def my_function(x, y, z=10):
    return x + y + z

func_config = run.Config(my_function, x=5, y=3, z=20)
result = func_config.build()()
```

**Parameters:**
- `fn_or_cls`: Class or function to configure
- `*args`: Positional arguments
- `**kwargs`: Keyword arguments

**Methods:**
- `.build()`: Build and return the configured object
- `.to_dict()`: Convert configuration to dictionary
- `.clone()`: Create a copy of the configuration

### `run.Partial`

Create partially applied functions with fixed parameters.

```python
import nemo_run as run

def train_model(model, learning_rate=0.001, epochs=100):
    # Training logic
    pass

# Create partial with fixed parameters
train_partial = run.Partial(train_model, learning_rate=0.01, epochs=50)

# Use in experiment
with run.Experiment("training") as experiment:
    experiment.add(train_partial, name="training_run")
    experiment.run()
```

**Parameters:**
- `fn`: Function to partially apply
- `*args`: Fixed positional arguments
- `**kwargs`: Fixed keyword arguments

### `run.Script`

Execute raw scripts and commands.

```python
import nemo_run as run

# Execute Python script
script = run.Script("python train.py --epochs 100")

# Execute shell command
script = run.Script("bash train.sh")

# Use in experiment
with run.Experiment("script_execution") as experiment:
    experiment.add(script, name="script_run")
    experiment.run()
```

## Execution Classes

### `run.Experiment`

Context manager for managing multiple runs and experiments.

```python
import nemo_run as run

with run.Experiment("my_experiment") as experiment:
    # Add tasks to experiment
    experiment.add(config1, name="task1")
    experiment.add(config2, name="task2")

    # Run all tasks
    experiment.run()

    # Check status
    experiment.status()

    # Get logs
    experiment.logs("task1")
```

**Constructor Parameters:**
- `title`: Experiment title/name
- `executor`: Default executor for the experiment
- `id`: Custom experiment ID
- `log_level`: Logging level (default: "INFO")

**Methods:**
- `.add(task, name="", executor=None, plugins=None, tail_logs=False, dependencies=None)`: Add task to experiment
- `.run(sequential=False, detach=False, tail_logs=False, direct=False)`: Run all tasks
- `.status(return_dict=False)`: Get experiment status
- `.logs(job_id, regex=None)`: Get logs for specific job
- `.cancel(job_id)`: Cancel specific job
- `.reset()`: Reset experiment state

**Class Methods:**
- `.from_id(id)`: Load experiment by ID
- `.from_title(title)`: Load experiment by title
- `.catalog(title="")`: List available experiments

### `run.run()`

Execute a single configured function.

```python
import nemo_run as run

# Direct execution
result = run.run(config)

# With executor
result = run.run(config, executor=executor)

# With options
result = run.run(
    config,
    executor=executor,
    name="my_run",
    dryrun=True,
    direct=False,
    detach=False,
    tail_logs=True,
    log_level="INFO"
)
```

**Parameters:**
- `fn_or_script`: Configured function or script to run
- `executor`: Executor to use (optional)
- `name`: Run name (optional)
- `dryrun`: Preview execution without running (default: False)
- `direct`: Run directly without executor (default: False)
- `detach`: Run in background (default: False)
- `tail_logs`: Follow logs in real-time (default: True)
- `log_level`: Logging level (default: "INFO")

## Executor Classes

### `run.LocalExecutor`

Execute tasks locally in a separate process.

```python
import nemo_run as run

executor = run.LocalExecutor(
    packager=run.Packager(),
    env_vars={"CUDA_VISIBLE_DEVICES": "0"},
    retries=0
)
```

**Parameters:**
- `packager`: Code packaging strategy
- `env_vars`: Environment variables
- `retries`: Number of retry attempts
- `launcher`: Task launcher (optional)

### `run.DockerExecutor`

Execute tasks in Docker containers.

```python
import nemo_run as run

executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    num_gpus=1,
    runtime="nvidia",
    volumes=["/host/data:/container/data"],
    env_vars={"PYTHONUNBUFFERED": "1"},
    packager=run.GitArchivePackager()
)
```

**Parameters:**
- `container_image`: Docker image to use
- `num_gpus`: Number of GPUs (-1 for all available)
- `runtime`: Docker runtime
- `volumes`: Volume mappings
- `env_vars`: Environment variables
- `packager`: Code packaging strategy
- `shm_size`: Shared memory size
- `ipc_mode`: IPC mode
- `network`: Docker network

### `run.SlurmExecutor`

Execute tasks on Slurm clusters.

```python
import nemo_run as run

executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    time="02:00:00",
    job_name="nemo_experiment",
    account="my_account",
    packager=run.GitArchivePackager()
)
```

**Parameters:**
- `partition`: Slurm partition
- `nodes`: Number of nodes
- `gpus_per_node`: GPUs per node
- `time`: Time limit
- `job_name`: Job name
- `account`: Account name
- `packager`: Code packaging strategy
- `tunnel`: SSH tunnel configuration

### `run.SkypilotExecutor`

Execute tasks on multiple cloud platforms.

```python
import nemo_run as run

executor = run.SkypilotExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    cloud="aws",
    region="us-west-2",
    gpus="A10G",
    gpus_per_node=1,
    num_nodes=1,
    use_spot=True
)
```

**Parameters:**
- `container_image`: Container image
- `cloud`: Cloud provider (aws, gcp, azure, etc.)
- `region`: Cloud region
- `gpus`: GPU type
- `gpus_per_node`: GPUs per node
- `num_nodes`: Number of nodes
- `use_spot`: Use spot instances
- `packager`: Code packaging strategy

### `run.DGXCloudExecutor`

Execute tasks on NVIDIA DGX Cloud.

```python
import nemo_run as run

executor = run.DGXCloudExecutor(
    project_id="your-project-id",
    cluster_id="your-cluster-id",
    node_count=1,
    gpus_per_node=8,
    image="nvidia/pytorch:24.05-py3"
)
```

**Parameters:**
- `project_id`: DGX Cloud project ID
- `cluster_id`: Cluster ID
- `node_count`: Number of nodes
- `gpus_per_node`: GPUs per node
- `image`: Container image
- `packager`: Code packaging strategy

### `run.LeptonExecutor`

Execute tasks on Lepton cloud platform.

```python
import nemo_run as run

executor = run.LeptonExecutor(
    workspace_id="your-workspace-id",
    resource_shape="gpu.a10",
    image="nvidia/pytorch:24.05-py3",
    packager=run.GitArchivePackager()
)
```

**Parameters:**
- `workspace_id`: Lepton workspace ID
- `resource_shape`: Resource configuration
- `image`: Container image
- `packager`: Code packaging strategy

## Packager Classes

### `run.Packager`

Base packager (pass-through).

```python
import nemo_run as run

packager = run.Packager()
```

### `run.GitArchivePackager`

Package code using Git archive.

```python
import nemo_run as run

packager = run.GitArchivePackager(
    subpath="src",  # Package only src directory
    exclude_patterns=["*.pyc", "__pycache__"]
)
```

**Parameters:**
- `subpath`: Subdirectory to package
- `exclude_patterns`: Patterns to exclude
- `include_patterns`: Patterns to include

### `run.PatternPackager`

Package files based on patterns.

```python
import nemo_run as run

packager = run.PatternPackager(
    include_pattern="src/**/*.py",
    exclude_pattern="**/__pycache__/**",
    relative_path=os.getcwd()
)
```

**Parameters:**
- `include_pattern`: Files to include
- `exclude_pattern`: Files to exclude
- `relative_path`: Base path for relative paths

### `run.HybridPackager`

Combine multiple packaging strategies.

```python
import nemo_run as run

packager = run.HybridPackager([
    run.GitArchivePackager(subpath="src"),
    run.PatternPackager(include_pattern="configs/**")
])
```

**Parameters:**
- `packagers`: List of packagers to combine

## Launcher Classes

### `run.Torchrun`

PyTorch distributed training launcher.

```python
import nemo_run as run

launcher = run.Torchrun(
    nnodes=2,
    nproc_per_node=4,
    rdzv_backend="c10d",
    rdzv_endpoint="localhost:29400"
)
```

**Parameters:**
- `nnodes`: Number of nodes
- `nproc_per_node`: Processes per node
- `rdzv_backend`: Rendezvous backend
- `rdzv_endpoint`: Rendezvous endpoint
- `rdzv_id`: Rendezvous ID

### `run.FaultTolerance`

Fault-tolerant training launcher.

```python
import nemo_run as run

launcher = run.FaultTolerance(
    max_restarts=3,
    restart_delay=60,
    checkpoint_interval=1000
)
```

**Parameters:**
- `max_restarts`: Maximum restart attempts
- `restart_delay`: Delay between restarts (seconds)
- `checkpoint_interval`: Checkpoint frequency (steps)

## CLI Framework

### `run.cli.entrypoint`

Decorator to create CLI entry points.

```python
import nemo_run as run

@run.cli.entrypoint
def train_model(
    model: str,
    learning_rate: float = 0.001,
    epochs: int = 10
):
    """Train a machine learning model."""
    # Training logic
    pass

# CLI usage: python script.py model=resnet learning_rate=0.01 epochs=20
```

**Parameters:**
- `name`: Custom entry point name
- `namespace`: Custom namespace
- `help`: Help text
- `skip_confirmation`: Skip user confirmation
- `enable_executor`: Enable executor functionality
- `type`: Entry point type ("task" or "experiment")

### `run.cli.factory`

Decorator to create factory functions.

```python
import nemo_run as run

@run.cli.factory
def create_default_model() -> run.Config[MyModel]:
    """Create a default model configuration."""
    return run.Config(MyModel, hidden_size=128, num_layers=6)

@run.cli.entrypoint
def train_model(model: run.Config[MyModel]):
    """Train a model."""
    # Training logic
    pass

# CLI usage: python script.py train_model model=create_default_model
```

**Parameters:**
- `target`: Target type or function
- `target_arg`: Specific argument name
- `is_target_default`: Make this the default factory
- `name`: Factory name
- `namespace`: Custom namespace

## Utility Functions

### `run.dryrun_fn()`

Preview execution without running.

```python
import nemo_run as run

# Preview configuration
run.dryrun_fn(config)

# Preview with executor
run.dryrun_fn(config, executor=executor)
```

### `run.autoconvert()`

Automatically convert between configuration types.

```python
import nemo_run as run

# Convert config to partial
partial = run.autoconvert(config, run.Partial)

# Convert partial to config
config = run.autoconvert(partial, run.Config)
```

## Error Handling

### Common Exceptions

```python
import nemo_run as run

try:
    with run.Experiment("test") as experiment:
        experiment.add(config)
        experiment.run()
except run.ExperimentError as e:
    print(f"Experiment failed: {e}")
except run.ConfigurationError as e:
    print(f"Configuration error: {e}")
except run.ExecutionError as e:
    print(f"Execution error: {e}")
```

## Best Practices

### Configuration Patterns

```python
import nemo_run as run
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 6
    dropout: float = 0.1

    def to_run_config(self):
        return run.Config(MyModel, **self.__dict__)

# Usage
config = ModelConfig(hidden_size=256)
run_config = config.to_run_config()
```

### Experiment Patterns

```python
import nemo_run as run

def create_experiment_with_monitoring(name: str, config, executor=None):
    """Create experiment with monitoring setup."""
    with run.Experiment(name) as experiment:
        experiment.add(config, executor=executor, name="main_task")

        # Add monitoring tasks
        experiment.add(
            run.Partial(log_metrics, config),
            name="monitoring",
            dependencies=["main_task"]
        )

        experiment.run()
    return experiment
```

### Error Recovery

```python
import nemo_run as run

def robust_execution(config, max_retries=3):
    """Execute with retry logic."""
    for attempt in range(max_retries):
        try:
            with run.Experiment(f"attempt_{attempt}") as experiment:
                experiment.add(config, name="robust_task")
                experiment.run()
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return False
```

## Next Steps

- Explore [CLI Reference](cli) for command-line interface details
- Check [FAQs](faqs) for common usage questions
- Review [Troubleshooting](troubleshooting) for error resolution
- See [Best Practices](../best-practices/index) for production usage

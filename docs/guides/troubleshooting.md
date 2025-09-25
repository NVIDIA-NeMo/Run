---
description: "Comprehensive troubleshooting guide for NeMo Run covering common issues, error messages, debugging techniques, and solutions."
tags: ["troubleshooting", "debugging", "errors", "solutions", "help", "support"]
categories: ["guides"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "text-only"
---

(troubleshooting)=

# Troubleshoot

This comprehensive guide helps you diagnose and resolve common issues when using NeMo Run. It covers error messages, debugging techniques, and solutions for various scenarios encountered by AI developers and scientists.

## Quick Diagnostic Commands

### Check NeMo Run Status

Run these commands to quickly assess your NeMo Run installation:

```bash
# Check NeMo Run installation
python -c "import nemo_run; print(nemo_run.__version__ if hasattr(nemo_run, '__version__') else 'Version not available')"

# Check environment variables
echo $NEMORUN_HOME

# Check Python environment
python -c "import nemo_run as run; print(dir(run))"
```

### Basic Functionality Test

```python
import nemo_run as run
result = run.run(run.Partial(lambda: "Hello"))
print(result)
```

## Common Issues and Solutions

### Installation Issues

#### Package Installation Problems

**Problem**: Unable to install NeMo Run from GitHub

**Solution**: Use the correct installation method:

```bash
# ✅ Correct installation
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# ❌ Incorrect (this package doesn't exist)
pip install nemo-run
```

**Problem**: Git installation fails

**Solution**: Ensure Git is available and use HTTPS:

```bash
# Check Git installation
git --version

# Use HTTPS instead of SSH
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Or install manually
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install .
```

#### Dependency Conflicts

**Problem**: Version conflicts with dependencies

**Solution**: Install with compatible versions:

```bash
# Install with --no-deps and resolve manually
pip install git+https://github.com/NVIDIA-NeMo/Run.git --no-deps

# Install core dependencies
pip install inquirerpy catalogue fabric fiddle torchx typer rich jinja2 cryptography networkx omegaconf leptonai packaging toml

# Install optional dependencies
pip install "skypilot[kubernetes]>=0.9.2"
pip install "ray[kubernetes]"
```

### Configuration Issues

#### Serialization Errors

**Problem**: Configuration serialization fails

**Solution**: Wrap non-serializable objects in `run.Config`:

```python
# ❌ This will fail
partial = run.Partial(some_function, something=Path("/tmp"))

# ✅ Correct: Wrap in run.Config
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))
```

**Problem**: Complex object serialization

**Solution**: Use proper configuration patterns:

```python
# ✅ Good: Use dataclasses for complex configurations
from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

# Create configuration with validated parameters
config = run.Config(
    train_model,
    config=run.Config(TrainingConfig, learning_rate=0.001, batch_size=32, epochs=10)
)
```

#### Type Safety Issues

**Problem**: Type validation errors

**Solution**: Use proper type hints and validation:

```python
# ✅ Good: Type hints and validation
def create_model_config(
    model_size: int,
    learning_rate: float,
    batch_size: int
) -> run.Config:
    """Create a validated model configuration."""

    # Validation
    if model_size <= 0:
        raise ValueError("model_size must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return run.Config(
        create_model,
        model_size=model_size,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
```

### Execution Issues

#### Resource Problems

**Problem**: Insufficient memory or CPU

**Solution**: Monitor and adjust resources:

```python
import psutil
import nemo_run as run

def check_resources():
    """Check if system has sufficient resources."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)

    if memory.percent > 90:
        raise RuntimeError("Insufficient memory")

    if cpu_percent > 95:
        raise RuntimeError("CPU overloaded")

    return True

# Use in your configuration
def resource_aware_training():
    check_resources()
    # Proceed with training
    return train_model()
```

#### Network Connectivity Issues

**Problem**: Remote execution fails due to network issues

**Solution**: Check connectivity and configuration:

```bash
# Test network connectivity
ping your-cluster.com

# Check SSH connectivity
ssh -o ConnectTimeout=10 user@your-cluster.com

# Verify firewall settings
telnet your-cluster.com 22
```

#### Docker Issues

**Problem**: Docker execution fails

**Solution**: Check Docker installation and permissions:

```bash
# Check Docker installation
docker --version

# Test Docker functionality
docker run hello-world

# Check Docker daemon
sudo systemctl status docker

# Ensure user is in docker group
sudo usermod -aG docker $USER
```

### Management Issues

#### Experiment Metadata Problems

**Problem**: Experiment metadata corruption

**Solution**: Check and repair metadata:

```bash
# Check metadata location
echo $NEMORUN_HOME

# List experiments
ls -la ~/.nemo_run/experiments/

# Clear corrupted metadata (use with caution)
rm -rf ~/.nemo_run/experiments/corrupted_experiment
```

#### Log Retrieval Issues

**Problem**: Unable to access experiment logs

**Solution**: Check log locations and permissions:

```python
import nemo_run as run

# Check experiment status
with run.Experiment("my-experiment") as exp:
    for job in exp.jobs:
        print(f"Job {job.id}: {job.state}")

        # Try to access logs
        try:
            logs = job.logs()
            print(f"Logs available: {len(logs)} lines")
        except Exception as e:
            print(f"Log access failed: {e}")
```

## Advanced Debugging Techniques

### Configuration Debugging

**Problem**: Complex configuration issues

**Solution**: Use configuration inspection tools:

```python
import nemo_run as run
from fiddle import visualization

# Inspect configuration structure
config = run.Config(MyModel, hidden_size=512, num_layers=6)
print(config.to_dict())

# Visualize configuration tree
graph = config.visualize()
graph.render("config_tree", format="png")

# Debug configuration building
try:
    instance = config.build()
except Exception as e:
    print(f"Configuration build failed: {e}")
    # Inspect the configuration more deeply
    print(f"Configuration dict: {config.to_dict()}")
```

### Execution Debugging

**Problem**: Remote execution debugging

**Solution**: Use debugging executors and logging:

```python
import nemo_run as run
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use debug executor for local testing
debug_executor = run.LocalExecutor(
    env_vars={"NEMORUN_DEBUG": "1"},
    retries=0  # Disable retries for debugging
)

# Test configuration locally first
with run.Experiment("debug-test") as exp:
    exp.add(my_config, executor=debug_executor)
    exp.run()
```

### Performance Debugging

**Problem**: Performance issues in distributed execution

**Solution**: Monitor and profile execution:

```python
import time
import nemo_run as run
from contextlib import contextmanager

@contextmanager
def timing_context(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}s")

# Profile execution steps
with timing_context("Configuration"):
    config = run.Config(MyModel, hidden_size=1024)

with timing_context("Execution"):
    with run.Experiment("profiled-run") as exp:
        exp.add(config, executor=run.LocalExecutor())
        exp.run()
```

## Common Error Messages

### Configuration Errors

- `UnserializableValueError` - Non-serializable objects in configuration
- `TypeError` - Type mismatches in configuration
- `ValueError` - Invalid configuration values

### Execution Errors

- `ConnectionError` - Network connectivity issues
- `TimeoutError` - Execution timeouts
- `ResourceError` - Insufficient resources

### Management Errors

- `ExperimentNotFoundError` - Missing experiment metadata
- `LogRetrievalError` - Unable to retrieve logs
- `MetadataError` - Corrupted experiment metadata

## Get Help

### Before Asking for Help

1. **Check this troubleshooting guide** - Your issue might already be covered
2. **Search existing issues** - Check the [GitHub issues](https://github.com/NVIDIA-NeMo/Run/issues)
3. **Provide minimal reproduction** - Create a simple example that demonstrates the problem
4. **Include error messages** - Copy the complete error traceback
5. **Specify your environment** - OS, Python version, NeMo Run version

### When Reporting Issues

Include the following information:

- **NeMo Run version**: `python -c "import nemo_run; print(nemo_run.__version__)"`
- **Python version**: `python --version`
- **Operating system**: `uname -a` (Linux/Mac) or system info
- **Complete error message**: Full traceback and error output
- **Minimal reproduction code**: Simple example that reproduces the issue
- **Expected vs actual behavior**: What you expected vs what happened

## Need More Help?

- Check the [FAQs](../references/faqs) for additional solutions
- Explore the [About section](../about/index) for conceptual information
- Review the [Guides](index) for detailed feature documentation
- Report issues on [GitHub](https://github.com/NVIDIA-NeMo/Run/issues)

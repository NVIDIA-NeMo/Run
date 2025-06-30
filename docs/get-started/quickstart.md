---
description: "Complete quickstart guide for AI developers - Install NeMo Run and run your first ML experiment in minutes."
tags: ["quickstart", "installation", "first-experiment", "ai-developer", "ml-workflow"]
categories: ["get-started"]
---

(quickstart)=

# Quickstart Guide for AI Developers

Get up and running with NeMo Run in under 10 minutes. This guide will walk you through installation, basic configuration, and your first ML experiment.

## Prerequisites

- **Python 3.8+** with pip
- **Git** for cloning repositories
- **Basic ML knowledge** (PyTorch, training loops, etc.)

## Installation

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv nemo-run-env
source nemo-run-env/bin/activate  # Linux/macOS
# or
nemo-run-env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install NeMo Run

```bash
# Core installation
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run as run; print('✅ NeMo Run installed successfully')"
```

### 3. Optional: Install Cloud Dependencies

For cloud execution (AWS, GCP, Azure):

```bash
# SkyPilot for multi-cloud support
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]

# Or install manually
pip install skypilot
```

## Your First Experiment

Let's create a complete ML experiment that demonstrates NeMo Run's core features.

### Step 1: Create Your Training Function

Create a file `train_model.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_model(
    model_size: int = 128,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    data_size: int = 1000
):
    """
    Simple ML training function with configurable parameters.

    Args:
        model_size: Hidden layer size
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        data_size: Size of synthetic dataset
    """
    # Generate synthetic data
    X = torch.randn(data_size, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(data_size, 1) * 0.1

    # Create model
    model = nn.Sequential(
        nn.Linear(10, model_size),
        nn.ReLU(),
        nn.Linear(model_size, 1)
    )

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X), y).item()

    return {
        "final_loss": test_loss,
        "loss_history": losses,
        "model_params": sum(p.numel() for p in model.parameters()),
        "config": {
            "model_size": model_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        }
    }
```

### Step 2: Configure Your Experiment

Create a file `experiment_config.py`:

```python
import nemo_run as run
from train_model import train_model

# Create a partial function with default parameters
train_fn = run.Partial(
    train_model,
    model_size=128,
    learning_rate=0.001,
    batch_size=32,
    epochs=10
)

# Create different configurations for experimentation
configs = [
    run.Config(train_fn, model_size=64, learning_rate=0.01),
    run.Config(train_fn, model_size=128, learning_rate=0.001),
    run.Config(train_fn, model_size=256, learning_rate=0.0001),
]

print("✅ Experiment configurations created")
print(f"Number of configurations: {len(configs)}")
```

### Step 3: Run Your First Experiment

Create a file `run_experiment.py`:

```python
import nemo_run as run
from experiment_config import configs

# Create an experiment to manage multiple runs
with run.Experiment("quickstart-experiment") as experiment:
    # Add all configurations to the experiment
    for i, config in enumerate(configs):
        print(f"\n🚀 Adding configuration {i+1}/{len(configs)}")

        # Add the task to the experiment
        experiment.add(
            config,
            executor=run.LocalExecutor(),
            name=f"config_{i+1}"
        )

    # Run all tasks in the experiment
    print("\n🚀 Launching experiment...")
    experiment.run()

    # Get results from the experiment
    results = []
    for i, job in enumerate(experiment.jobs):
        if job.state == run.AppState.SUCCEEDED:
            # For local execution, we can get the result directly
            # In a real scenario, you'd access logs and artifacts
            print(f"✅ Configuration {i+1} completed successfully")
            print(f"   Job ID: {job.id}")
            print(f"   Status: {job.state}")
        else:
            print(f"❌ Configuration {i+1} failed with status: {job.state}")

print("\n🎉 Your first NeMo Run experiment is complete!")
```

### Alternative: Simple Single Task Execution

If you prefer a simpler approach without experiment management:

```python
import nemo_run as run
from experiment_config import configs

# Run each configuration individually
results = []
for i, config in enumerate(configs):
    print(f"\n🚀 Running configuration {i+1}/{len(configs)}")

    # Run the task directly
    result = run.run(config, executor=run.LocalExecutor())
    results.append(result)

    print(f"✅ Configuration {i+1} completed")
    print(f"   Result: {result}")

print("\n🎉 All configurations completed!")
```

### Step 4: Execute the Experiment

```bash
# Run the experiment
python run_experiment.py
```

You should see output similar to:

```
✅ Experiment configurations created
Number of configurations: 3

🚀 Adding configuration 1/3
🚀 Adding configuration 2/3
🚀 Adding configuration 3/3

🚀 Launching experiment...
Epoch 0: Loss = 0.1234
Epoch 2: Loss = 0.0987
...
✅ Configuration 1 completed successfully
   Job ID: config_1
   Status: SUCCEEDED
✅ Configuration 2 completed successfully
   Job ID: config_2
   Status: SUCCEEDED
✅ Configuration 3 completed successfully
   Job ID: config_3
   Status: SUCCEEDED

🎉 Your first NeMo Run experiment is complete!
```

## Next Steps

### Explore Advanced Features

1. **Remote Execution**: Try running on different backends
   ```python
   # Docker execution
   executor = run.DockerExecutor(image="pytorch/pytorch:latest")

   # Slurm execution (if available)
   executor = run.SlurmExecutor(partition="gpu", gpus_per_node=1)
   ```

2. **Experiment Tracking**: Add metrics and logging
   ```python
   # In your training function, you can return metrics
   return {
       "loss": loss.item(),
       "accuracy": accuracy,
       "learning_rate": learning_rate,
       "epoch": epoch
   }
   ```

3. **Hyperparameter Tuning**: Create parameter sweeps
   ```python
   # Grid search
   with run.Experiment("hyperparameter-sweep") as exp:
       for lr in [0.001, 0.01, 0.1]:
           for batch_size in [16, 32, 64]:
               config = run.Config(train_fn, learning_rate=lr, batch_size=batch_size)
               exp.add(config, executor=run.LocalExecutor())
       exp.run()
   ```

### Learn More

- **Configuration Guide**: Master `run.Config` and `run.Partial`
- **Execution Guide**: Explore different executors and backends
- **Management Guide**: Advanced experiment tracking and management
- **Tutorials**: Hands-on examples and advanced workflows

## Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'nemo_run'`
```bash
# Ensure virtual environment is activated
source nemo-run-env/bin/activate
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

**CUDA Issues**: If you encounter CUDA-related errors
```python
# Force CPU execution
import torch
torch.cuda.is_available = lambda: False
```

**Memory Issues**: For large models or datasets
```python
# Use smaller batch sizes or model sizes
config = run.Config(train_fn, batch_size=16, model_size=64)
```

## What You've Learned

✅ **Configuration Management**: Using `run.Config` and `run.Partial` for flexible parameter management

✅ **Experiment Tracking**: Creating and managing experiments with `run.Experiment`

✅ **Local Execution**: Running ML workloads with `run.LocalExecutor`

✅ **Result Collection**: Accessing and analyzing experiment results

✅ **Basic Workflow**: Complete ML experiment lifecycle with NeMo Run

You're now ready to scale your ML experiments across different environments and build more complex workflows!

## Need Help?

- **Documentation**: Explore the [Configuration](../guides/configuration), [Execution](../guides/execution), and [Management](../guides/management) guides
- **Examples**: Check out the [tutorials](tutorials.md) for more advanced examples
- **Reference**: Consult the [CLI reference](../reference/cli) and [glossary](../reference/glossary) for detailed information

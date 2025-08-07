---
description: "Create your first ML experiment with NeMo Run - a step-by-step tutorial for beginners"
categories: ["tutorials"]
tags: ["first-experiment", "beginner", "quickstart", "installation", "configuration", "local-execution"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(first-experiment)=

# Your First NeMo Run Experiment

Welcome to NeMo Run! This tutorial will guide you through creating and running your first machine learning experiment. By the end of this tutorial, you'll understand the core concepts and be ready to explore more advanced features.

## What You'll Learn

- Install and configure NeMo Run
- Create your first configuration
- Run a simple training experiment
- Understand basic NeMo Run concepts

## Prerequisites

- **Python 3.10+** with pip
- **Basic ML knowledge** (PyTorch, training loops)
- **A computer** with Python installed

## Step 1: Installation

First, let's install NeMo Run:

```bash
# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run as run; print('✅ NeMo Run ready')"
```

If you see the success message, you're ready to proceed!

## Step 2: Basic Setup

Let's configure your environment (optional but recommended):

```bash
# Set custom home directory (optional)
export NEMORUN_HOME=~/.nemo_run

# Verify the directory was created
ls ~/.nemo_run
```

This directory will store your experiment metadata and logs.

## Step 3: Create Your Training Function

Let's start with a simple neural network training function:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nemo_run as run

def train_model(
    model_size: int = 128,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a simple neural network for regression."""

    # Generate synthetic data
    X = torch.randn(1000, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1

    # Create model
    model = nn.Sequential(
        nn.Linear(10, model_size),
        nn.ReLU(),
        nn.Linear(model_size, 1)
    )

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Return results
    return {
        "final_loss": loss.item(),
        "model_size": model_size,
        "learning_rate": learning_rate,
        "epochs": epochs
    }
```

Save this code in a file called `train_model.py`. This function:
- Creates synthetic data for demonstration
- Defines a simple neural network
- Trains the model with configurable parameters
- Returns training results

## Step 4: Create Your First Configuration

Now let's configure this training function with NeMo Run:

```python
import nemo_run as run
from dataclasses import dataclass

# Define a configuration class for type safety
@dataclass
class TrainingConfig:
    model_size: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

# Create your first NeMo Run configuration
config = run.Config(
    train_model,
    model_size=256,      # Override default
    learning_rate=0.001,
    batch_size=64,       # Override default
    epochs=20            # Override default
)

print("✅ Configuration created successfully!")
```

## Step 5: Run Your First Experiment

Now let's execute your experiment:

```python
# Simple execution
result = run.run(config)
print(f"Training completed!")
print(f"Final loss: {result['final_loss']:.4f}")
print(f"Model size: {result['model_size']}")
print(f"Learning rate: {result['learning_rate']}")
print(f"Epochs: {result['epochs']}")
```

Congratulations! You've just run your first NeMo Run experiment. Let's break down what happened:

1. **Configuration**: NeMo Run created a type-safe configuration for your training function
2. **Execution**: The system executed your function with the specified parameters
3. **Results**: You received structured results with all the training information

## Step 6: Experiment with Multiple Configurations

Let's try running multiple experiments with different configurations:

```python
# Create an experiment with multiple configurations
with run.Experiment(name="hyperparameter_sweep") as exp:

    # Small model configuration
    exp.add(
        run.Config(train_model, model_size=128, learning_rate=0.001),
        name="small_model"
    )

    # Large model configuration
    exp.add(
        run.Config(train_model, model_size=512, learning_rate=0.0005),
        name="large_model"
    )

    # High learning rate configuration
    exp.add(
        run.Config(train_model, model_size=256, learning_rate=0.01),
        name="high_lr"
    )

# Execute all experiments
results = exp.run()

# Compare results
for name, result in results.items():
    print(f"{name}: Loss = {result['final_loss']:.4f}")
```

This demonstrates how NeMo Run can:
- Run multiple experiments in parallel
- Compare different configurations
- Organize results by experiment name

## Understanding the Concepts

### Configuration (`run.Config`)

`run.Config` creates type-safe, serializable configurations:

```python
# Basic configuration
config = run.Config(train_model, learning_rate=0.001)

# Configuration with validation
@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 6

    def __post_init__(self):
        assert self.hidden_size > 0, "hidden_size must be positive"

config = run.Config(train_model, model_config=ModelConfig(hidden_size=768))
```

### Execution (`run.run()`)

`run.run()` executes your configured function:

```python
# Simple execution
result = run.run(config)

# Execution with custom executor
result = run.run(config, executor=run.LocalExecutor())
```

### Experiments (`run.Experiment`)

`run.Experiment` manages multiple related tasks:

```python
with run.Experiment("my_experiment") as exp:
    exp.add(config1, name="task1")
    exp.add(config2, name="task2")
    results = exp.run()
```

## Next Steps

Now that you've completed your first experiment, you're ready to explore:

1. **[Configuring Your First Model](configuring-your-first-model)** - Learn advanced configuration patterns
2. **[Running Your First Experiment](running-your-first-experiment)** - Master experiment management and debugging
3. **[Examples](../examples/index.md)** - See more complex examples
4. **[Reference](../references/index.md)** - Explore the complete API

## Troubleshooting

### Common Issues

**Import Error**: If you get an import error, make sure NeMo Run is installed:
```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

**Configuration Error**: If your configuration fails, check your parameter types:
```python
# Make sure parameters match your function signature
config = run.Config(train_model, model_size=256)  # ✅ Correct
config = run.Config(train_model, model_size="256")  # ❌ Wrong type
```

**Execution Error**: If execution fails, check your function:
```python
# Make sure your function returns a dictionary
def train_model(...):
    # ... training code ...
    return {"final_loss": loss.item()}  # ✅ Correct
    # return loss.item()  # ❌ Wrong return type
```

## Summary

You've successfully:
- ✅ Installed NeMo Run
- ✅ Created your first configuration
- ✅ Run a simple experiment
- ✅ Executed multiple experiments
- ✅ Understood basic concepts

You're now ready to explore more advanced features and build complex ML workflows with NeMo Run!
